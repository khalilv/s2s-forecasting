"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import contextlib
import dataclasses
from datetime import timedelta
from typing import Optional, Tuple, Union

import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)

from s2s.irvine.model.batch import Batch
from s2s.irvine.model.decoder import WeatherDecoder
from s2s.irvine.model.encoder import WeatherEncoder
from s2s.irvine.model.swin3d import BasicLayer3D, Swin3DTransformerBackbone

__all__ = ["WeatherForecast"]


class WeatherForecast(torch.nn.Module):
    """The weather forecast model.
    """

    def __init__(
        self,
        surf_vars: Tuple[str, ...] = ("t2m", "10u", "10v", "msl"),
        static_vars: Tuple[str, ...] = ("lsm", "z", "slt"),
        atmos_vars: Tuple[str, ...] = ("z", "u", "v", "t", "q"),
        atmos_levels: Tuple[Union[int, float], ...] = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000),
        window_size: Tuple[int, int, int] = (2, 6, 12),
        encoder_depths: Tuple[int, ...] = (2, 4, 2),
        encoder_num_heads: Tuple[int, ...] = (4, 8, 16),
        decoder_depths: Tuple[int, ...] = (2, 4, 2),
        decoder_num_heads: Tuple[int, ...] = (16, 8, 4),
        latent_levels: int = 4,
        patch_size: int = 2,
        embed_dim: int = 512,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        drop_rate: float = 0.0,
        history_size: int = 2,
        delta_time: int = 6,
        autocast: bool = False,
        temporal_attention: bool = False,
    ) -> None:
        """Construct an instance of the model.

        Args:
            surf_vars (Tuple[str, ...], optional): All surface-level variables supported by the
                model.
            static_vars (Tuple[str, ...], optional): All static variables supported by the
                model.
            atmos_vars (Tuple[str, ...], optional): All atmospheric variables supported by the
                model.
            atmos_levels (Tuple[int | float, ...]): All supported pressure levels.
            window_size (Tuple[int, int, int], optional): Vertical height, height, and width of the
                window of the underlying Swin transformer.
            encoder_depths (Tuple[int, ...], optional): Number of blocks in each encoder layer.
            encoder_num_heads (Tuple[int, ...], optional): Number of attention heads in each encoder
                layer. The dimensionality doubles after every layer. To keep the dimensionality of
                every head constant, you want to double the number of heads after every layer. The
                dimensionality of attention head of the first layer is determined by `embed_dim`
                divided by the value here.
            decoder_depths (Tuple[int, ...], optional): Number of blocks in each decoder layer.
                Generally, you want this to be the reversal of `encoder_depths`.
            decoder_num_heads (Tuple[int, ...], optional): Number of attention heads in each decoder
                layer. Generally, you want this to be the reversal of `encoder_num_heads`.
            latent_levels (int, optional): Number of latent pressure levels. Must be >= 2. Defaults to `4`.
            patch_size (int, optional): Patch size. Defaults to `2`.
            embed_dim (int, optional): Patch embedding dimension.
            num_heads (int, optional): Number of attention heads in the aggregation and
                deaggregation blocks.
            mlp_ratio (float, optional): Hidden dim. to embedding dim. ratio for MLPs.
            drop_rate (float, optional): Drop-out rate.
            drop_path (float, optional): Drop-path rate.
            history_size (int, optional): Number of history steps in the model.
            delta_time (int, optional): The default forecast horizon in hours, representing how far ahead to predict at each step.
            autocast (bool, optional): Set to `True` to reduce memory footprint. Defaults to `False`.
            temporal_attention (bool): If true perform explicit cross attention over the temporal dimension. Defaults to `False`
        """
        super().__init__()
        self.surf_vars = surf_vars
        self.atmos_vars = atmos_vars
        self.patch_size = patch_size
        self.timedelta = timedelta(hours=delta_time)
        self.autocast = autocast
        self.history_size = history_size
        self.temporal_attention = temporal_attention

        self.encoder = WeatherEncoder(
            surf_vars=surf_vars,
            static_vars=static_vars,
            atmos_vars=atmos_vars,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            drop_rate=drop_rate,
            latent_levels=latent_levels,
            history_size=history_size,
            temporal_attention=self.temporal_attention, 
        )

        self.backbone = Swin3DTransformerBackbone(
            window_size=window_size,
            encoder_depths=encoder_depths,
            encoder_num_heads=encoder_num_heads,
            decoder_depths=decoder_depths,
            decoder_num_heads=decoder_num_heads,
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path,
            drop_rate=drop_rate,
        )

        self.decoder = WeatherDecoder(
            surf_vars=surf_vars,
            atmos_vars=atmos_vars,
            atmos_levels=atmos_levels,
            patch_size=patch_size,
            embed_dim=embed_dim * 2, # concatenation at the backbone end doubles the dim.
            num_heads=num_heads,
        )

    def forward(self, batch: Batch, lead_time: Optional[timedelta] = None) -> Batch:
        """Forward pass.

        Args:
            batch (:class:`Batch`): Batch to run the model on.
            lead_time: (timedelta, optional): if provided, model will forecast for this lead time, 
                otherwise it will use the default timedelta provided during initialization 

        Returns:
            :class:`Batch`: Prediction for the batch.
        """
        # Get the first parameter. We'll derive the data type and device from this parameter.
        p = next(self.parameters())
        batch = batch.type(p.dtype)
        batch = batch.crop(patch_size=self.patch_size)
        batch = batch.to(p.device)

        H, W = batch.spatial_shape
        patch_res = (
            self.encoder.latent_levels,
            H // self.encoder.patch_size,
            W // self.encoder.patch_size,
        )

        # Insert batch and history dimension for static variables.
        B, T = next(iter(batch.surf_vars.values())).shape[:2]
        batch = dataclasses.replace(
            batch,
            static_vars={k: v[None, None].repeat(B, T, 1, 1) for k, v in batch.static_vars.items()},
        )

        x = self.encoder(
            batch,
            lead_time=lead_time if lead_time else self.timedelta,
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16) if self.autocast else contextlib.nullcontext():
            x = self.backbone(
                x,
                lead_time=lead_time if lead_time else self.timedelta,
                patch_res=patch_res,
                rollout_step=batch.metadata.rollout_step,
            )
        pred = self.decoder(
            x,
            batch,
            lead_time=lead_time if lead_time else self.timedelta,
            patch_res=patch_res,
        )

        # remove batch and history dimension from static variables.
        pred = dataclasses.replace(
            pred,
            static_vars={k: v[0, 0] for k, v in batch.static_vars.items()},
        )

        # insert history dimension in prediction. the time should already be right.
        pred = dataclasses.replace(
            pred,
            surf_vars={k: v[:, None] for k, v in pred.surf_vars.items()},
            atmos_vars={k: v[:, None] for k, v in pred.atmos_vars.items()},
        )

        return pred

    def load_checkpoint_local(self, path: str, strict: bool = True) -> None:
        """Load a checkpoint directly from a file.

        Args:
            path (str): Path to the checkpoint.
            strict (bool, optional): Error if the model parameters are not exactly equal to the
                parameters in the checkpoint. Defaults to `True`.
        """
        print(f"Loading checkpoint weights from {path}")
        # Assume that all parameters are either on the CPU or on the GPU.
        device = next(self.parameters()).device

        d = torch.load(path, map_location=device, weights_only=True)

        #if loading from lightning checkpoint, isolate the state_dict
        if 'state_dict' in d.keys():
            d = d['state_dict']
        
        for k, v in list(d.items()):
            if k.startswith("net."):
                del d[k]
                d[k[4:]] = v
        
        self.load_state_dict(d, strict=strict)
    

    def configure_activation_checkpointing(self):
        """Configure activation checkpointing to reduce memory footprint.
        """
        apply_activation_checkpointing(self, check_fn=lambda x: isinstance(x, BasicLayer3D))