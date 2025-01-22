"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

from datetime import timedelta
from typing import Optional, Tuple

import torch
from einops import rearrange
from torch import nn

from s2s.aurora.batch import Batch
from s2s.aurora.model.fourier import (
    absolute_time_expansion,
    lead_time_expansion,
    levels_expansion,
    pos_expansion,
    scale_expansion,
    variable_expansion
)
from s2s.aurora.model.patchembed import LevelPatchEmbed, VariablePatchEmbed
from s2s.aurora.model.perceiver import MLP, PerceiverResampler
from s2s.aurora.model.posencoding import pos_scale_enc
from s2s.aurora.model.util import (
    check_lat_lon_dtype,
    init_weights,
)
from s2s.utils.data_utils import AURORA_VARIABLE_CODES

__all__ = ["Perceiver3DEncoder"]


class Perceiver3DEncoder(nn.Module):
    """Multi-scale multi-source multi-variable encoder based on the Perceiver architecture."""

    def __init__(
        self,
        surf_vars: Tuple[str, ...],
        static_vars: Optional[Tuple[str, ...]],
        atmos_vars: Tuple[str, ...],
        patch_size: int = 4,
        latent_levels: int = 8,
        embed_dim: int = 1024,
        num_heads: int = 16,
        head_dim: int = 64,
        drop_rate: float = 0.1,
        depth: int = 2,
        mlp_ratio: float = 4.0,
        max_history_size: int = 2,
        perceiver_ln_eps: float = 1e-5,
        latent_atmos_vars: int = 1,
        latent_surf_vars: int = 1,
        temporal_attention: bool = True
    ) -> None:
        """Initialise.

        Args:
            surf_vars (Tuple[str, ...]): All supported surface-level variables.
            static_vars (Optional[Tuple[str, ...]]): All supported static variables.
            atmos_vars (Tuple[str, ...]): All supported atmospheric variables.
            patch_size (int, optional): Patch size. Defaults to `4`.
            latent_levels (int): Number of latent pressure levels. Defaults to `8`.
            embed_dim (int, optional): Embedding dim. used in the aggregation blocks. Defaults
                to `1024`.
            num_heads (int, optional): Number of attention heads used in aggregation blocks.
                Defaults to `16`.
            head_dim (int, optional): Dimension of attention heads used in aggregation blocks.
                Defaults to `64`.
            drop_rate (float, optional): Drop out rate for input patches. Defaults to `0.1`.
            depth (int, optional): Number of Perceiver cross-attention and feed-forward blocks.
                Defaults to `2`.
            mlp_ratio (float, optional): Ratio of hidden dimensionality to embedding dimensionality
                for MLPs. Defaults to `4.0`.
            max_history_size (int, optional): Maximum number of history steps to consider. Defaults
                to `2`.
            perceiver_ln_eps (float, optional): Epsilon value for layer normalisation in the
                Perceiver. Defaults to 1e-5.
        """
        super().__init__()
        # We treat the static variables as surface variables in the model.
        surf_vars = surf_vars + static_vars if static_vars is not None else surf_vars

        self.drop_rate = drop_rate
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.temporal_attention = temporal_attention
        self.atmos_vars = atmos_vars
        self.surf_vars = surf_vars

        assert latent_levels > 1, "At least two latent levels are required."
        self.latent_levels = latent_levels
        self.atmos_latents = nn.Parameter(torch.randn(latent_levels - 1, embed_dim)) # One latent level will be used by the surface level.

        # Learnable embedding to encode the surface level.
        self.surf_level_encoding = nn.Parameter(torch.randn(embed_dim))
        self.surf_mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout=drop_rate)
        self.surf_norm = nn.LayerNorm(embed_dim)

        # Position, scale, and time embeddings
        self.pos_embed = nn.Linear(embed_dim, embed_dim)
        self.scale_embed = nn.Linear(embed_dim, embed_dim)
        self.lead_time_embed = nn.Linear(embed_dim, embed_dim)
        self.absolute_time_embed = nn.Linear(embed_dim, embed_dim)
        self.atmos_levels_embed = nn.Linear(embed_dim, embed_dim)

        # Patch embeddings
        assert max_history_size > 0, "At least one history step is required."
        if self.temporal_attention:
            self.surf_token_embeds = VariablePatchEmbed(
                self.surf_vars,
                patch_size,
                embed_dim,
            )
            self.atmos_token_embeds = VariablePatchEmbed(
                self.atmos_vars,
                patch_size,
                embed_dim,
            )
            self.surf_var_embed = nn.Linear(embed_dim, embed_dim)
            self.atmos_var_embed = nn.Linear(embed_dim, embed_dim)        
            assert latent_atmos_vars > 0, "At least one latent atmospheric variable is required."
            assert latent_surf_vars > 0, "At least one latent surface variable is required."
            self.atmos_var_latents = nn.Parameter(torch.randn(latent_atmos_vars, embed_dim))
            self.surf_var_latents = nn.Parameter(torch.randn(latent_surf_vars, embed_dim))
            self.atmos_var_agg = PerceiverResampler(
                latent_dim=embed_dim,
                context_dim=embed_dim,
                depth=depth,
                head_dim=head_dim,
                num_heads=num_heads,
                drop=drop_rate,
                mlp_ratio=mlp_ratio,
                ln_eps=perceiver_ln_eps,
            )
            self.surf_var_agg = PerceiverResampler(
                latent_dim=embed_dim,
                context_dim=embed_dim,
                depth=depth,
                head_dim=head_dim,
                num_heads=num_heads,
                drop=drop_rate,
                mlp_ratio=mlp_ratio,
                ln_eps=perceiver_ln_eps,
            )
        else:
            self.surf_token_embeds = LevelPatchEmbed(
                self.surf_vars,
                patch_size,
                embed_dim,
                max_history_size
            )
            self.atmos_token_embeds = LevelPatchEmbed(
                self.atmos_vars,
                patch_size,
                embed_dim,
                max_history_size
            )

        # Learnable pressure level aggregation
        self.level_agg = PerceiverResampler(
            latent_dim=embed_dim,
            context_dim=embed_dim,
            depth=depth,
            head_dim=head_dim,
            num_heads=num_heads,
            drop=drop_rate,
            mlp_ratio=mlp_ratio,
            ln_eps=perceiver_ln_eps,
        )

        # Drop patches after encoding.
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.apply(init_weights)

        # Initialize the latents like in the Huggingface implementation of the Perceiver:
        #
        #   https://github.com/huggingface/transformers/blob/v4.36.1/src/transformers/models/perceiver/modeling_perceiver.py#L628
        #
        torch.nn.init.trunc_normal_(self.atmos_latents, std=0.02)
        torch.nn.init.trunc_normal_(self.surf_level_encoding, std=0.02)
        if self.temporal_attention:
            torch.nn.init.trunc_normal_(self.atmos_var_latents, std=0.02)
            torch.nn.init.trunc_normal_(self.surf_var_latents, std=0.02)

    def aggregate_levels(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate pressure level information.

        Args:
            x (torch.Tensor): Tensor of shape `(B, C_A, L, D)` where `C_A` refers to the number
                of pressure levels.

        Returns:
            torch.Tensor: Tensor of shape `(B, C, L, D)` where `C` is the number of
                aggregated pressure levels.
        """
        B, _, L, _ = x.shape
        latents = self.atmos_latents.to(dtype=x.dtype)
        latents = latents.unsqueeze(1).expand(B, -1, L, -1)  # (C_A, D) to (B, C_A, L, D)

        x = torch.einsum("bcld->blcd", x)
        x = x.flatten(0, 1)  # (B * L, C_A, D)
        latents = torch.einsum("bcld->blcd", latents)
        latents = latents.flatten(0, 1)  # (B * L, C_A, D)

        self.level_agg.to(dtype=x.dtype)
        x = self.level_agg(latents, x)  # (B * L, C, D)
        x = x.unflatten(dim=0, sizes=(B, L))  # (B, L, C, D)
        x = torch.einsum("blcd->bcld", x)  # (B, C, L, D)
        return x
    
    def aggregate_atmos_variables(self, x_atmos: torch.Tensor) -> torch.Tensor:
        """Aggregate atmospheric variable information across time.

        Args:
            x (torch.Tensor): Tensor of shape `(B, T*V_a, L, D)` where `V_a` refers to the number
                of atmospheric variables and `T` refers to the number of timestamps for each 
                variable.

        Returns:
            torch.Tensor: Tensor of shape `(B, R_a, L, D)` where `R_a` is the number of
                aggregated atmospheric variables across time.
        """
        B, _, L, _ = x_atmos.shape
        latents = self.atmos_var_latents.to(dtype=x_atmos.dtype)
        latents = latents.unsqueeze(1).expand(B, -1, L, -1)  # (T*V_a, D) to (B, T*V_a, L, D)

        x_atmos = torch.einsum("bcld->blcd", x_atmos)
        x_atmos = x_atmos.flatten(0, 1)  # (B * L, T*V_a, D)
        latents = torch.einsum("bcld->blcd", latents)
        latents = latents.flatten(0, 1)  # (B * L, T*V_a, D)

        self.atmos_var_agg.to(dtype=x_atmos.dtype)
        x_atmos = self.atmos_var_agg(latents, x_atmos)  # (B * L, R_a, D)
        x_atmos = x_atmos.unflatten(dim=0, sizes=(B, L))  # (B, L, R_a, D)
        x_atmos = torch.einsum("blcd->bcld", x_atmos)  # (B, R_a, L, D)
        return x_atmos
    
    def aggregate_surf_variables(self, x_surf: torch.Tensor) -> torch.Tensor:
        """Aggregate surface variable information across time.

        Args:
            x (torch.Tensor): Tensor of shape `(B, T*V_s, L, D)` where `V_s` refers to the number
                of surface variables and `T` refers to the number of timestamps for each 
                variable.

        Returns:
            torch.Tensor: Tensor of shape `(B, R_s, L, D)` where `R_s` is the number of
                aggregated surface variables across time.
        """
        B, _, L, _ = x_surf.shape
        latents = self.surf_var_latents.to(dtype=x_surf.dtype)
        latents = latents.unsqueeze(1).expand(B, -1, L, -1)  # (T*V_s, D) to (B, T*V_s, L, D)

        x_surf = torch.einsum("bcld->blcd", x_surf)
        x_surf = x_surf.flatten(0, 1)  # (B * L, T*V_s, D)
        latents = torch.einsum("bcld->blcd", latents)
        latents = latents.flatten(0, 1)  # (B * L, T*V_s, D)

        self.surf_var_agg.to(dtype=x_surf.dtype)
        x_surf = self.surf_var_agg(latents, x_surf)  # (B * L, R_s, D)
        x_surf = x_surf.unflatten(dim=0, sizes=(B, L))  # (B, L, R_s, D)
        x_surf = torch.einsum("blcd->bcld", x_surf)  # (B, R_s, L, D)
        return x_surf


    def forward(self, batch: Batch, lead_time: timedelta) -> torch.Tensor:
        """Peform encoding.

        Args:
            batch (:class:`.Batch`): Batch to encode.
            lead_time (timedelta): Lead time.

        Returns:
            torch.Tensor: Encoding of shape `(B, L, D)`.
        """
        surf_vars = tuple(batch.surf_vars.keys())
        static_vars = tuple(batch.static_vars.keys())
        atmos_vars = tuple(batch.atmos_vars.keys())
        atmos_levels = batch.metadata.atmos_levels

        x_surf = torch.stack(tuple(batch.surf_vars.values()), dim=2)
        x_static = torch.stack(tuple(batch.static_vars.values()), dim=2)
        x_atmos = torch.stack(tuple(batch.atmos_vars.values()), dim=2)

        B, T, _, C, H, W = x_atmos.size()
        assert x_surf.shape[:2] == (B, T), f"Expected shape {(B, T)}, got {x_surf.shape[:2]}."

        if static_vars is None:
            assert x_static is None, "Static variables given, but not configured."
        else:
            assert x_static is not None, "Static variables not given."
            x_static = x_static.expand((B, T, -1, -1, -1))
            x_surf = torch.cat((x_surf, x_static), dim=2)  # (B, T, V_S + V_Static, H, W)
            surf_vars = surf_vars + static_vars

        lat, lon = batch.metadata.lat, batch.metadata.lon
        check_lat_lon_dtype(lat, lon)
        lat, lon = lat.to(dtype=torch.float32), lon.to(dtype=torch.float32)
        assert lat.shape[0] == H and lon.shape[-1] == W

        # Patch embed the surface level.
        if self.temporal_attention:
            x_surf = rearrange(x_surf, "b t v h w -> (b t) v h w")
            x_surf = self.surf_token_embeds(x_surf, surf_vars)  # (B, V, L, D)
            x_surf = rearrange(x_surf, "(b t) v l d -> b t v l d", b=B, t=T)
        else:
            x_surf = rearrange(x_surf, "b t v h w -> b v t h w")
            x_surf = self.surf_token_embeds(x_surf, surf_vars)  # (B, L, D)
        dtype = x_surf.dtype  # When using mixed precision, we need to keep track of the dtype.

        # Patch embed the atmospheric levels.
        if self.temporal_attention:
            x_atmos = rearrange(x_atmos, "b t v c h w -> (b t c) v h w")
            x_atmos = self.atmos_token_embeds(x_atmos, atmos_vars)
            x_atmos = rearrange(x_atmos, "(b t c) v l d -> b t c v l d", b=B, c=C, t=T)
        else:
            x_atmos = rearrange(x_atmos, "b t v c h w -> (b c) v t h w")
            x_atmos = self.atmos_token_embeds(x_atmos, atmos_vars)
            x_atmos = rearrange(x_atmos, "(b c) l d -> b c l d", b=B, c=C)

        if self.temporal_attention:

            #add surface variable embedding
            surf_vars_tensor = torch.tensor([AURORA_VARIABLE_CODES[var] for var in surf_vars], device=x_surf.device)
            surf_vars_encode = variable_expansion(surf_vars_tensor, self.embed_dim).to(dtype=dtype)
            surf_vars_embed = self.surf_var_embed(surf_vars_encode)[None, None, :, None, :]
            x_surf = x_surf + surf_vars_embed

            #add atmospheric variable embedding
            atmos_vars_tensor = torch.tensor([AURORA_VARIABLE_CODES[var] for var in atmos_vars], device=x_atmos.device)
            atmos_vars_encode = variable_expansion(atmos_vars_tensor, self.embed_dim).to(dtype=dtype)
            atmos_vars_embed = self.atmos_var_embed(atmos_vars_encode)[None, None, None, :, None, :]
            x_atmos = x_atmos + atmos_vars_embed

            #add absolute time embedding
            absolute_timestamps = [[dt.timestamp() / 3600  for dt in batch] for batch in batch.metadata.time]
            absolute_time_tensor = torch.tensor(absolute_timestamps, dtype=torch.float32, device=x_surf.device)
            absolute_time_encode = absolute_time_expansion(absolute_time_tensor, self.embed_dim).to(dtype=dtype)
            absolute_time_embed = self.absolute_time_embed(absolute_time_encode)
            x_atmos = x_atmos + absolute_time_embed[:, :, None, None, None, :]
            x_surf = x_surf + absolute_time_embed[:, :, None, None, :]

            x_atmos = rearrange(x_atmos, "b t c v l d -> (b c) (t v) l d")
            x_atmos = self.aggregate_atmos_variables(x_atmos)  # (B*C, T*V_a, L, D) to (B*C, R_a, L, D)
            x_atmos = rearrange(x_atmos, "(b c) r l d -> b c r l d", b=B, c=C)

            x_surf = rearrange(x_surf, "b t v l d -> b (t v) l d")
            x_surf = self.aggregate_surf_variables(x_surf)  # (B, T*V_s, L, D) to (B, R_s, L, D)
            
            if x_surf.shape[1] == 1:
                x_surf = x_surf.squeeze(1) # (B, L, D)
            else:
                raise NotImplementedError(f'Number of latent surface variables R_s > 1 is not supported yet')
            
            if x_atmos.shape[2] == 1:
                x_atmos = x_atmos.squeeze(2) # (B, C, L, D)
            else:
                raise NotImplementedError(f'Number of latent atmospheric variables R_a > 1 is not supported yet')
            
        # Add surface level encoding. This helps the model distinguish between surface and
        # atmospheric levels.
        x_surf = x_surf + self.surf_level_encoding[None, None, :].to(dtype=dtype)
        # Since the surface level is not aggregated, we add a Perceiver-like MLP only.
        self.surf_norm.to(dtype=dtype)
        self.surf_mlp.to(dtype=dtype)
        x_surf = x_surf + self.surf_norm(self.surf_mlp(x_surf))

        # Add atmospheric pressure encoding of shape (C_A, D) and subsequent embedding.
        atmos_levels_tensor = torch.tensor(atmos_levels, device=x_atmos.device)
        atmos_levels_encode = levels_expansion(atmos_levels_tensor, self.embed_dim).to(dtype=dtype)
        atmos_levels_embed = self.atmos_levels_embed(atmos_levels_encode)[None, :, None, :]
        x_atmos = x_atmos + atmos_levels_embed  # (B, C_A, L, D)

        # Aggregate over pressure levels.
        x_atmos = self.aggregate_levels(x_atmos)  # (B, C_A, L, D) to (B, C, L, D)

        # Concatenate the surface level with the amospheric levels.
        x = torch.cat((x_surf.unsqueeze(1), x_atmos), dim=1)

        # Add position and scale embeddings to the 3D tensor.
        pos_encode, scale_encode = pos_scale_enc(
            self.embed_dim,
            lat,
            lon,
            self.patch_size,
            pos_expansion=pos_expansion,
            scale_expansion=scale_expansion,
        )
        # Encodings are (L, D).
        pos_encode = self.pos_embed(pos_encode[None, None, :].to(dtype=dtype))
        scale_encode = self.scale_embed(scale_encode[None, None, :].to(dtype=dtype))
        x = x + pos_encode + scale_encode

        # Flatten the tokens.
        x = x.reshape(B, -1, self.embed_dim)  # (B, C + 1, L, D) to (B, L', D)

        # Add lead time embedding.
        lead_hours = lead_time.total_seconds() / 3600
        lead_times = lead_hours * torch.ones(B, dtype=dtype, device=x.device)
        lead_time_encode = lead_time_expansion(lead_times, self.embed_dim).to(dtype=dtype)
        lead_time_emb = self.lead_time_embed(lead_time_encode)  # (B, D)
        x = x + lead_time_emb.unsqueeze(1)  # (B, L', D) + (B, 1, D)

        if not self.temporal_attention:
            # Add absolute time embedding.
            absolute_times_list = [t[-1].timestamp() / 3600 for t in batch.metadata.time]  # Times in hours
            absolute_times = torch.tensor(absolute_times_list, dtype=torch.float32, device=x.device)
            absolute_time_encode = absolute_time_expansion(absolute_times, self.embed_dim)
            absolute_time_embed = self.absolute_time_embed(absolute_time_encode.to(dtype=dtype))
            x = x + absolute_time_embed.unsqueeze(1)  # (B, L, D) + (B, 1, D)

        x = self.pos_drop(x)
        return x
