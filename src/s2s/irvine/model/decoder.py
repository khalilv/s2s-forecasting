"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

from datetime import timedelta
from typing import Tuple, Union

import torch
from einops import rearrange
from torch import nn

from s2s.irvine.model.batch import Batch, Metadata
from s2s.irvine.model.util import (
    check_lat_lon_dtype,
    init_weights,
    unpatchify,
)

__all__ = ["WeatherDecoder"]


class WeatherDecoder(nn.Module):
    """Weather specific decoding module"""

    def __init__(
        self,
        surf_vars: Tuple[str, ...],
        atmos_vars: Tuple[str, ...],
        atmos_levels: Tuple[Union[int, float], ...],
        patch_size: int = 2,
        embed_dim: int = 1024,
        num_heads: int = 16,
    ) -> None:
        """Initialise.

        Args:
            surf_vars (Tuple[str, ...]): All supported surface-level variables.
            atmos_vars (Tuple[str, ...]): All supported atmospheric variables.
            atmos_levels (Tuple[int | float, ...]): All supported pressure levels.
            patch_size (int, optional): Patch size. Defaults to `2`.
            embed_dim (int, optional): Embedding dim. Defaults to `1024`.
            num_heads (int, optional): Number of attention heads used in the deaggregation blocks.
                Defaults to `16`.
        """
        super().__init__()

        self.patch_size = patch_size
        self.surf_vars = surf_vars
        self.atmos_vars = atmos_vars
        self.embed_dim = embed_dim
        self.atmos_levels = atmos_levels
        
        self.level_query = nn.ParameterDict(
            {str(level): nn.Parameter(torch.randn(1, 1, embed_dim)) for level in atmos_levels}
        )
        self.level_deagg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.level_deagg_norm = nn.LayerNorm(embed_dim)
        self.atmos_var_query = nn.ParameterDict(
            {name: nn.Parameter(torch.randn(1, 1, embed_dim)) for name in atmos_vars}
        )
        self.atmos_var_deagg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.atmos_var_deagg_norm = nn.LayerNorm(embed_dim)
        self.surf_var_query = nn.ParameterDict(
            {name: nn.Parameter(torch.randn(1, 1, embed_dim)) for name in surf_vars}
        )        
        self.surf_var_deagg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.surf_var_deagg_norm = nn.LayerNorm(embed_dim)

        self.surf_heads = nn.ParameterDict(
            {name: nn.Linear(embed_dim, patch_size**2) for name in surf_vars}
        )
        self.atmos_heads = nn.ParameterDict(
            {name: nn.Linear(embed_dim, patch_size**2) for name in atmos_vars}
        )

        self.apply(init_weights)

        for level in atmos_levels:
            torch.nn.init.trunc_normal_(self.level_query[str(level)], std=0.02)
        for var in surf_vars:
            torch.nn.init.trunc_normal_(self.surf_var_query[var], std=0.02)
        for var in atmos_vars:
            torch.nn.init.trunc_normal_(self.atmos_var_query[var], std=0.02)

    
    def deaggregate(self, x: torch.Tensor, deaggregator: nn.MultiheadAttention, query: torch.Tensor, norm: nn.LayerNorm) -> torch.Tensor:
        """Deaggregate information using nn.MultiHeadAttention

        Args:
            x (torch.Tensor): Tensor of shape `(B, C_out, L, D)` where `C_out` refers to the number
                of aggregated levels.
            deaggregator (nn.MultiheadAttention): nn.MultiheadAttention module to perform the deaggregation
            query (torch.Tensor): Tensor of shape `(1, C_in, D)` where `C_in` refers to the number
                of input levels.
            norm (nn.LayerNorm): nn.LayerNorm module to perform normalization after deaggregating

        Returns:
            torch.Tensor: Tensor of shape `(B, C_in, L, D)` where `C_in` refers to the number 
                of input levels.
        """
        B, _, L, _ = x.shape
        x = torch.einsum("bcld->blcd", x)
        x = x.flatten(0, 1)  # (B * L, C_out, D)
        
        query = query.repeat_interleave(x.shape[0], dim=0) # (B * L, C_in, D)

        x, _ = deaggregator(query, x, x) # (B * L, C_in, D)
        x = norm(x)
        x = x.unflatten(dim=0, sizes=(B, L))  # (B, L, C_in, D)
        x = torch.einsum("blcd->bcld", x)  # (B, C_in, L, D)
        return x
    

    def forward(
        self,
        x: torch.Tensor,
        batch: Batch,
        patch_res: Tuple[int, int, int],
        lead_time: timedelta,
    ) -> Batch:
        """Forward pass of WeatherDecoder.

        Args:
            x (torch.Tensor): Backbone output of shape `(B, L, D)`.
            batch (:class:`irvine.batch.Batch`): Batch to make predictions for.
            patch_res (Tuple[int, int, int]): Patch resolution
            lead_time (timedelta): Lead time.

        Returns:
            :class:`irvine.batch.Batch`: Prediction for `batch`.
        """
        surf_vars = tuple(batch.surf_vars.keys())
        atmos_vars = tuple(batch.atmos_vars.keys())
        atmos_levels = batch.metadata.atmos_levels

        #construct queries based on batch data
        level_query = torch.cat([self.level_query[str(level)] for level in atmos_levels], dim=1)
        surf_var_query = torch.cat([self.surf_var_query[var] for var in surf_vars], dim=1)
        atmos_var_query = torch.cat([self.atmos_var_query[var] for var in atmos_vars], dim=1)

        B, L, D = x.shape

        # extract the lat, lon and convert to float32.
        lat, lon = batch.metadata.lat, batch.metadata.lon
        check_lat_lon_dtype(lat, lon)
        lat, lon = lat.to(dtype=torch.float32), lon.to(dtype=torch.float32)
        H, W = lat.shape[0], lon.shape[-1]

        # Unwrap the latent level dimension.
        x = rearrange(
            x,
            "b (c h w) d -> b (h w) c d",
            c=patch_res[0],
            h=patch_res[1],
            w=patch_res[2],
        )

        x_surf = x[:, :, 0, :]
        x_surf = rearrange(x_surf, 'b l d -> b 1 l d')
        x_surf = self.deaggregate(x_surf, self.surf_var_deagg, surf_var_query, self.surf_var_deagg_norm) # (B, V_s, L, D)
        x_surf = torch.stack([self.surf_heads[name](x_surf[:, i]) for i, name in enumerate(surf_vars)], dim=-2) # (B, L, V_s, p*p)
        x_surf = rearrange(x_surf, 'b l v p -> b l 1 (v p)')
        x_surf = unpatchify(x_surf, len(surf_vars), H, W, self.patch_size).squeeze(2) #(B, V_s, H, W)

        x_atmos = x[:, :, 1:, :]
        x_atmos = rearrange(x_atmos, 'b l c d -> b c l d')
        x_atmos = self.deaggregate(x_atmos, self.level_deagg, level_query, self.level_deagg_norm)
        x_atmos = rearrange(x_atmos, 'b c l d -> (b c) 1 l d')
        x_atmos = self.deaggregate(x_atmos, self.atmos_var_deagg, atmos_var_query, self.atmos_var_deagg_norm)
        x_atmos = rearrange(x_atmos, '(b c) v l d -> b l c v d', b=B, c=len(atmos_levels))
        x_atmos = torch.stack([self.atmos_heads[name](x_atmos[:, :, :, i, :]) for i, name in enumerate(atmos_vars)], dim=-2) #(B, L, C, V_a, p*p)
        x_atmos = rearrange(x_atmos, 'b l c v p -> b l c (v p)')
        x_atmos = unpatchify(x_atmos, len(atmos_vars), H, W, self.patch_size)

        return Batch(
            {v: x_surf[:, i] for i, v in enumerate(surf_vars)},
            batch.static_vars,
            {v: x_atmos[:, i] for i, v in enumerate(atmos_vars)},
            Metadata(
                lat=lat,
                lon=lon,
                time=tuple((t[-1] + lead_time,) for t in batch.metadata.time),
                atmos_levels=atmos_levels,
                rollout_step=batch.metadata.rollout_step + 1,
            ),
        )