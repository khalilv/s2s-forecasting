"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

from datetime import timedelta
from typing import Optional, Tuple

import torch
from einops import rearrange
from torch import nn

from s2s.irvine.model.batch import Batch
from s2s.irvine.model.fourier import (
    absolute_time_expansion,
    lead_time_expansion,
    levels_expansion,
    pos_expansion,
    scale_expansion,
)
from s2s.irvine.model.patchembed import PatchEmbed
from s2s.irvine.model.posencoding import pos_scale_enc
from s2s.irvine.model.util import (
    check_lat_lon_dtype,
    init_weights,
)

__all__ = ["WeatherEncoder"]


class WeatherEncoder(nn.Module):
    """Weather specific encoding module"""

    def __init__(
        self,
        surf_vars: Tuple[str, ...],
        static_vars: Optional[Tuple[str, ...]],
        atmos_vars: Tuple[str, ...],
        patch_size: int = 2,
        latent_levels: int = 4,
        embed_dim: int = 512,
        num_heads: int = 16,
        drop_rate: float = 0.1,
        history_size: int = 2,
        temporal_attention: bool = False
    ) -> None:
        """Initialise.

        Args:
            surf_vars (Tuple[str, ...]): All supported surface-level variables.
            static_vars (Optional[Tuple[str, ...]]): All supported static variables.
            atmos_vars (Tuple[str, ...]): All supported atmospheric variables.
            patch_size (int): Patch size. Defaults to `2`.
            latent_levels (int): Number of latent pressure levels. Defaults to `4`.
            embed_dim (int): Embedding dimension used in the aggregation blocks. Defaults
                to `512`.
            num_heads (int): Number of attention heads used in aggregation blocks.
                Defaults to `16`.
            drop_rate (float): Drop out rate for input patches. Defaults to `0.1`.
            history_size (int): History steps in input. Defaults to `2`.
            temporal_attention (bool): If true perform explicit cross attention over the temporal dimension. Defaults to `False`
        """
        super().__init__()
        assert history_size > 0, "At least one history step is required."
        assert latent_levels > 1, "At least two latent levels are required."

        # We treat the static variables as surface variables in the model.
        surf_vars = surf_vars + static_vars if static_vars is not None else surf_vars

        self.drop_rate = drop_rate
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.temporal_attention = temporal_attention
        self.atmos_vars = atmos_vars
        self.surf_vars = surf_vars

        self.latent_levels = latent_levels
        self.level_query = nn.Parameter(torch.randn(1, latent_levels - 1, embed_dim))
        self.level_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.level_agg_norm = nn.LayerNorm(embed_dim)
        self.atmos_var_query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.atmos_var_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.atmos_var_agg_norm = nn.LayerNorm(embed_dim)
        self.surf_var_query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.surf_var_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.surf_var_agg_norm = nn.LayerNorm(embed_dim)

        if self.temporal_attention:
            self.atmos_time_query = nn.Parameter(torch.randn(1, 1, embed_dim))
            self.atmos_time_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            self.atmos_time_agg_norm = nn.LayerNorm(embed_dim)
            self.surf_time_query = nn.Parameter(torch.randn(1, 1, embed_dim))
            self.surf_time_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            self.surf_time_agg_norm = nn.LayerNorm(embed_dim)

        # learnable embedding to encode the surface level.
        self.surf_level_encode = nn.Parameter(torch.randn(embed_dim))

        # position, scale, variable, and time embeddings
        self.pos_embed = nn.Linear(embed_dim, embed_dim)
        self.scale_embed = nn.Linear(embed_dim, embed_dim)
        self.lead_time_embed = nn.Linear(embed_dim, embed_dim)
        self.absolute_time_embed = nn.Linear(embed_dim, embed_dim)
        self.levels_embed = nn.Linear(embed_dim, embed_dim)
        self.surf_var_embed = nn.ParameterDict({
                var: nn.Parameter(torch.randn(1, embed_dim)) for var in self.surf_vars
        })
        self.atmos_var_embed = nn.ParameterDict({
                var: nn.Parameter(torch.randn(1, embed_dim)) for var in self.atmos_vars
        })
        
        self.surf_token_embeds = PatchEmbed(
            self.surf_vars,
            patch_size,
            embed_dim,
            in_channels = 1 if self.temporal_attention else history_size
        )
        self.atmos_token_embeds = PatchEmbed(
            self.atmos_vars,
            patch_size,
            embed_dim,
            in_channels= 1 if self.temporal_attention else history_size
        )
        
        # drop patches after encoding.
        self.pos_drop = nn.Dropout(p=drop_rate)

        #final normalization layer before backbone
        self.final_norm = nn.LayerNorm(embed_dim)

        self.apply(init_weights)

        torch.nn.init.trunc_normal_(self.atmos_var_query, std=0.02)
        torch.nn.init.trunc_normal_(self.surf_var_query, std=0.02)
        torch.nn.init.trunc_normal_(self.level_query, std=0.02)
        torch.nn.init.trunc_normal_(self.surf_level_encode, std=0.02)
        for var in self.surf_vars:
            nn.init.trunc_normal_(self.surf_var_embed[var], std=0.02)
        for var in self.atmos_vars:
            nn.init.trunc_normal_(self.atmos_var_embed[var], std=0.02)
        if self.temporal_attention:
            torch.nn.init.trunc_normal_(self.surf_time_query, std=0.02)
            torch.nn.init.trunc_normal_(self.atmos_time_query, std=0.02)

    def aggregate(self, x: torch.Tensor, aggregator: nn.MultiheadAttention, query: torch.Tensor, norm: nn.LayerNorm) -> torch.Tensor:
        """Aggregate information using nn.MultiHeadAttention

        Args:
            x (torch.Tensor): Tensor of shape `(B, C_in, L, D)` where `C_in` refers to the number
                of input levels.
            aggregator (nn.MultiheadAttention): nn.MultiheadAttention module to perform the aggregation
            query (torch.Tensor): Tensor of shape `(1, C_out, D)` where `C_out` refers to the number
                of aggregated levels.
            norm (nn.LayerNorm): nn.LayerNorm module to perform normalization after aggregating

        Returns:
            torch.Tensor: Tensor of shape `(B, C_out, L, D)` where `C_out` refers to the number 
                of aggregated levels.
        """
        B, _, L, _ = x.shape
        x = torch.einsum("bcld->blcd", x)
        x = x.flatten(0, 1)  # (B * L, C_in, D)
        
        query = query.repeat_interleave(x.shape[0], dim=0) # (B * L, C_out, D)

        x, _ = aggregator(query, x, x) # (B * L, C, D)
        x = norm(x)
        x = x.unflatten(dim=0, sizes=(B, L))  # (B, L, C, D)
        x = torch.einsum("blcd->bcld", x)  # (B, C, L, D)
        return x
    
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

        if self.temporal_attention:
            x_surf = rearrange(x_surf, "b t v h w -> (b t) v 1 h w")
            x_surf = self.surf_token_embeds(x_surf, surf_vars)  
            x_surf = rearrange(x_surf, "(b t) v l d -> b t v l d", b=B, t=T)         

            x_atmos = rearrange(x_atmos, "b t v c h w -> (b t c) v 1 h w")
            x_atmos = self.atmos_token_embeds(x_atmos, atmos_vars)  
            x_atmos = rearrange(x_atmos, "(b t c) v l d -> b t c v l d", b=B, c=C, t=T)

            timestamp_hours = [[dt.timestamp() / 3600  for dt in batch] for batch in batch.metadata.time]
            timestamp_hours_tensor = torch.tensor(timestamp_hours, dtype=torch.float32, device=x_surf.device)
            timestamp_hours_encoded = absolute_time_expansion(timestamp_hours_tensor, self.embed_dim).to(dtype=x_surf.dtype)
            timestamp_hours_embedding = self.absolute_time_embed(timestamp_hours_encoded)
            x_atmos = x_atmos + timestamp_hours_embedding[:, :, None, None, None, :]
            x_surf = x_surf + timestamp_hours_embedding[:, :, None, None, :]

            x_surf = rearrange(x_surf, "b t v l d -> (b v) t l d")
            x_surf = self.aggregate(x_surf, self.surf_time_agg, self.surf_time_query, self.surf_time_agg_norm).squeeze()
            x_surf = rearrange(x_surf, "(b v) l d -> b v l d", b=B)

            x_atmos = rearrange(x_atmos, "b t c v l d -> (b c v) t l d")
            x_atmos = self.aggregate(x_atmos, self.atmos_time_agg, self.atmos_time_query, self.atmos_time_agg_norm).squeeze()
            x_atmos = rearrange(x_atmos, "(b c v) l d -> b c v l d", b=B, c=C)

        else:
            x_surf = rearrange(x_surf, "b t v h w -> b v t h w")
            x_surf = self.surf_token_embeds(x_surf, surf_vars)  # (B, V, L, D)

            x_atmos = rearrange(x_atmos, "b t v c h w -> (b c) v t h w")
            x_atmos = self.atmos_token_embeds(x_atmos, atmos_vars)
            x_atmos = rearrange(x_atmos, "(b c) v l d -> b c v l d", b=B, c=C)

            
        surf_var_embedding = torch.stack([self.surf_var_embed[var] for var in surf_vars], dim=1)
        x_surf = x_surf + surf_var_embedding[:,:,None,:]
        x_surf = self.aggregate(x_surf, self.surf_var_agg, self.surf_var_query, self.surf_var_agg_norm) #(B, 1, L, D)
        
        atmos_var_embedding = torch.stack([self.atmos_var_embed[var] for var in atmos_vars], dim=1)
        x_atmos = x_atmos + atmos_var_embedding[:,None,:,None,:]
        x_atmos = rearrange(x_atmos, "b c v l d -> (b c) v l d")
        x_atmos = self.aggregate(x_atmos, self.atmos_var_agg, self.atmos_var_query, self.atmos_var_agg_norm)
        x_atmos = rearrange(x_atmos, "(b c) 1 l d -> b c l d", b=B, c=C)

        atmos_levels_tensor = torch.tensor(atmos_levels, device=x_atmos.device)
        atmos_levels_encoded = levels_expansion(atmos_levels_tensor, self.embed_dim).to(dtype=x_atmos.dtype)
        atmos_levels_embedding = self.levels_embed(atmos_levels_encoded)
        x_atmos = x_atmos + atmos_levels_embedding[None, :, None, :]

        x_atmos = self.aggregate(x_atmos, self.level_agg, self.level_query, self.level_agg_norm) #(B, C_l, L, D)

        # add learnable surface level encoding.
        x_surf = x_surf + self.surf_level_encode[None, None, None, :].to(dtype=x_surf.dtype)

        # Concatenate the surface level with the amospheric levels.
        x = torch.cat((x_surf, x_atmos), dim=1)

        # Add position and scale embeddings to the 3D tensor.
        pos_encode, scale_encode = pos_scale_enc(
            self.embed_dim,
            lat,
            lon,
            self.patch_size,
            pos_expansion=pos_expansion,
            scale_expansion=scale_expansion,
        )
        
        # encodings are (L, D).
        pos_encode = self.pos_embed(pos_encode[None, None, :, :].to(dtype=x.dtype))
        scale_encode = self.scale_embed(scale_encode[None, None, :, :].to(dtype=x.dtype))
        x = x + pos_encode + scale_encode

        # flatten the tokens.
        x = x.reshape(B, -1, self.embed_dim)  # (B, C_l + 1, L, D) to (B, L', D)

        # add lead time embedding.
        lead_hours = lead_time.total_seconds() / 3600
        lead_times = lead_hours * torch.ones(B, dtype=x.dtype, device=x.device)
        lead_time_encode = lead_time_expansion(lead_times, self.embed_dim).to(dtype=x.dtype)
        lead_time_emb = self.lead_time_embed(lead_time_encode)  # (B, D)
        x = x + lead_time_emb.unsqueeze(1)  # (B, L', D) + (B, 1, D)

        # add absolute time embedding here if temporal aggregation was not used. just use latest timestamp for each batch element
        if not self.temporal_attention:
            timestamp_hours = [t[-1].timestamp() / 3600 for t in batch.metadata.time] 
            timestamp_hours_tensor = torch.tensor(timestamp_hours, dtype=torch.float32, device=x.device)
            timestamp_hours_encoded = absolute_time_expansion(timestamp_hours_tensor, self.embed_dim)
            timestamp_hours_embedding = self.absolute_time_embed(timestamp_hours_encoded.to(dtype=x.dtype))
            x = x + timestamp_hours_embedding.unsqueeze(1)  # (B, L, D) + (B, 1, D)
        
        x = self.final_norm(x)
        x = self.pos_drop(x)
        return x
