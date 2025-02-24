# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed, trunc_normal_

from s2s.climaX.pos_embed import (
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed,
)

#4) Core architecture of ClimaX - should make an extension of this class to modify forward passes

class ClimaX(nn.Module):
    """Implements the ClimaX model as described in the paper,
    https://arxiv.org/abs/2301.10343

    Args:
        in_vars (list): list of variables to be used for training
        img_size (list): image size of the input data
        patch_size (int): patch size of the input data
        embed_dim (int): embedding dimension
        depth (int): number of transformer layers
        decoder_depth (int): number of decoder layers
        num_heads (int): number of attention heads
        mlp_ratio (float): ratio of mlp hidden dimension to embedding dimension
        drop_path (float): stochastic depth rate
        drop_rate (float): dropout rate
    """

    def __init__(
        self,
        in_vars,
        img_size=[32, 64],
        patch_size=2,
        embed_dim=1024,
        depth=8,
        decoder_depth=2,
        num_heads=16,
        mlp_ratio=4.0,
        drop_path=0.1,
        drop_rate=0.1,
        history_size=2,
        history_step=1,
        temporal_attention=False
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_vars = in_vars
        self.history_size = history_size
        self.history_step = history_step
        self.temporal_attention = temporal_attention

        # variable tokenization: separate embedding layer for each input variable

        self.token_embeds = nn.ModuleList(
            [PatchEmbed(img_size, patch_size, 1 if temporal_attention else history_size, embed_dim) for i in range(len(in_vars))]
        )
        self.num_patches = self.token_embeds[0].num_patches

        # variable embedding to denote which variable each token belongs to
        self.var_embed = nn.Parameter(torch.zeros(1, len(in_vars), embed_dim))
        self.var_map = {}
        for i, var in enumerate(in_vars):
            self.var_map[var] = i
        
        # variable aggregation: a learnable query and a single-layer cross attention
        self.var_query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.var_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # positional embedding and lead time embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.lead_time_embed = nn.Linear(1, embed_dim)

        if temporal_attention:
            # time embedding to denote which timestep each token belongs to
            self.time_embed = nn.Parameter(torch.zeros(1, history_size, embed_dim))
            # time aggregation: a learnable query and a single-layer cross attention
            self.time_queries = nn.ParameterList(
                [nn.Parameter(torch.zeros(1,1,embed_dim)) for i in range(len(in_vars))]
            )
            self.time_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # --------------------------------------------------------------------------

        # ViT backbone
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    drop_path=dpr[i],
                    norm_layer=nn.LayerNorm,
                    drop=drop_rate,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # --------------------------------------------------------------------------

        # prediction head
        self.head = nn.ModuleList()
        for _ in range(decoder_depth):
            self.head.append(nn.Linear(embed_dim, embed_dim))
            self.head.append(nn.GELU())
        self.head.append(nn.Linear(embed_dim, len(in_vars) * patch_size**2))
        self.head = nn.Sequential(*self.head)

        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        # initialize pos_emb and var_emb
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.img_size[0] / self.patch_size),
            int(self.img_size[1] / self.patch_size),
            cls_token=False,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        var_embed = get_1d_sincos_pos_embed_from_grid(self.var_embed.shape[-1], np.arange(len(self.in_vars)), scale=10000)
        self.var_embed.data.copy_(torch.from_numpy(var_embed).float().unsqueeze(0))

        if self.temporal_attention:
            timesteps = np.arange(self.history_size)[::-1] * -self.history_step
            time_embed = get_1d_sincos_pos_embed_from_grid(self.time_embed.shape[-1], timesteps, scale=1000)
            self.time_embed.data.copy_(torch.from_numpy(time_embed).float().unsqueeze(0))

            #start with large initial weight on last timestep for time aggregation 
            for i in range(len(self.in_vars)):
                self.time_queries[i].data.copy_(self.time_embed.data[:, -1:])
                
            embed_dim = self.time_agg.embed_dim
            self.time_agg.in_proj_weight[embed_dim:2*embed_dim].data.copy_(torch.eye(embed_dim))
            self.time_agg.in_proj_bias[embed_dim:2*embed_dim].data.zero_()
            self.time_agg.in_proj_weight[:embed_dim].data.copy_(torch.eye(embed_dim))
            self.time_agg.in_proj_bias[:embed_dim].data.zero_()

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @lru_cache(maxsize=None)
    def get_var_ids(self, vars, device):
        ids = np.array([self.var_map[var] for var in vars])
        return torch.from_numpy(ids).to(device)

    def get_var_emb(self, var_emb, vars):
        ids = self.get_var_ids(vars, var_emb.device)
        return var_emb[:, ids, :]

    def unpatchify(self, x: torch.Tensor, h=None, w=None):
        """
        x: (B, L, V * patch_size**2)
        return imgs: (B, V, H, W)
        """
        p = self.patch_size
        c = len(self.in_vars)
        h = self.img_size[0] // p if h is None else h // p
        w = self.img_size[1] // p if w is None else w // p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    #aggregation step 
    def aggregate(self, x: torch.Tensor, aggregator: nn.MultiheadAttention, query: torch.Tensor):
        """Aggregate information using nn.MultiHeadAttention

        Args:
            x (torch.Tensor): Tensor of shape `(B, C_in, L, D)` where `C_in` refers to the number
                of input channels.
            aggregator (nn.MultiheadAttention): nn.MultiheadAttention module to perform the aggregation
            query (torch.Tensor): Tensor of shape `(1, C_out, D)` where `C_out` refers to the number
                of aggregated channels.
        Returns:
            torch.Tensor: Tensor of shape `(B, C_out, L, D)` where `C_out` refers to the number 
                of aggregated channels.
        """
        b, _, l, _ = x.shape
        x = torch.einsum("bvld->blvd", x)
        x = x.flatten(0, 1)  # BxL, V, D

        #perform the aggregation using multi-headed attention
        q = query.repeat_interleave(x.shape[0], dim=0)
        x, _ = aggregator(q, x, x)  # BxL, D
        x = x.unflatten(dim=0, sizes=(b, l))  # B, L, C_out, D
        x = torch.einsum("blcd->bcld", x)  # (B, C_out, L, D)
        return x

    def encoder(self, x: torch.Tensor, lead_times: torch.Tensor, in_variables):
        """Encode input weather state
        
        Args:
            x (torch.Tensor): `[B, T, V, H, W]` shape. Input weather state
            lead_times: `[B]` shape. Forecasting lead times for each element of the batch.
            in_variables: `[V]` shape. Names of input variables.
        Returns:
            x (torch.Tensor): `[B, L, D]` shape. Encoded weather state
        """
        B, T, V, _, _ = x.shape

        if isinstance(in_variables, list):
            in_variables = tuple(in_variables)

        # tokenize each variable separately
        embeds = []
        var_ids = self.get_var_ids(in_variables, x.device)
        
        #D = embed dimension
        #L = HW/(p^2) where p = patch size
        if self.temporal_attention:
            x = x.flatten(0, 1)  # (B*T, V, H, W)
            for i in range(len(var_ids)):
                id = var_ids[i]
                embeds.append(self.token_embeds[id](x[:, i : i + 1]))
            x = torch.stack(embeds, dim=1)  # B*T, V, L, D
            x = x.unflatten(0, (B, T))  # (B, T, V, L, D)

            x = x + self.time_embed[:, :T].unsqueeze(2).unsqueeze(3)
            x = torch.einsum("btvld->bvtld", x)  # (B, V, T, L, D)            
            x_agg = []
            for i in range(len(var_ids)):
                id = var_ids[i]
                x_agg.append(self.aggregate(x[:,i], self.time_agg, self.time_queries[id]).squeeze(1))  # B, L, D
            x = torch.stack(x_agg, dim=1)  # B, V, L, D
        else:
            x = torch.einsum("btvhw->bvthw", x)  # (B, V, T, H, W)
            for i in range(len(var_ids)):
                id = var_ids[i]
                embeds.append(self.token_embeds[id](x[:, i]))
            x = torch.stack(embeds, dim=1)  # B, V, L, D

        # add variable embedding
        var_embed = self.get_var_emb(self.var_embed, in_variables)
        x = x + var_embed.unsqueeze(2)  # B, V, L, D

        # variable aggregation
        x = self.aggregate(x, self.var_agg, self.var_query)  # B, 1, L, D
        x = x.squeeze(1)

        # add pos embedding
        x = x + self.pos_embed

        # add lead time embedding
        lead_time_emb = self.lead_time_embed(lead_times.unsqueeze(-1))  # B, D
        lead_time_emb = lead_time_emb.unsqueeze(1)
        x = x + lead_time_emb  # B, L, D

        x = self.pos_drop(x)
        return x

    def backbone(self, x: torch.Tensor):
        """Backbone consisting of several attention blocks.

        Args:
            x (torch.Tensor): `[B, L, D]` shape. Encoded weather state across the globe
        Returns:
            x (torch.Tensor): `[B, L, D]` shape. Transformed weather state
        """
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def decoder(self, x: torch.Tensor):
        """Decode weather state back to the spatial grid
        
        Args:
            x (torch.Tensor): `[B, L, D]` shape. Transformed weather state from backbone
        Returns:
            x (torch.Tensor): `[B, V, H, W]` shape. Decoded weather state
        """
        x = self.head(x)  # B, L, V*p*p
        x = self.unpatchify(x) # B, V, H, W
        return x

    def forward(self, x, lead_times, in_variables, out_variables):
        """Forward pass through the model.

        Args:
            x: `[B, T, V, H, W]` shape. Input weather/climate variables
            lead_times: `[B]` shape. Forecasting lead times of each element of the batch.
            variables: `[V]` shape. Names of input variables.
            output_variables: `[Vo]` shape. Names of output variables.

        Returns:
            preds (torch.Tensor): `[B, Vo, H, W]` shape. Predicted weather/climate variables.
        """
        x = self.encoder(x, lead_times, in_variables)  # B, L, D
        x = self.backbone(x)
        x = self.decoder(x)
        out_var_ids = self.get_var_ids(tuple(out_variables), x.device)
        x = x[:, out_var_ids]

        return x
