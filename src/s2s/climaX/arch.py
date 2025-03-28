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
        history (list): history steps to include in input
        hrs_each_step: number of hours between each step
        temporal_attention (bool): if true, perform multiheaded attention to aggregate time
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
        history=[],
        hrs_each_step=6,
        temporal_attention=False
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_vars = in_vars
        self.history = history
        self.temporal_attention = temporal_attention
        self.hrs_each_step = hrs_each_step

        # variable tokenization: separate embedding layer for each input variable

        self.token_embeds = nn.ModuleList(
            [PatchEmbed(img_size, patch_size, 1 if temporal_attention else len(history), embed_dim) for i in range(len(in_vars))]
        )
        self.num_patches = self.token_embeds[0].num_patches

        # variable embedding to denote which variable each token belongs to
        self.var_embed = nn.Parameter(torch.zeros(1, len(in_vars), embed_dim))
        self.var_map = {}
        for i, var in enumerate(in_vars):
            self.var_map[var] = i
        
        # variable aggregation: a learnable query and a single-layer cross attention
        self.var_query = nn.Parameter(torch.zeros(self.num_patches, 1, embed_dim))
        self.var_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # positional embedding and lead time embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.lead_time_embed = nn.Linear(1, embed_dim)

        if temporal_attention:
            # time embedding to denote which timestep each token belongs to
            self.time_embed = nn.Parameter(torch.zeros(1, len(history), embed_dim))
            # time aggregation: a learnable query and a single-layer cross attention
            self.time_query = nn.Parameter(torch.zeros(self.num_patches, 1, embed_dim))
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
            time_embed = get_1d_sincos_pos_embed_from_grid(self.time_embed.shape[-1], np.array(self.history), scale=10000)
            self.time_embed.data.copy_(torch.from_numpy(time_embed).float().unsqueeze(0))

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
    def aggregate(self, x: torch.Tensor, aggregator: nn.MultiheadAttention, query: torch.Tensor, need_weights: bool):
        """Aggregate information using nn.MultiHeadAttention

        Args:
            x (torch.Tensor): Tensor of shape `(B, C_in, L, D)` where `C_in` refers to the number
                of input channels.
            aggregator (nn.MultiheadAttention): nn.MultiheadAttention module to perform the aggregation
            query (torch.Tensor): Tensor of shape `(L, C_out, D)` where `C_out` refers to the number
                of aggregated channels.
            need_weights (bool): If true, attention weights with shape (B, L, C_out, C_in) will be returned 
                alongside output tensor
        Returns:
            x_agg (torch.Tensor): Aggregated tensor of shape `(B, C_out, L, D)` where `C_out` refers 
                to the number of aggregated channels.
            attn_weights (torch.Tensor): Attention weights with shape (B, L, C_out, C_in) if need_weights is
                set to True. Otherwise None
        """
        b, _, l, _ = x.shape
        x = torch.einsum("bvld->blvd", x)
        x = x.flatten(0, 1)  # (BxL, C_in, D)

        #perform the aggregation using multi-headed attention
        q = query.unsqueeze(0) # (1, L, C_out, D)
        q = q.expand(b, -1, -1, -1) # (B, L, C_out, D)
        q = q.flatten(0, 1) # (B*L, C_out, D)

        x, attn_weights = aggregator(q, x, x, need_weights=need_weights)
        x = x.unflatten(dim=0, sizes=(b, l))  # (B, L, C_out, D)
        x = torch.einsum("blcd->bcld", x)  # (B, C_out, L, D)
        
        if need_weights:
            attn_weights = attn_weights.unflatten(dim=0, sizes=(b, l))  # B, L, C_out, C_in

        return x, attn_weights

    def encoder(self, x: torch.Tensor, lead_times: torch.Tensor, in_variables: list, need_weights: bool):
        """Encode input weather state
        
        Args:
            x (torch.Tensor): `[B, T, V, H, W]` shape. Input weather state
            lead_times: `[B]` shape. Forecasting lead times for each element of the batch.
            in_variables: `[V]` shape. Names of input variables.
            need_weights (bool): If true, attention weights for variable aggregation
                will be computed and returned alongside output tensor
        Returns:
            x (torch.Tensor): `[B, L, D]` shape. Encoded weather state
            var_agg_weights (torch.Tensor): `[B, L, 1, V]` shape. Attention weights for variable aggregation if 
                need_weights is True. Otherwise None
        """        
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
            x = torch.stack(embeds, dim=1)  # (B*T, V, L, D)
        else:
            x = torch.einsum("btvhw->bvthw", x)  # (B, V, T, H, W)
            for i in range(len(var_ids)):
                id = var_ids[i]
                embeds.append(self.token_embeds[id](x[:, i]))
            x = torch.stack(embeds, dim=1)  # (B, V, L, D)

        ##############################################
        #let B' = B*T if self.temporal_attention else B
        ##############################################

        # add variable embedding
        var_embed = self.get_var_emb(self.var_embed, in_variables)
        x = x + var_embed.unsqueeze(2)  # (B', V, L, D) or 

        # variable aggregation
        x, var_agg_weights = self.aggregate(x, self.var_agg, self.var_query, need_weights=need_weights)  # (B', 1, L, D)
        x = x.squeeze(1)

        # add pos embedding
        x = x + self.pos_embed

        # add lead time embedding
        if self.temporal_attention:
            dts = torch.tensor(self.history).to(lead_times.device) * -self.hrs_each_step/100
            lead_times = lead_times[:, None] + dts[None, :]
            lead_times = lead_times.flatten(0, 1)

        lead_time_emb = self.lead_time_embed(lead_times.unsqueeze(-1))  # (B', D)
        
        lead_time_emb = lead_time_emb.unsqueeze(1)
        x = x + lead_time_emb  # (B', L, D)

        x = self.pos_drop(x)
        return x, var_agg_weights

    def backbone(self, x: torch.Tensor):
        """Backbone consisting of several attention blocks.

        Args:
            x (torch.Tensor): `[B', L, D]` shape. Encoded weather state across the globe
        Returns:
            x (torch.Tensor): `[B', L, D]` shape. Transformed weather state
        """
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def decoder(self, x: torch.Tensor, B: int, T:int, need_weights: bool):
        """Decode weather state back to the spatial grid
        
        Args:
            x (torch.Tensor): `[B', L, D]` shape. Transformed weather state from backbone
            B (int): Batch size
            T (int): History size
            need_weights (bool): If true, attention weights for time aggregation
                will be computed and returned alongside output tensor if temporal attention is used
        Returns:
            x (torch.Tensor): `[B, V, H, W]` shape. Decoded weather state
            time_agg_weights (torch.Tensor): `[B, V, L, 1, T]` shape. Attention weights for time aggregation if 
                need_weights is True and temporal attention was used. Otherwise None
        """
        time_agg_weights = None

        if self.temporal_attention:
            x = x.unflatten(0, (B, T)) # (B, T, L, D)
            x = x + self.time_embed[:, :T].unsqueeze(2)
            x, time_agg_weights = self.aggregate(x, self.time_agg, self.time_query, need_weights=need_weights)  # (B, 1, L, D)
            x = x.squeeze(1) # (B, L, D)

        x = self.head(x)  # B, L, V*p*p
        x = self.unpatchify(x) # B, V, H, W
        return x, time_agg_weights

    def forward(self, x: torch.tensor, lead_times: torch.tensor, in_variables: list, out_variables: list, need_weights: bool = False):
        """Forward pass through the model.

        Args:
            x (torch.Tensor): `[B, T, V, H, W]` shape. Input weather/climate variables
            lead_times (torch.Tensor): `[B]` shape. Forecasting lead times of each element of the batch.
            variables (list): `[V]` shape. Names of input variables.
            output_variables (list): `[Vo]` shape. Names of output variables.
            need_weights (bool): If true, attention weights for variable aggregation and time aggregation 
                will be computed and returned alongside output tensor
        Returns:
            preds (torch.Tensor): `[B, Vo, H, W]` shape. Predicted weather/climate variables.
            var_agg_weights (torch.Tensor): `[B, H/p, W/p, 1, Vo]` shape. Attention weights for variable aggregation if 
                need_weights is True. Otherwise None
            time_agg_weights (torch.Tensor): `[B, Vo, H/p, W/p, 1, T]` shape. Attention weights for time aggregation if 
                need_weights is True and temporal attention was used. Otherwise None
        """
        B, T, _, _, _ = x.shape

        x, var_agg_weights = self.encoder(x, lead_times, in_variables, need_weights)  # B, L, D
        x = self.backbone(x)
        x, time_agg_weights = self.decoder(x, B, T, need_weights)
        out_var_ids = self.get_var_ids(tuple(out_variables), x.device)
        x = x[:, out_var_ids]

        if need_weights:
            hp, wp = int(self.img_size[0] / self.patch_size), int(self.img_size[1] / self.patch_size)
            var_agg_weights = var_agg_weights.unflatten(dim=1, sizes=(hp, wp))[:, :, :, :, out_var_ids] # B, H/p, W/p, 1, Vo
            if self.temporal_attention:
                time_agg_weights = time_agg_weights.unflatten(dim=2, sizes=(hp, wp))[:, out_var_ids] # B, Vo, H/p, W/p, 1, T

        return x, var_agg_weights, time_agg_weights