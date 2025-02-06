"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers.helpers import to_2tuple

__all__ = ["PatchEmbed"]

class PatchEmbed(nn.Module):
    """At either the surface or at a single pressure level, map each variable into a seperate
        embedding."""
    
    def __init__(
        self,
        var_names: Tuple[str, ...],
        patch_size: int = 2,
        embed_dim: int = 1024,
        in_channels: int = 1,
        norm_layer: Optional[nn.Module] = None,
        flatten: bool = True
    ) -> None:
        """Initialise.

        Args:
            var_names (Tuple[str, ...]): Variables to embed.
            patch_size (int): Patch size. Defaults to 2.
            embed_dim (int): Embedding dimensionality. Defaults to 1024.
            in_channels (int): Number of input channels. Defaults to 1.
            norm_layer (torch.nn.Module, optional): Normalisation layer to be applied at the very
                end. Defaults to None.
            flatten (bool): At the end of the forward pass, flatten the two spatial dimensions
                into a single dimension. See :meth:`PatchEmbed.forward` for more details. Defaults to true.
        """
        super().__init__()

        self.var_names = var_names
        self.kernel_size = to_2tuple(patch_size)
        self.flatten = flatten
        self.embed_dim = embed_dim
        self.in_channels = in_channels

        self.weights = nn.ParameterDict(
            {
                # Shape (D, C_in, H, W)
                name: nn.Parameter(torch.empty(self.embed_dim, self.in_channels, *self.kernel_size))
                for name in self.var_names
            }
        )
        self.bias = nn.ParameterDict(
            {
                name: nn.Parameter(torch.empty(self.embed_dim))
                for name in self.var_names
            }
        )
        self.norm = norm_layer(self.embed_dim) if norm_layer else nn.Identity()

        self.init_weights()
    
    def init_weights(self) -> None:
        """Initialise weights."""
        for var in self.var_names:
            nn.init.kaiming_uniform_(self.weights[var], a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights[var])
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias[var], -bound, bound) 

    def forward(self, x: torch.Tensor, var_names: Tuple[str, ...]) -> torch.Tensor:
        """Run the embedding.

        Args:
            x (:class:`torch.Tensor`): Tensor to embed of a shape of `(B, V, T, H, W)`.
            var_names (Tuple[str, ...]): Names of the variables in `x`. The length should be equal
                to `V`.

        Returns:
            :class:`torch.Tensor`: Embedded tensor a shape of `(B, V, L, D]) if flattened,
                where `L = H * W / P^2`. Otherwise, the shape is `(B, V, D, H', W')`.

        """
        _, V, T, H, W = x.shape
        assert len(var_names) == V, f"{V} != {len(var_names)}."
        assert H % self.kernel_size[0] == 0, f"{H} % {self.kernel_size[0]} != 0."
        assert W % self.kernel_size[1] == 0, f"{W} % {self.kernel_size[1]} != 0."
        assert len(set(var_names)) == len(var_names), f"{var_names} contains duplicates."
        assert T == self.in_channels, f'Input channels: {T} does not match configured input channels: {self.in_channels}'
        
        out = []
        for idx, var in enumerate(var_names):
            # select the weights of the variables that are present in the batch.
            weights = self.weights[var]
            bias = self.bias[var]
            
            proj = F.conv2d(
                    input=x[:,idx],        # (B, T, H, W)
                    weight=weights,        # (D, C_in, P, P)
                    bias=bias,             # (D)
                    stride=self.kernel_size,
                )
                        
            if self.flatten:
                proj = proj.flatten(2).transpose(1, 2) # (B, L, D)

            proj = self.norm(proj)
            out.append(proj) 
        return torch.stack(out, dim=1)