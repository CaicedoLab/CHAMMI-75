# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
adapted from Dinov1 and Dinov2: https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
"""
import math
from functools import partial
import torch.nn.functional as F
from torch import Tensor
import torch
import torch.nn as nn
from typing import Union
import warnings
import os
from einops import rearrange
from torch.nn.attention import SDPBackend, sdpa_kernel


from model_utils import trunc_normal_


def exists(val):
    return val is not None


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None

try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (Attention)")
    else:
        warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xFormers is not available (Attention)")


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        sim = (q @ k.transpose(-2, -1)) * self.scale

        ## TODO: double check
        if exists(mask):
            mask_value = -torch.finfo(sim.dtype).max
            sim = sim.masked_fill(~mask, mask_value)

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_mask=None) -> tuple[Tensor, None]:
        if not XFORMERS_AVAILABLE:
            if attn_mask is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)
        if attn_mask is not None:
            L, S = q.size(1), k.size(1)
            attn_bias = torch.zeros(B, self.num_heads, L, S, dtype=q.dtype, device=q.device)
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = None

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, None


class MemEffAttentionV2(Attention):
    def forward(self, x: Tensor, mask=None) -> tuple[Tensor, None]:
        B, N, C = x.shape

        q, k, v = rearrange(self.qkv(x), "b n (l h d) -> l b h n d", l=3, h=self.num_heads, d=C // self.num_heads)
        ## NOTE: shape of q, k, v: [head, num_tokens, dim]

        with sdpa_kernel(
            [
                SDPBackend.FLASH_ATTENTION,  ## Tri Dao's flash attention
                SDPBackend.EFFICIENT_ATTENTION,  ## Meta's xFormers
                SDPBackend.MATH,
            ],
        ):
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)  ## output, x: [B, num_heads, N, dim_per_head]

        x = rearrange(x, "b h n d -> b n (h d)")

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, None


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attention_cls="regular",
        init_values=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        num_gpus = torch.cuda.device_count()

        if XFORMERS_AVAILABLE and num_gpus > 0:
            if attention_cls == "v1":
                AttentionClass = MemEffAttention
            elif attention_cls == "v2":
                AttentionClass = MemEffAttentionV2
            elif attention_cls == "regular":
                AttentionClass = Attention
        else:
            AttentionClass = Attention

        # print(f"-------- AttnClass: {AttentionClass}")
        self.attn = AttentionClass(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        ## add LayerScale ls1 and ls2, followed DINOv2: https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/layers/block.py#L89
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, mask, return_attention=False):
        y, attn = self.attn(self.norm1(x), mask)

        if return_attention:  ## TODO: check if this is using efficient attention, which won't return attention weights
            return attn
        x = x + self.drop_path(self.ls1(y))
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))
        return x
