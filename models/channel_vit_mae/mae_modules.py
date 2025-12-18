### adapted from Recursion Pharmaceuticals 2024
from functools import partial
from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from einops import repeat
from vit import Block, Mlp


def exists(val):
    return val is not None


def default(val, default):
    return val if exists(val) else default


class SelfStandardize(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_standardize = nn.LazyInstanceNorm2d(
            affine=False, track_running_stats=False
        )

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        x = pixels.float() / 255.0
        return self.self_standardize(x)


class ChAMAEViTDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        depth: int = 8,
        num_heads: int = 16,
        mlp_ratio: float = 4,
        qkv_bias: bool = True,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),  # type: ignore[assignment]
        num_channels: int = -1,
        attention_cls: str = "v2",
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.pos_embeddings = None  # to be overwritten by MAE class
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.channel_tokens = nn.Embedding(num_channels, embed_dim)
        self.blocks = nn.Sequential(
            *[
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    attention_cls=attention_cls,
                )
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        ## initialize modality tokens and mask token
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(
        self, x: Tensor, pos_embeddings: Tensor, full_patch_masks: Tensor | None = None
    ) -> torch.Tensor:
        x = x + pos_embeddings
        for blk in self.blocks:
            x = blk(x, mask=full_patch_masks)
        x = self.norm(x)
        return x  # type: ignore[no-any-return]

    def forward_masked(
        self,
        x: Tensor,
        ind_restore: Tensor,
        pos_embeddings: Tensor,
        channel_token_patches: Optional[Tensor],
        full_patch_masks: Optional[Tensor],
        num_extra_tokens: int,
    ) -> torch.Tensor:
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ind_restore.shape[1] + num_extra_tokens - x.shape[1], 1
        )
        ## remove class token and potentially meta tokens, then concat all unmasked tokens and mask tokens
        x_ = torch.cat([x[:, num_extra_tokens:, :], mask_tokens], dim=1)
        x_ = torch.gather(
            x_, dim=1, index=ind_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle

        ## add channel token to each patch to help with reconstruction
        if channel_token_patches is not None:
            x_ = x_ + channel_token_patches
        x = torch.cat([x[:, :num_extra_tokens, :], x_], dim=1)  # add class token
        x = self.forward(x, pos_embeddings, full_patch_masks)
        return x  # type: ignore[no-any-return]


class CrossAttention(nn.Module):
    def __init__(
        self, embed_dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = head_dim**-0.5

        self.q = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.kv = nn.Linear(embed_dim, embed_dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context):
        B, N, C = x.shape
        _, M, _ = context.shape

        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        kv = (
            self.kv(context)
            .reshape(B, M, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CAMAEDecoder(nn.Module):
    def __init__(
        self,
        num_modalities: int = 6,
        tokens_per_modality: int = 256,
        embed_dim: int = 256,
        depth: int = 2,
        num_heads: int = 16,
        mlp_ratio: float = 4,
        qkv_bias: bool = True,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),  # type: ignore[assignment]
    ) -> None:
        super().__init__()
        self.num_modalities = num_modalities
        self.tokens_per_modality = tokens_per_modality
        self.embed_dim = embed_dim
        self.pos_embeddings = None  # to be overwritten by MAE class
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.placeholder = nn.Parameter(
            torch.zeros(1, 1, embed_dim), requires_grad=False
        )
        self.modality_tokens = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                for _ in range(self.num_modalities)
            ]
        )

        ## initialize modality tokens and mask token
        nn.init.normal_(self.mask_token, std=0.02)
        for m_token in self.modality_tokens:
            nn.init.normal_(m_token, std=0.02)

        self.cross_attention = CrossAttention(embed_dim=self.embed_dim)
        self.mlp = Mlp(self.embed_dim, hidden_features=int(self.embed_dim * mlp_ratio))
        print("-------- depth", depth)
        self.decoders = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        Block(
                            embed_dim,
                            num_heads,
                            mlp_ratio,
                            qkv_bias=qkv_bias,
                            norm_layer=norm_layer,
                        )
                        for i in range(depth)
                    ]
                )
                for _ in range(self.num_modalities)
            ]
        )
        self.context_norm = norm_layer(embed_dim)
        self.query_norm = norm_layer(embed_dim)
        self.out_norm = norm_layer(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        channel_ids_list: List[int],
        num_extra_tokens: int = 1,
        pos_embeddings: Optional[Tensor] = None,
        each_img_has_different_channels: bool = False,
    ) -> torch.Tensor:
        placeholder_tokens = repeat(
            self.placeholder, "1 1 d -> 1 n d", n=num_extra_tokens
        )

        ##### collect modality tokens for decoders
        if (
            not each_img_has_different_channels
        ):  ### regular case (ie, all images have the same channels), no channel masking
            channel_ids = channel_ids_list[0]
            modality_tokens_concat = torch.cat(
                [
                    placeholder_tokens,
                ]  # placeholder for class token
                + [
                    m_t.repeat(1, self.tokens_per_modality, 1)
                    for m_idx, m_t in enumerate(self.modality_tokens)
                    if m_idx in channel_ids
                ],
                dim=1,
            )
        else:
            modality_tokens = []
            max_channel_count = max([len(c_ids) for c_ids in channel_ids_list])
            for channel_ids in channel_ids_list:
                modality_tokens_per_sample = torch.cat(
                    [
                        placeholder_tokens,
                    ]  # placeholder for class token
                    + [
                        self.modality_tokens[i].repeat(1, self.tokens_per_modality, 1)
                        for i in channel_ids
                    ],
                    dim=1,
                )
                padding_needed = max_channel_count - len(channel_ids)
                if padding_needed > 0:
                    padding_tokens = self.placeholder.repeat(
                        1, padding_needed * self.tokens_per_modality, 1
                    )
                    modality_tokens_per_sample = torch.cat(
                        [modality_tokens_per_sample, padding_tokens], dim=1
                    )
                modality_tokens.append(modality_tokens_per_sample)
            modality_tokens_concat = torch.cat(modality_tokens, dim=0)

        x = (
            x + pos_embeddings + modality_tokens_concat
        )  # add pos and tiled modality tokens
        x_ = x[:, num_extra_tokens:, :]  # no class token

        #### pass each input channel through its own decoder
        if (
            not each_img_has_different_channels
        ):  ### regular case (ie, all images have the same channels), no channel masking
            channel_ids = channel_ids_list[0]
            decoders = [self.decoders[idx] for idx in channel_ids]
            x_m_s = []
            for m, decoder in enumerate(
                decoders
            ):  # iterate through modalities and decoders
                x_m = x_[
                    :,
                    m * self.tokens_per_modality : (m + 1) * self.tokens_per_modality,
                    :,
                ]
                x_m = self.cross_attention(self.query_norm(x_m), self.context_norm(x_))
                x_m = x_m + self.mlp(self.out_norm(x_m))
                x_m = decoder(x_m)
                x_m_s.append(x_m)
            x_m_s = torch.cat(x_m_s, dim=1)  # concat all tokens
            x_m_s = torch.cat(
                [x[:, :num_extra_tokens, :], x_m_s], dim=1
            )  # add back class token
        else:
            ### general case, each image can have different channels
            x_m_s = []
            max_channel_count = max([len(c_ids) for c_ids in channel_ids_list])
            for i, channel_ids in enumerate(channel_ids_list):
                decoders = [self.decoders[idx] for idx in channel_ids]
                x_m_per_sample = []
                for m, decoder in enumerate(
                    decoders
                ):  # iterate through modalities and decoders
                    x_m = x_[
                        i : i + 1,
                        m * self.tokens_per_modality : (m + 1)
                        * self.tokens_per_modality,
                        :,
                    ]
                    x_m = self.cross_attention(
                        self.query_norm(x_m),
                        self.context_norm(
                            x_[
                                i : i + 1,
                                : self.tokens_per_modality * len(channel_ids),
                                :,
                            ]
                        ),
                    )
                    x_m = x_m + self.mlp(self.out_norm(x_m))
                    x_m = decoder(x_m)
                    x_m_per_sample.append(x_m)
                paddings = self.placeholder.repeat(
                    1,
                    (max_channel_count - len(channel_ids)) * self.tokens_per_modality,
                    1,
                )
                x_m_per_sample = torch.cat(x_m_per_sample + [paddings], dim=1)
                x_m_per_sample = torch.cat(
                    [x[i : i + 1, :num_extra_tokens, :], x_m_per_sample], dim=1
                )
                x_m_s.append(x_m_per_sample)
            x_m_s = torch.cat(x_m_s, dim=0)

        return x_m_s

    def forward_masked(
        self,
        x: torch.Tensor,
        ind_restore: torch.Tensor,
        channel_indices: List[int],
        num_extra_tokens: int = 1,
        pos_embeddings: Optional[Tensor] = None,
        each_img_has_different_channels: bool = False,
    ) -> torch.Tensor:
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ind_restore.shape[1] + num_extra_tokens - x.shape[1], 1
        )
        # CAMAEDecoder: remove class token, then concat all unmasked tokens and mask tokens
        x_ = torch.cat([x[:, num_extra_tokens:, :], mask_tokens], dim=1)
        x_ = torch.gather(
            x_, dim=1, index=ind_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle
        x = torch.cat([x[:, :num_extra_tokens, :], x_], dim=1)  # add class token back
        x = self.forward(
            x,
            channel_indices,
            num_extra_tokens,
            pos_embeddings,
            each_img_has_different_channels=each_img_has_different_channels,
        )
        return x
