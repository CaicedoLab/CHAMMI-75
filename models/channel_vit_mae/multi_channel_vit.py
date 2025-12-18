import math
from functools import partial
from typing import Optional

from torch import Tensor
import torch
import torch.nn as nn
from einops import rearrange, repeat
import numpy as np
import random
import timm

from vit import Block
from model_utils import trunc_normal_, maybe_flatten_images
from mae_modules import ChAMAEViTDecoder, CAMAEDecoder
from loss_func import (
    compute_proxy_loss,
    MultiPosConLoss,
    SimCLRContrastiveLoss,
    FourierLoss,
)


class PatchEmbedPerChannel(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int] = (224, 224),
        patch_size: int = 16,
        max_in_channels: int = 3,
        embed_dim: int = 768,
        use_channel_tokens: bool = True,
        channel_tokens_init: str = "orthogonal",
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (
            (img_size[0] // patch_size) * (img_size[1] // patch_size) * max_in_channels
        )
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(
            1,
            embed_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size),
        )

        if use_channel_tokens:
            self.channel_tokens = nn.parameter.Parameter(
                torch.zeros(1, embed_dim, max_in_channels, 1, 1)
            )
            if channel_tokens_init == "orthogonal":
                orthogonal_tensor = torch.empty(embed_dim, max_in_channels)
                nn.init.orthogonal_(orthogonal_tensor)  # produces orthogonal columns
                with torch.no_grad():
                    self.channel_tokens[:, :, :, 0, 0].copy_(orthogonal_tensor)
            elif channel_tokens_init == "random":
                trunc_normal_(self.channel_tokens, std=0.02)
            elif channel_tokens_init == "zero":
                trunc_normal_(self.channel_tokens, std=0.02) / 1000  ## close to zero
                # pass  ## already initialized to zero
            else:
                raise ValueError(f"Unknown channel_tokens_init: {channel_tokens_init}")
        else:
            self.channel_tokens = None

    def forward(
        self,
        x: Tensor,
        channel_ids_list: list[list[int]] | None,
        valid_channel_masks: Optional[Tensor] = None,
        bag_of_channels_mode: bool = False,
    ):  ## return x, channel_token_patches
        if bag_of_channels_mode:
            return self.forward_single_channel(x, channel_ids_list)
        else:
            return self.forward_multi_channel(x, channel_ids_list, valid_channel_masks)

    def forward_single_channel(
        self, x: Tensor, channel_ids_list: list[list[int]] | None = None
    ):
        """Bag of channels mode.
        "channel_ids_list": list of channel_ids for each image.
            during training, each image has 1 channel E.g., channel_ids_list= [[3], [5], [2]...]
            during inference, each image can have multiple channels, e.g., channel_ids_list = [[3, 5], [2], ...]
        Training mode:
            each channel is treated as an independent image, x shape [c1 + c2 + ... + cb, 1, H, W], ci is the number of channels of image i in the batch.
        Inference mode:
            after processing each image in the batch independently, we concatenate the results using `num_channels`, which is a list of length batch_size, each element is the number of channels of the corresponding image.
        e.g., given 2 images, with 3 and 5 channels respectively, then x shape = [8, 1, H, W], num_channels = [3, 5]
        """
        # shared projection layer across channels
        if self.proj.weight.dtype != x.dtype:
            x = x.to(dtype=self.proj.weight.dtype)

        x = self.proj(x.unsqueeze(1))  # B Cout 1 H W, note B is c1 + c2 + ... + cb

        # # channel specific offsets
        if self.channel_tokens is not None:
            ## extract channel tokens for each channel based on channel_ids_list
            assert channel_ids_list is not None
            flat_idxs = [i for group in channel_ids_list for i in group]
            flat_idxs_tensor = torch.tensor(
                flat_idxs, dtype=torch.long, device=x.device
            )
            channel_tokens = torch.index_select(
                self.channel_tokens, dim=2, index=flat_idxs_tensor
            )
            channel_tokens = rearrange(
                channel_tokens, "1 d b 1 1 -> b d 1 1 1", b=flat_idxs_tensor.shape[0]
            )
            x += channel_tokens  # B Cout 1 H W
        else:
            channel_tokens = None
        # preparing the output sequence
        x = x.flatten(2)  # B Cout HW
        x = x.transpose(1, 2)  # B HW Cout
        return x, channel_tokens

    def forward_multi_channel(
        self,
        x: Tensor,
        channel_ids_list: list[list[int]] | None,
        valid_channel_masks: Optional[Tensor] = None,
    ):
        """
        channel_ids: list of `batch_size` elements, each indicates channels of the img.  E.g., [[3,  5], [2]]
        valid_channel_masks: Attention mask (bool) with False at the end to indicate channel padding, e.g., [[True, True, False], [True, False, False]]
        """
        REGULAR_CASE = channel_ids_list is None and valid_channel_masks is None
        SAME_SUBSET_CHANNELS_FOR_ALL_IMG = (
            channel_ids_list is not None and valid_channel_masks is None
        )
        DIFFERENT_CHANNELS_FOR_EACH_IMG = (
            channel_ids_list is not None and valid_channel_masks is not None
        )

        device = x.device

        ## get channel tokens for this batch
        if self.channel_tokens is not None:
            if (
                REGULAR_CASE
            ):  ## Assume all images in the batch have the same channels, no masks.
                channel_tokens = self.channel_tokens
            elif SAME_SUBSET_CHANNELS_FOR_ALL_IMG:  ## E.g., each img has 8 channels, but only 5 channels are used for each image in the batch
                channel_ids = channel_ids_list[0]  # type: ignore
                channel_ids_tensor = torch.tensor(
                    channel_ids, dtype=torch.long, device=device
                )
                channel_tokens = torch.index_select(
                    self.channel_tokens, dim=2, index=channel_ids_tensor
                )
            elif (
                DIFFERENT_CHANNELS_FOR_EACH_IMG
            ):  ## E.g., first image has 3 channels, second image has 5 channels, etc.
                ## get corresponding channel tokens for each image in the batch
                # 1. Flatten all indices and group size
                flat_idxs = [i for group in channel_ids_list for i in group]  # type: ignore
                flat_idxs_tensor = torch.tensor(
                    flat_idxs, dtype=torch.long, device=device
                )
                group_sizes = [len(group) for group in channel_ids_list]  # type: ignore

                # 2. Gather once along the channel token's dim (dim=2)
                #    result shape = [B, d, sum(group_sizes), 1, 1]
                selected_flat = torch.index_select(
                    self.channel_tokens, dim=2, index=flat_idxs_tensor
                )

                # 3. Split
                channel_tokens = list(torch.split(selected_flat, group_sizes, dim=2))

                # 4. padding to make channel_tokens the same size
                max_num_channels = max(group_sizes)
                dim = self.embed_dim
                channel_tokens = [
                    torch.cat(
                        [
                            ct,
                            torch.zeros(
                                1,
                                dim,
                                max_num_channels - ct.shape[2],
                                1,
                                1,
                                device=device,
                            ),
                        ],
                        dim=2,
                    )
                    for ct in channel_tokens
                ]
                channel_tokens = torch.cat(channel_tokens, dim=0)  # B Cout Cin 1 1
            else:
                raise ValueError(
                    f"Unknown case: channel_ids_list={channel_ids_list}, valid_channel_masks={valid_channel_masks}"
                )
        else:
            channel_tokens = None

        # shared projection layer across channels
        x = self.proj(x.unsqueeze(1))  # B Cout Cin H W

        # channel specific offsets
        if channel_tokens is not None:
            x += channel_tokens  # B Cout Cin H W

        # preparing the output sequence
        x = x.flatten(2)  # B Cout CinHW
        x = x.transpose(1, 2)  # B CinHW Cout
        return x, channel_tokens


class MultiChannelViT(nn.Module):
    def __init__(
        self,
        img_size=[224, 224],
        patch_size=16,
        in_chans=3,
        num_classes=0,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        attention_cls="regular",
        proxy_loss_lambda=1.0,
        proxy_temperature=0.11,
        proxy_orthogonal_init=True,
        supervised_contrastive_lambda=0.0,
        supervised_contrastive_temperature=0.1,
        simclr_lambda=0.0,
        simclr_temperature=0.1,
        mae_lambda=0.0,
        mae_loss_norm: Optional[str] = None,
        mae_recon_fourier_lambda: float = 0.0,
        mask_recon_fourier_loss: bool = True,
        training_sample=None,
        init_values: float | None = None,
        pretrained: str | None = None,
        use_cls_head=False,
        use_channel_tokens: bool = True,
        channel_tokens_init: str = "orthogonal",
        use_self_image_norm: bool = False,
        decoder: dict = None,
        **kwargs,
    ):
        super().__init__()
        self.num_features = self.embed_dim = self.out_dim = embed_dim
        self.max_in_channels = in_chans
        self.training_sample = (
            training_sample.upper() if training_sample is not None else None
        )
        self.proxy_orthogonal_init = proxy_orthogonal_init
        self.patch_size = patch_size

        if use_self_image_norm:
            self.image_norm = nn.InstanceNorm2d(
                None, affine=False, track_running_stats=False, eps=1e-5
            )
        else:
            self.image_norm = None

        self.patch_embed = PatchEmbedPerChannel(
            img_size=img_size,
            patch_size=patch_size,
            max_in_channels=self.max_in_channels,
            embed_dim=embed_dim,
            use_channel_tokens=use_channel_tokens,
            channel_tokens_init=channel_tokens_init,
        )
        self.num_patches_per_channel = (
            self.patch_embed.num_patches // self.max_in_channels
        )
        self.num_heads = num_heads
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.num_extra_tokens = 1  # cls token
        self.pos_embed = nn.Parameter(
            torch.zeros(
                1, self.num_patches_per_channel + self.num_extra_tokens, embed_dim
            )
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    attention_cls=attention_cls,
                    init_values=init_values,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # Classifier head
        if use_cls_head:
            self.cls_head = nn.Linear(embed_dim, num_classes)
        else:
            self.cls_head = nn.Identity()

        #### Define Losses
        self.use_proxy_loss = proxy_loss_lambda > 0
        self.proxy_loss_lambda = proxy_loss_lambda

        self.use_supcon_loss = supervised_contrastive_lambda > 0
        self.supcon_lambda = supervised_contrastive_lambda

        self.use_simclr_loss = simclr_lambda > 0
        self.simclr_lambda = simclr_lambda

        self.use_mae_loss = mae_lambda > 0
        self.mae_lambda = mae_lambda

        ## Proxy loss
        if self.use_proxy_loss:
            num_proxies = num_classes
            self.output_proxies = torch.nn.Parameter(
                (torch.randn(num_proxies, embed_dim) / 8)
            )
            if self.proxy_orthogonal_init:
                nn.init.orthogonal_(self.output_proxies)  ## initlaize orthogonally
            self.proxy_scale = np.sqrt(1.0 / proxy_temperature)

        ## Supervised Contrastive Loss
        if self.use_supcon_loss:
            self.compute_supcon_loss = MultiPosConLoss(
                temperature=supervised_contrastive_temperature
            )

        ## SimCLR Loss
        if self.use_simclr_loss:
            self.compute_simclr_loss = SimCLRContrastiveLoss(
                temperature=simclr_temperature
            )

        ## MAE Loss
        if self.use_mae_loss:
            self.mae_loss_norm = mae_loss_norm
            self.reconstruct_loss_fn = nn.MSELoss(reduction="none")
            self.recon_fourier_loss_fn = FourierLoss()
            self.mae_recon_fourier_lambda = mae_recon_fourier_lambda
            self.mask_recon_fourier_loss = mask_recon_fourier_loss
            self.mask_ratio_min = decoder["mask_ratio_min"]
            self.mask_ratio_max = decoder["mask_ratio_max"]

            self.decoder_dim = (
                decoder["embed_dim"] if decoder["embed_dim"] is not None else embed_dim
            )
            self.decoder_type = decoder["decoder_type"]
            decoder["num_channels"] = in_chans
            if decoder["decoder_type"] == "chamaevit_decoder":
                self.decoder = ChAMAEViTDecoder(
                    depth=decoder["depth"],
                    embed_dim=self.decoder_dim,
                    mlp_ratio=decoder["mlp_ratio"],
                    norm_layer=partial(
                        nn.LayerNorm, eps=float(decoder["norm_layer"]["eps"])
                    ),
                    num_heads=decoder["num_heads"],
                    qkv_bias=decoder["qkv_bias"],
                    num_channels=decoder["num_channels"],
                    attention_cls=attention_cls,
                )
            elif decoder["decoder_type"] == "camae_decoder":
                self.decoder = CAMAEDecoder(
                    depth=decoder["depth"],
                    embed_dim=self.decoder_dim,
                    mlp_ratio=decoder["mlp_ratio"],
                    norm_layer=partial(nn.LayerNorm, eps=decoder["norm_layer"]["eps"]),
                    num_heads=decoder["num_heads"],
                    num_modalities=decoder["num_channels"],
                    qkv_bias=decoder["qkv_bias"],
                    tokens_per_modality=self.num_patches_per_channel,
                )
            else:
                raise ValueError(f"Unknown decoder type: {decoder.decoder_type}")

            # projection layer between the encoder and decoder
            self.encoder_decoder_proj = nn.Linear(
                embed_dim, self.decoder_dim, bias=True
            )

            # linear layer from decoder embedding to input dims
            decoder_out_dim = patch_size**2
            self.decoder_pred = nn.Linear(
                self.decoder_dim,
                decoder_out_dim,
                bias=True,
            )

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

        self.directly_loaded = None
        if pretrained:
            self._load_pretrained_weights(pretrained)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _load_pretrained_weights(self, model_name: str):
        model_name = model_name.lower()
        EXCLUDED_LIST = None  # ["pos_embed"]
        if model_name.startswith("dinov2"):  ## e.g., dinov2_vits14
            pretrained_model = torch.hub.load("facebookresearch/dinov2", model_name)
        elif model_name.startswith("timm--"):
            model_name = model_name.replace(
                "timm--", ""
            )  ## e.g., timm--vit_small_patch16_224.augreg_in21k
            pretrained_model = timm.create_model(model_name, pretrained=True)
        else:
            raise NotImplementedError(f"{model_name} is not valid!")

        self.directly_loaded, _, _ = self._initialize_from_pretrained_model(
            pretrained_model,
            pretrained_name=model_name,
            exclude_list=EXCLUDED_LIST,
        )

    def freeze_pretrained_directly_loaded(self):
        if self.directly_loaded is None:
            print("[WARN] No directly loaded parameters to freeze.")
            return
        for name, param in self.named_parameters():
            if name in self.directly_loaded:
                param.requires_grad = False
                print(f"[INFO] Freezing parameter: {name}")
            else:
                print(f"[INFO] Not freezing parameter: {name}")

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def _initialize_from_pretrained_model(
        self,
        pretrained_model,
        pretrained_name: str,
        exclude_list: Optional[list] = None,
    ):
        print("-" * 50)
        if exclude_list is None:
            exclude_list = []

        pretrained_state = dict(pretrained_model.named_parameters())
        pretrained_state.update(dict(pretrained_model.named_buffers()))

        my_state = dict(self.named_parameters())
        my_state.update(dict(self.named_buffers()))

        directly_loaded = []  ## direcly copy parameters from pretrained model
        loaded = []  ## all loaded parameters, consisting of the directly loaded ones and the ones that are not directly loaded
        missing = []

        for name, my_param in my_state.items():
            if any(ex_key in name for ex_key in exclude_list):
                print(f"[SKIP] Excluded: {name}")
                continue

            if name not in pretrained_state:
                print(f"[WARN] {name} not found in {pretrained_name}.")
                continue

            pretrained_param = pretrained_state[name]

            # Special case: patch embedding weights
            if "patch_embed.proj.weight" in name and len(pretrained_param.shape) == 4:
                # DINO: [384, 3, 14, 14] → [384, 1, C, 14, 14]
                with torch.no_grad():
                    avg_weight = pretrained_param.mean(
                        dim=1, keepdim=True
                    )  # (384, 1, 14, 14)
                    my_param.copy_(avg_weight.unsqueeze(2))  #  (384, 1, 1, 14, 14)
                    loaded.append(name)
                continue
            # Special case: pos_embed
            if "pos_embed" in name and len(pretrained_param.shape) == 3:
                ##  torch.Size([1, 257, 384]) (our model) vs torch.Size([1, 1370, 384]) (dinov2)
                ##  we need to interpolate the pos_embed
                with torch.no_grad():
                    w, h = self.patch_embed.img_size[0], self.patch_embed.img_size[1]
                    total_tokens = pretrained_param.shape[1]
                    ## TODO: check on this
                    my_param.copy_(
                        self.interpolate_pos_encoding(
                            total_tokens, pretrained_param.shape[-1], w, h, c=1
                        )
                    )
                loaded.append(name)
                continue

            # Direct match
            if my_param.shape == pretrained_param.shape:
                with torch.no_grad():
                    my_param.copy_(pretrained_param)
                if "patch_embed.proj" not in name:
                    directly_loaded.append(name)
                loaded.append(name)
            else:
                print(
                    f"[SKIP] Shape mismatch: {name}, {my_param.shape} (our model) vs. {pretrained_param.shape} ({pretrained_name})"
                )
                missing.append(name)

        print(
            f"\n Loaded {len(loaded)} parameters from {pretrained_name} into MultiChannelViT."
        )
        print(f" Loaded keys: {loaded}")
        print(
            f" Missing {len(missing)} parameters from {pretrained_name} into MultiChannelViT."
        )
        print(f" Missing parameters: {missing}")
        print("-" * 50)

        return directly_loaded, loaded, missing

    def interpolate_pos_encoding(self, total_tokens, dim, w, h, nc):
        # number of auxilary dimensions before the patches
        num_extra_tokens = self.num_extra_tokens

        n_patches = total_tokens - num_extra_tokens
        n_patches_per_channel = self.pos_embed.shape[1] - num_extra_tokens
        if n_patches == n_patches_per_channel and w == h:
            return self.pos_embed

        class_pos_embed = self.pos_embed[:, :num_extra_tokens]
        patch_pos_embed = self.pos_embed[:, num_extra_tokens:]

        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(
                1,
                int(math.sqrt(n_patches_per_channel)),
                int(math.sqrt(n_patches_per_channel)),
                dim,
            ).permute(0, 3, 1, 2),
            scale_factor=(
                w0 / math.sqrt(n_patches_per_channel),
                h0 / math.sqrt(n_patches_per_channel),
            ),
            mode="bicubic",
        )

        assert (
            int(w0) == patch_pos_embed.shape[-2]
            and int(h0) == patch_pos_embed.shape[-1]
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, 1, -1, dim)

        # create copies of the positional embeddings for each channel
        patch_pos_embed = patch_pos_embed.expand(1, nc, -1, dim).reshape(1, -1, dim)

        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def generate_patch_masks_from_channel_masks(self, valid_channel_masks: Tensor):
        """
        Generate attention masks (patch level) for the channel masks
        + `valid_channel_masks`: B, C. Attention mask (bool) with False at the end to indicate channel padding. Example input [[True, True, False], [True, False, False]]

        + output (`patch_masks`): Expand each channel mask into `num_patches_per_channel` patches, and add a mask for the cls token.
        E.g., Assume `num_patches_per_channel`=2, output for the example input above would be
                    [[True, True, True, True, True, False False], [True, True, True, False, False, False, False]]
        """
        patch_masks = repeat(
            valid_channel_masks, "b j -> b (j c)", c=self.num_patches_per_channel
        )

        ## add masks for the cls token
        cls_mask = torch.ones(
            (patch_masks.shape[0], 1),
            dtype=patch_masks.dtype,
            device=patch_masks.device,
        )
        patch_masks = torch.cat([cls_mask, patch_masks], dim=1)
        B, L = patch_masks.shape
        patch_masks = patch_masks.view(B, 1, 1, L).bool()
        return patch_masks

    def generate_patch_masks_from_patch_sampling(self, patch_masks: Tensor):
        ## add masks for the cls token
        cls_mask = torch.ones(
            (patch_masks.shape[0], 1),
            dtype=patch_masks.dtype,
            device=patch_masks.device,
        )
        patch_masks = torch.cat([cls_mask, patch_masks], dim=1)
        B, L = patch_masks.shape
        patch_masks = patch_masks.view(B, 1, 1, L).bool()
        return patch_masks

    def prepare_tokens(
        self,
        x: Tensor,
        channel_ids_list: list[list[int]] | None,
        valid_channel_masks: Optional[Tensor],
        total_tokens: int,
        tokens_to_keep: Optional[Tensor],
        bag_of_channels_mode: bool = False,
    ):
        B, nc, w, h = x.shape

        ## patchify and embed
        x, channel_tokens = self.patch_embed(
            x,
            channel_ids_list,
            valid_channel_masks,
            bag_of_channels_mode=bag_of_channels_mode,
        )

        ## add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(total_tokens, self.embed_dim, w, h, nc)

        ## mask out some patches, resulting in fewer patches than x
        if tokens_to_keep is not None:
            tokens_to_keep = tokens_to_keep.unsqueeze(-1).repeat(1, 1, self.embed_dim)
            x_masked = torch.gather(
                x[:, self.num_extra_tokens :, :], dim=1, index=tokens_to_keep
            )
            ## add class token (and potentially meta tokens) back
            x = torch.cat([x[:, : self.num_extra_tokens, :], x_masked], dim=1)

        return self.pos_drop(x), channel_tokens

    def sample_patches(
        self,
        training_sample: str,
        batch_size: int,
        n_patches: int,
        device: torch.device,
        n_patches_per_channel: int,
        n_channels_origin: int,
        mask_ratio_min: float,
        mask_ratio_max: float,
        valid_channel_masks: Tensor | None = None,
    ):
        B, L = batch_size, n_patches
        if mask_ratio_min != mask_ratio_max:
            mask_ratio = np.random.uniform(mask_ratio_min, mask_ratio_max)
        else:
            mask_ratio = mask_ratio_min

        if (
            valid_channel_masks is not None
        ):  ### if each image has different number of channels, True is valid channels, False is paddings
            ## set the noise of the masked channels to be large value, so that they are not selected
            patch_masks = repeat(
                valid_channel_masks.bool(), "b c -> b (c n)", n=n_patches_per_channel
            )  ## B, L
            ### `patch_padding_masks``: True is valid patches, False is paddings

        if training_sample == "PATCH_RANDOM":
            n_patches_keep = int(
                (1 - mask_ratio) * n_patches_per_channel * n_channels_origin
            )
            ## generate `noise` to keep patches with smallest noise in each channel
            noise = torch.rand(B, L, device=device)
        elif training_sample == "PATCH_BY_CHANNEL":
            ## generate `noise` to keep patches with smallest noise in each channel
            constant_noise = torch.ones(B, L, device=device)
            keep_per_channel = int((1 - mask_ratio) * n_patches_per_channel)
            noise = torch.rand(
                B, n_channels_origin, n_patches_per_channel, device=device
            )
            keep_idxs = torch.argsort(noise, dim=-1)[:, :, :keep_per_channel]
            keep_idxs = (
                keep_idxs
                + torch.arange(0, n_patches, n_patches_per_channel, device=device)[
                    :, None
                ]
            )
            keep_idxs = rearrange(keep_idxs, "b c p -> b (c p)")
            constant_noise[torch.arange(B).unsqueeze(1), keep_idxs] = 0
            noise = constant_noise
            n_patches_keep = keep_per_channel * n_channels_origin
        else:
            raise ValueError(f"Unknown patch training_sample: {training_sample}")

        if valid_channel_masks is not None:
            LARGE_VALUE = 1000000
            noise += (
                ~patch_masks * LARGE_VALUE
            )  ## set noise of masked channels to be large value

        #### Create mask: 0 is keep (i.e., visible), 1 is remove (ie, mask out, used to compute reconstruction loss)
        shuffled_tokens = torch.argsort(noise, dim=1)  # shuffled index
        ind_restore = torch.argsort(shuffled_tokens, dim=1)  # unshuffled index

        #### get masked input
        if (
            valid_channel_masks is not None
        ):  ## each image has different number of channels
            n_patches_keep_list = [
                int((1 - mask_ratio) * n_patches_per_channel)
                * valid_channel_masks[i].sum().item()
                for i in range(B)
            ]
            tokens_to_keep = torch.zeros(
                B, max(n_patches_keep_list), dtype=torch.long, device=device
            )

            max_patches_keep = max(n_patches_keep_list)
            mae_patch_masks = torch.zeros(
                B, max_patches_keep, dtype=torch.bool, device=device
            )  ## True is valid patches, False is paddings

            for i, n_patches_keep_i in enumerate(n_patches_keep_list):
                tokens_to_keep[i, :n_patches_keep_i] = shuffled_tokens[
                    i, :n_patches_keep_i
                ]
                mae_patch_masks[i, :n_patches_keep_i] = 1  ## valid patches

        else:
            tokens_to_keep = shuffled_tokens[
                :, :n_patches_keep
            ]  # keep the first n_patches_keep indices
            mae_patch_masks = None
        # x_masked = torch.gather(x, dim=1, index=tokens_to_keep.unsqueeze(-1).repeat(1, 1, D))

        # get binary mask used for loss masking: 0 is keep, 1 is remove

        mask = torch.ones([B, L], device=device)
        if valid_channel_masks is not None:
            for i, n_patches_keep_i in enumerate(n_patches_keep_list):
                mask[i, :n_patches_keep_i] = (
                    0  ## we don't compute loss on the visible patches
                )
                padding_start = (
                    valid_channel_masks[i].sum().item() * n_patches_per_channel
                )
                mask[i, padding_start:] = (
                    0  ## we don't compute loss on the padded patches either
                )
        else:
            mask[:, :n_patches_keep] = 0
        mask = torch.gather(
            mask, dim=1, index=ind_restore
        )  # unshuffle to get the binary mask

        res = {
            "tokens_to_keep": tokens_to_keep,
            "mask": mask,
            "ind_restore": ind_restore,
            "channels_sampled": None,
            "n_patches_keep": n_patches_keep,
            "mae_patch_masks": mae_patch_masks,
        }
        return res

    def channel_dropout(
        self,
        x,
        channel_sample: str,
        channel_ids_list: list[list[int]] | None,
        valid_channel_masks: Optional[Tensor],
    ) -> dict[str, Tensor | list]:
        REGULAR_CASE = channel_ids_list is None and valid_channel_masks is None
        SAME_SUBSET_CHANNELS_FOR_ALL_IMG = (
            channel_ids_list is not None and valid_channel_masks is None
        )
        DIFFERENT_CHANNELS_FOR_EACH_IMG = (
            channel_ids_list is not None and valid_channel_masks is not None
        )

        res = {"x": None, "channel_ids_list": None, "patch_masks": None}

        Cin = x.shape[1]
        if (
            REGULAR_CASE
        ):  ##  all images have the same channels, use all channels, no masks.
            if channel_sample == "HCS":
                Cin_new = random.randint(1, Cin)
                channel_indices = torch.tensor(
                    random.sample(range(Cin), k=Cin_new),
                    device=x.device,
                    dtype=torch.long,
                )
                res["x"] = x[:, channel_indices]
                return res
            else:
                raise ValueError(f"Unknown channel sampling method: {channel_sample}")
        elif SAME_SUBSET_CHANNELS_FOR_ALL_IMG:
            ## TODO: work on this later
            raise NotImplementedError(
                "Channel sampling is not implemented for this case."
            )
        elif DIFFERENT_CHANNELS_FOR_EACH_IMG:
            #### E.g., first image has 3 channels, second image has 5 channels, etc.
            ## get Cin_new for each image in the batch

            ### for SimCLR and Supervised Contrastive Learning, we have pair of positive samples,
            ### thus one option is to do channel sampling consistently for both samples in the pair
            ## (i.e., do channel sampling for one view, and use the same channel indices for the other view)
            if channel_sample == "HCS_SYMMETRIC":
                assert self.use_simclr_loss or self.use_supcon_loss, (
                    "HCS_SYMMETRIC only works with simclr or supcon loss!"
                )
                channel_ids_list = channel_ids_list[
                    0 : len(channel_ids_list) // 2
                ]  ## only take the first half

            channel_ids_list_new = []
            channel_indices_list_new = []
            for channel_ids in channel_ids_list:
                if channel_sample.startswith("HCS"):
                    Cin = len(channel_ids)
                    Cin_new = random.randint(1, Cin)
                    channel_indices_new = torch.tensor(
                        random.sample(range(Cin), k=Cin_new),
                        device=x.device,
                        dtype=torch.long,
                    )
                    channel_indices_list_new.append(channel_indices_new)
                    channel_ids_new = [
                        channel_ids[i] for i in channel_indices_new.tolist()
                    ]
                    channel_ids_list_new.append(channel_ids_new)

            if channel_sample == "HCS_SYMMETRIC":
                channel_ids_list_new = channel_ids_list_new + channel_ids_list_new
                channel_indices_list_new = (
                    channel_indices_list_new + channel_indices_list_new
                )

            ## get the new images and masks
            max_Cin_new = max(
                [len(channel_ids) for channel_ids in channel_ids_list_new]
            )
            images_new = torch.zeros(
                (x.shape[0], max_Cin_new, x.shape[2], x.shape[3]), device=x.device
            )
            channel_masks_new = torch.zeros(
                (x.shape[0], max_Cin_new), dtype=torch.bool, device=x.device
            )

            for i, channel_ids in enumerate(channel_ids_list_new):
                channel_indices = channel_indices_list_new[i]
                images_new[i, : len(channel_indices), :, :] = x[
                    i, channel_indices, :, :
                ]
                channel_masks_new[i, : len(channel_ids)] = True

            res = {
                "x": images_new,
                "channel_ids_list": channel_ids_list_new,
                "patch_masks": channel_masks_new,
            }
            return res
        else:
            raise ValueError(
                f"Unknown case: channel_ids_list={channel_ids_list}, patch_masks={valid_channel_masks}"
            )

    def _norm_pix_loss(self, target: Tensor):
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.0e-6) ** 0.5
        return target

    def compute_mae_loss(
        self,
        reconstruction: Tensor,
        img: Tensor,
        mask: Tensor,
    ) -> dict[str, Tensor]:
        """Computes MAE loss"""
        mae_loss_dict = {}
        num_channels = img.shape[1]
        if self.mae_loss_norm == "norm_pix_loss":
            ## flat first, then norm per patch (based on MAE original paper)
            target_flattened = maybe_flatten_images(
                img, patch_size=self.patch_size, channel_agnostic=True
            )
            target_flattened = self._norm_pix_loss(target_flattened)
        # elif self.mae_loss_norm == "instance_norm":  ## norm first, then flat (based on CAMAE paper)
        #     img = self.image_norm(img) ## check to initialize this
        #     target_flattened = maybe_flatten_images(img, patch_size=self.patch_size, channel_agnostic=True)
        elif self.mae_loss_norm is None:
            target_flattened = maybe_flatten_images(
                img, patch_size=self.patch_size, channel_agnostic=True
            )

        ## img: b c h p1 w p2
        ## target_flattened: b (c p1 p2) (h w), where p1=p2=#patches, h=w=patch_size. h*w is the #pixels in a patch (ie, patch dim)
        ## e.g., for jumpcp, target_flattened shape = torch.Size([128, 8*14*14, 16*16])
        ## Should be with MSE or MAE (L1) with reduction='none'
        loss = self.reconstruct_loss_fn(reconstruction, target_flattened)
        loss = loss.mean(
            dim=-1
        )  # average over embedding dim -> mean loss per patch (N,L)

        loss = (loss * mask).sum() / mask.sum()  # mean loss on masked patches only
        mae_loss_dict["mae_img_loss"] = loss
        # compute fourier loss
        if self.mae_recon_fourier_lambda > 0:
            floss = self.recon_fourier_loss_fn(
                reconstruction, target_flattened, num_channels
            )  ## B, L, C
            if not self.mask_recon_fourier_loss:
                floss = floss.mean()
            else:
                floss = floss.mean(dim=-1)
                floss = (floss * mask).sum() / mask.sum()
        else:
            floss = torch.tensor(0.0, device=img.device)
        mae_loss_dict["mae_fourier_loss"] = floss

        return mae_loss_dict

    def forward_decoder(
        self,
        x_latent: Tensor,
        position_embeddings: Tensor,
        ind_restore: Tensor,
        channel_indices: list[int] = None,
        each_img_has_different_channels: bool = False,
        channel_token_patches: Tensor | None = None,
        full_patch_masks: Tensor | None = None,
    ):
        cur_num_extra_tokens = self.num_extra_tokens
        decoder_latent = self.encoder_decoder_proj(x_latent)

        if self.decoder_type == "camae_decoder":
            decoder_tokens = self.decoder.forward_masked(
                decoder_latent,
                ind_restore=ind_restore,
                channel_indices=channel_indices,
                num_extra_tokens=cur_num_extra_tokens,
                pos_embeddings=position_embeddings,
                each_img_has_different_channels=each_img_has_different_channels,
            )
        elif self.decoder_type == "chamaevit_decoder":
            decoder_tokens = self.decoder.forward_masked(
                decoder_latent,
                ind_restore=ind_restore,
                pos_embeddings=position_embeddings,
                channel_token_patches=channel_token_patches,
                num_extra_tokens=cur_num_extra_tokens,
                full_patch_masks=full_patch_masks,
            )  # decoder.embed_dim
        else:
            raise ValueError(f"Unknown decoder type: {self.decoder_type}")

        predicted_reconstruction = self.decoder_pred(
            decoder_tokens
        )  # linear projection to input
        reconstruction = predicted_reconstruction[
            :, cur_num_extra_tokens:, :
        ]  # drop class token and meta tokens

        return reconstruction

    def forward(
        self,
        x: Tensor,
        channel_ids_list: list[list[int]] | None = None,
        valid_channel_masks: Optional[Tensor] = None,
        y: Optional[Tensor] = None,
        bag_of_channels_mode: bool = False,
    ):
        if self.image_norm is not None:
            x = self.image_norm(x)

        x_origin = x.clone()  ## for reconstruction loss if any
        b, c, w, h = x.shape

        n_patches = self.num_patches_per_channel * c
        total_tokens = n_patches + self.num_extra_tokens

        ### channel dropout, used in ChannelViT
        if self.training and self.training_sample in ["HCS", "HCS_SYMMETRIC"]:
            sample_res = self.channel_dropout(
                x,
                channel_sample=self.training_sample,
                channel_ids_list=channel_ids_list,
                valid_channel_masks=valid_channel_masks,
            )
            x, channel_ids_list, valid_channel_masks = (
                sample_res["x"],
                sample_res["channel_ids_list"],
                sample_res["channel_masks"],
            )

        ### generate patch masks from channel masks if any
        if valid_channel_masks is not None:
            full_patch_masks = self.generate_patch_masks_from_channel_masks(
                valid_channel_masks
            )
        else:
            full_patch_masks = None

        ### sampling for MAEs
        tokens_to_keep = None
        sample_strategy = self.training_sample
        if self.training and sample_strategy not in [None, "HCS", "HCS_SYMMETRIC"]:
            if sample_strategy in ["PATCH_RANDOM", "PATCH_BY_CHANNEL"]:
                ### patch sampling, used in MAEs
                sample_res = self.sample_patches(
                    training_sample=sample_strategy,
                    batch_size=b,
                    n_patches=c * self.num_patches_per_channel,
                    device=x.device,
                    n_patches_per_channel=self.num_patches_per_channel,
                    n_channels_origin=c,
                    mask_ratio_min=self.mask_ratio_min,
                    mask_ratio_max=self.mask_ratio_max,
                    valid_channel_masks=valid_channel_masks,
                )
                tokens_to_keep = sample_res["tokens_to_keep"]
                mae_patch_masks = sample_res["mae_patch_masks"]
            else:
                raise ValueError(f"Unknown training_sample: {sample_strategy}")

        x, channel_tokens = self.prepare_tokens(
            x,
            channel_ids_list,
            valid_channel_masks,
            total_tokens=total_tokens,
            tokens_to_keep=tokens_to_keep,
            bag_of_channels_mode=bag_of_channels_mode,
        )

        forward_patch_masks = full_patch_masks
        if not bag_of_channels_mode and self.training:
            if sample_strategy in ["PATCH_RANDOM", "PATCH_BY_CHANNEL"]:
                ## if we do patch sampling, we have the patch masks for visible patches
                forward_patch_masks = self.generate_patch_masks_from_patch_sampling(
                    mae_patch_masks
                )
                ## other cases, we use the full patch masks (if any)

        for blk in self.blocks:
            x = blk(x, mask=forward_patch_masks)
        x = self.norm(x)
        out = self.cls_head(x[:, 0]).clone()

        ############## Pass into a decoder if any (e.g., for MAE reconstruction)
        if self.training and self.use_mae_loss:
            if channel_tokens is not None:
                channel_token_patches = repeat(
                    channel_tokens,
                    "b d c 1 1 -> b d c p",
                    p=self.num_patches_per_channel,
                )
                channel_token_patches = rearrange(
                    channel_token_patches, "b d c p -> b (c p) d"
                )  ## b, c*p1*p2, d
            else:
                channel_token_patches = None
            reconstruction = self.forward_decoder(
                x_latent=x,
                position_embeddings=self.interpolate_pos_encoding(
                    total_tokens, self.embed_dim, w, h, c
                ),
                ind_restore=sample_res["ind_restore"],
                channel_indices=channel_ids_list,
                each_img_has_different_channels=valid_channel_masks is not None,
                channel_token_patches=channel_token_patches,
                full_patch_masks=full_patch_masks,
            )

        ############ Bag of Channels: during inference, if bag_of_channels_mode, we need to group the output back to the original multi-channel images
        if not self.training and bag_of_channels_mode:
            assert channel_ids_list is not None
            group_sizes = [len(group) for group in channel_ids_list]
            out = list(torch.split(out, group_sizes, dim=0))

            ## assume we run inference on images with the same number of channels
            out = torch.stack(out, dim=0)  ## b, c, d
            out = rearrange(out, "b c d -> b (c d)")  ## b, c*d

            ## otherwise, just return a list of tensors, each tensor is of shape (ci, d), ci is the number of channels of image i,
            ## but we keep it simple for now.

        ############## return predictions + losses if any
        res = {"output": out}

        ############## compute losses during training
        if self.training:
            ## compute proxy loss
            if self.use_proxy_loss:
                proxy_loss = compute_proxy_loss(
                    proxies=self.output_proxies,
                    img_emb=out,
                    gt_imgs=y,
                    scale=self.proxy_scale,
                )
                res["proxy_loss"] = proxy_loss
            else:
                res["proxy_loss"] = torch.tensor(0.0)

            ## compute supervised contrastive loss
            if self.use_supcon_loss:
                sc_loss = self.compute_supcon_loss(feats=out, labels=y)
                res["supcon_loss"] = sc_loss
            else:
                res["supcon_loss"] = torch.tensor(0.0)

            ## compute SimCLR loss
            if self.use_simclr_loss:
                simclr_loss = self.compute_simclr_loss(feats=out)
                res["simclr_loss"] = simclr_loss
            else:
                res["simclr_loss"] = torch.tensor(0.0)

            ## compute MAE loss
            if self.use_mae_loss:
                ######## reconstruct img
                ## reconstruction: (b, c * n_patches, patch_size**2)
                ## x_origin: (b, c, h, w),    (b, d, c, n_pathes)

                recon_loss_dict = self.compute_mae_loss(
                    reconstruction, x_origin.float(), sample_res["mask"]
                )
                lambda_f = self.mae_recon_fourier_lambda
                recon_loss_dict["mae_loss"] = (1 - lambda_f) * recon_loss_dict[
                    "mae_img_loss"
                ] + lambda_f * recon_loss_dict["mae_fourier_loss"]
                res.update(recon_loss_dict)
            else:
                for k in ["mae_img_loss", "mae_fourier_loss", "mae_loss"]:
                    res[k] = torch.tensor(0.0)

            ## final loss
            res["loss"] = (
                self.proxy_loss_lambda * res["proxy_loss"]
                + self.supcon_lambda * res["supcon_loss"]
                + self.simclr_lambda * res["simclr_loss"]
                + self.mae_lambda * res["mae_loss"]
            )

        return res

    def get_last_selfattention(
        self,
        x,
        channel_ids_list: list[list[int]] | None = None,
        valid_channel_masks: Optional[Tensor] = None,
        tokens_to_keep: Optional[Tensor] = None,
        bag_of_channels_mode: bool = False,
    ):
        x = self.prepare_tokens(
            x,
            channel_ids_list,
            valid_channel_masks,
            tokens_to_keep,
            bag_of_channels_mode=bag_of_channels_mode,
        )
        patch_masks = (
            self.generate_patch_masks(valid_channel_masks)
            if valid_channel_masks is not None
            else None
        )

        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x, mask=patch_masks)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(
        self,
        x,
        channel_ids_list: list[list[int]] | None = None,
        valid_channel_masks: Optional[Tensor] = None,
        n=1,
        tokens_to_keep: Optional[Tensor] = None,
        bag_of_channels_mode: bool = False,
    ):
        x = self.prepare_tokens(
            x,
            channel_ids_list,
            valid_channel_masks,
            tokens_to_keep,
            bag_of_channels_mode=bag_of_channels_mode,
        )
        patch_masks = (
            self.generate_patch_masks(valid_channel_masks)
            if valid_channel_masks is not None
            else None
        )

        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, mask=patch_masks)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


def get_multi_channel_vit(model_size: str, patch_size=16, **kwargs):
    if model_size == "tiny":
        model = MultiChannelViT(
            patch_size=patch_size,
            embed_dim=192,
            depth=12,
            num_heads=3,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **kwargs,
        )
    elif model_size == "small":
        model = MultiChannelViT(
            patch_size=patch_size,
            embed_dim=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **kwargs,
        )
    elif model_size == "base":
        model = MultiChannelViT(
            patch_size=patch_size,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown model name: {model_size}")
    return model
