import math
from functools import partial
from typing import Optional

from torch import Tensor
import torch
import torch.nn as nn
from einops import rearrange, repeat
import random
import timm

from vit import Block
from model_utils import trunc_normal_
from loss_func import compute_proxy_loss, MultiPosConLoss, SimCLRContrastiveLoss
import numpy as np


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
        channel_masks: Optional[Tensor] = None,
        bag_of_channels_mode: bool = False,
    ):
        if bag_of_channels_mode:
            return self.forward_single_channel(x, channel_ids_list)
        else:
            return self.forward_multi_channel(x, channel_ids_list, channel_masks)

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

        # preparing the output sequence
        x = x.flatten(2)  # B Cout HW
        x = x.transpose(1, 2)  # B HW Cout
        return x

    def forward_multi_channel(
        self,
        x: Tensor,
        channel_ids_list: list[list[int]] | None,
        channel_masks: Optional[Tensor] = None,
    ):
        """
        channel_ids: list of `batch_size` elements, each indicates channels of the img.  E.g., [[3,  5], [2]]
        channel_masks: Attention mask (bool) with False at the end to indicate channel padding, e.g., [[True, True, False], [True, False, False]]
        """
        REGULAR_CASE = channel_ids_list is None and channel_masks is None
        SAME_SUBSET_CHANNELS_FOR_ALL_IMG = (
            channel_ids_list is not None and channel_masks is None
        )
        DIFFERENT_CHANNELS_FOR_EACH_IMG = (
            channel_ids_list is not None and channel_masks is not None
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
                    f"Unknown case: channel_ids_list={channel_ids_list}, channel_masks={channel_masks}"
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
        return x


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
        channel_sample="",
        init_values: float | None = None,
        pretrained: str | None = None,
        use_cls_head=False,
        use_channel_tokens: bool = True,
        channel_tokens_init: str = "orthogonal",
        use_self_image_norm: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.num_features = self.embed_dim = self.out_dim = embed_dim
        self.max_in_channels = in_chans
        self.channel_sample = channel_sample
        self.proxy_orthogonal_init = proxy_orthogonal_init

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
                    my_param.copy_(
                        self.interpolate_pos_encoding(pretrained_param, w, h, c=1)
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

    def interpolate_pos_encoding(self, x, w, h, c):
        # number of auxilary dimensions before the patches
        if not hasattr(self, "num_extra_tokens"):
            # backward compatibility
            num_extra_tokens = 1
        else:
            num_extra_tokens = self.num_extra_tokens

        npatch = x.shape[1] - num_extra_tokens
        N = self.pos_embed.shape[1] - num_extra_tokens

        if npatch == N and w == h:
            return self.pos_embed

        class_pos_embed = self.pos_embed[:, :num_extra_tokens]
        patch_pos_embed = self.pos_embed[:, num_extra_tokens:]

        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(
                1, int(math.sqrt(N)), int(math.sqrt(N)), dim
            ).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )
        assert (
            int(w0) == patch_pos_embed.shape[-2]
            and int(h0) == patch_pos_embed.shape[-1]
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, 1, -1, dim)

        # create copies of the positional embeddings for each channel
        patch_pos_embed = patch_pos_embed.expand(1, c, -1, dim).reshape(1, -1, dim)

        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def generate_patch_masks(self, channel_masks: Tensor):
        """
        Generate attention masks (patch level) for the channel masks
        + `channel_masks`: B, C. Attention mask (bool) with False at the end to indicate channel padding. Example input [[True, True, False], [True, False, False]]

        + output (`patch_masks`): Expand each channel mask into `num_patches_per_channel` patches, and add a mask for the cls token.
        E.g., Assume `num_patches_per_channel`=2, output for the example input above would be
                    [[True, True, True, True, True, False False], [True, True, True, False, False, False, False]]
        """
        patch_masks = repeat(
            channel_masks, "b j -> b (j c)", c=self.num_patches_per_channel
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

    def prepare_tokens(
        self,
        x: Tensor,
        channel_ids_list: list[list[int]] | None,
        channel_masks: Optional[Tensor],
        bag_of_channels_mode: bool = False,
    ):
        B, nc, w, h = x.shape
        x = self.patch_embed(
            x,
            channel_ids_list,
            channel_masks,
            bag_of_channels_mode=bag_of_channels_mode,
        )

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h, nc)

        return self.pos_drop(x)

    def sample_channels(
        self,
        x,
        channel_ids_list: list[list[int]] | None,
        channel_masks: Optional[Tensor],
    ) -> dict[str, Tensor | list]:
        REGULAR_CASE = channel_ids_list is None and channel_masks is None
        SAME_SUBSET_CHANNELS_FOR_ALL_IMG = (
            channel_ids_list is not None and channel_masks is None
        )
        DIFFERENT_CHANNELS_FOR_EACH_IMG = (
            channel_ids_list is not None and channel_masks is not None
        )

        res = {"x": None, "channel_ids_list": None, "channel_masks": None}

        Cin = x.shape[1]
        if (
            REGULAR_CASE
        ):  ##  all images have the same channels, use all channels, no masks.
            if self.channel_sample.lower() == "hcs":
                Cin_new = random.randint(1, Cin)
                channel_indices = torch.tensor(
                    random.sample(range(Cin), k=Cin_new),
                    device=x.device,
                    dtype=torch.long,
                )
                res["x"] = x[:, channel_indices]
                return res
            else:
                raise ValueError(
                    f"Unknown channel sampling method: {self.channel_sample}"
                )
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
            if self.channel_sample.lower() == "hcs_symmetric":
                assert self.use_simclr_loss or self.use_supcon_loss, (
                    "hcs_symmetric only works with simclr or supcon loss!"
                )
                channel_ids_list = channel_ids_list[
                    0 : len(channel_ids_list) // 2
                ]  ## only take the first half

            channel_ids_list_new = []
            channel_indices_list_new = []
            for channel_ids in channel_ids_list:
                if self.channel_sample.lower().startswith("hcs"):
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

            if self.channel_sample.lower() == "hcs_symmetric":
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
                "channel_masks": channel_masks_new,
            }
            return res
        else:
            raise ValueError(
                f"Unknown case: channel_ids_list={channel_ids_list}, channel_masks={channel_masks}"
            )

    def forward(
        self,
        x: Tensor,
        channel_ids_list: list[list[int]] | None = None,
        channel_masks: Optional[Tensor] = None,
        y: Optional[Tensor] = None,
        bag_of_channels_mode: bool = False,
    ):
        if self.image_norm is not None:
            x = self.image_norm(x)

        if self.training and self.channel_sample is not None:
            sample_res = self.sample_channels(x, channel_ids_list, channel_masks)
            x, channel_ids_list, channel_masks = (
                sample_res["x"],
                sample_res["channel_ids_list"],
                sample_res["channel_masks"],
            )

        x = self.prepare_tokens(
            x,
            channel_ids_list,
            channel_masks,
            bag_of_channels_mode=bag_of_channels_mode,
        )

        if not bag_of_channels_mode and channel_masks is not None:
            patch_masks = self.generate_patch_masks(channel_masks)
        else:
            patch_masks = None

        for blk in self.blocks:
            x = blk(x, mask=patch_masks)
        x = self.norm(x)
        out = self.cls_head(x[:, 0]).clone()

        if not self.training and bag_of_channels_mode:
            ## during inference, if bag_of_channels_mode, we need to group the output back to the original multi-channel images
            assert channel_ids_list is not None
            group_sizes = [len(group) for group in channel_ids_list]
            out = list(torch.split(out, group_sizes, dim=0))

            ## assume we run inference on images with the same number of channels
            out = torch.stack(out, dim=0)  ## b, c, d
            out = rearrange(out, "b c d -> b (c d)")  ## b, c*d

            ## otherwise, just return a list of tensors, each tensor is of shape (ci, d), ci is the number of channels of image i,
            ## but we keep it simple for now.

        res = {"output": out}

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

            ## final loss
            res["loss"] = (
                self.proxy_loss_lambda * res["proxy_loss"]
                + self.supcon_lambda * res["supcon_loss"]
                + self.simclr_lambda * res["simclr_loss"]
            )

        return res

    def get_last_selfattention(
        self,
        x,
        channel_ids_list: list[list[int]] | None = None,
        channel_masks: Optional[Tensor] = None,
        bag_of_channels_mode: bool = False,
    ):
        x = self.prepare_tokens(
            x,
            channel_ids_list,
            channel_masks,
            bag_of_channels_mode=bag_of_channels_mode,
        )
        patch_masks = (
            self.generate_patch_masks(channel_masks)
            if channel_masks is not None
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
        channel_masks: Optional[Tensor] = None,
        n=1,
        bag_of_channels_mode: bool = False,
    ):
        x = self.prepare_tokens(
            x,
            channel_ids_list,
            channel_masks,
            bag_of_channels_mode=bag_of_channels_mode,
        )
        patch_masks = (
            self.generate_patch_masks(channel_masks)
            if channel_masks is not None
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
    elif model_size == "large":
        model = MultiChannelViT(
            patch_size=patch_size,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown model name: {model_size}")
    return model
