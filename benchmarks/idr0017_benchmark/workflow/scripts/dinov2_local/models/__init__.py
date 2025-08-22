# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

from . import vision_transformer as vits
from .new_backbone import new_backbone#, DinoVisionTransformerClassifier
from torch import nn
from copy import deepcopy

logger = logging.getLogger("dinov2")


def build_model(args, img_size=224):
    if "vit" in args.arch:
        vit_kwargs = dict(
            img_size=img_size,
            in_chans = args.in_chans,
            patch_size=args.patch_size,
            init_values=args.layerscale,
            ffn_layer=args.ffn_layer,
            block_chunks=args.block_chunks,
            qkv_bias=args.qkv_bias,
            proj_bias=args.proj_bias,
            ffn_bias=args.ffn_bias,
            num_register_tokens=args.num_register_tokens,
            interpolate_offset=args.interpolate_offset,
            interpolate_antialias=args.interpolate_antialias,
        )
        student = vits.__dict__[args.arch](
            **vit_kwargs,
            drop_path_rate=args.drop_path_rate,
            drop_path_uniform=args.drop_path_uniform,
        )
        embed_dim = student.embed_dim
    return student, embed_dim


def build_model_from_cfg(cfg):
    return build_model(cfg.student, img_size=cfg.crops.global_crops_size)


class NewClassifier(nn.Module):
    def __init__(self, model, model_embed_dimension, num_classes):
        super(NewClassifier, self).__init__()
        self.transformer = deepcopy(model)
        self.classifier = nn.Sequential(nn.Linear(model_embed_dimension, 256), nn.ReLU(), nn.Linear(256, num_classes))

    def forward(self, x):
        x = self.transformer(x)
        x = self.transformer.norm(x)
        x = self.classifier(x)
        return x