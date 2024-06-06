from functools import partial
import numpy as np

import torch
import torch.nn as nn

from ts2.models.vit import VisionTransformer
from typing import Dict


class ViTBackbone(VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """

    def __init__(self, img_size: int, patch_size: int, **kwargs):
        super(ViTBackbone, self).__init__(img_size=img_size,
                                          patch_size=patch_size,
                                          **kwargs)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_out = self.embed_dim


def get_vit_backbone(which: str = "vit_base", params: Dict = {}):
    if which == "vit_tiny":
        return ViTBackbone(num_classes=0,
                           embed_dim=192,
                           depth=12,
                           num_heads=3,
                           mlp_ratio=4,
                           qkv_bias=True,
                           norm_layer=partial(nn.LayerNorm, eps=1e-6),
                           **params)
    elif which == "vit_small":
        return ViTBackbone(num_classes=0,
                           embed_dim=384,
                           depth=12,
                           num_heads=6,
                           mlp_ratio=4,
                           qkv_bias=True,
                           norm_layer=partial(nn.LayerNorm, eps=1e-6),
                           **params)
    elif which == "vit_base":
        return ViTBackbone(num_classes=0,
                           embed_dim=768,
                           depth=12,
                           num_heads=12,
                           mlp_ratio=4,
                           qkv_bias=True,
                           norm_layer=partial(nn.LayerNorm, eps=1e-6),
                           **params)
    elif which == "vit_large":
        return ViTBackbone(num_classes=0,
                           embed_dim=1024,
                           depth=24,
                           num_heads=16,
                           mlp_ratio=4,
                           qkv_bias=True,
                           norm_layer=partial(nn.LayerNorm, eps=1e-6),
                           **params)
    elif which == "vit_huge":
        return ViTBackbone(num_classes=0,
                           embed_dim=1280,
                           depth=32,
                           num_heads=16,
                           mlp_ratio=4,
                           qkv_bias=True,
                           norm_layer=partial(nn.LayerNorm, eps=1e-6),
                           **params)
    elif which == "vit":
        return ViTBackbone(**params)
    else:
        raise ValueError(
            "ViT backbone name must be in [tiny, small, base, large, huge]")
