# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
import math
from functools import partial
import numpy as np

import torch
import torch.nn as nn

import timm.models.vision_transformer

from typing import Dict

#class Attention(nn.Module):
#
#    def __init__(self,
#                 dim,
#                 num_heads=8,
#                 qkv_bias=False,
#                 qk_scale=None,
#                 attn_drop=0.,
#                 proj_drop=0.):
#        super().__init__()
#        self.num_heads = num_heads
#        head_dim = dim // num_heads
#        self.scale = qk_scale or head_dim**-0.5
#
#        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#        self.attn_drop = nn.Dropout(attn_drop)
#        self.proj = nn.Linear(dim, dim)
#        self.proj_drop = nn.Dropout(proj_drop)
#
#    def forward(self, x):
#        B, N, C = x.shape
#        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
#                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
#        q, k, v = qkv[0], qkv[1], qkv[2]
#
#        attn = (q @ k.transpose(-2, -1)) * self.scale
#        attn = attn.softmax(dim=-1)
#        attn = self.attn_drop(attn)
#
#        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#        x = self.proj(x)
#        x = self.proj_drop(x)
#        return x, attn
#
#
#def drop_path(x, drop_prob: float = 0., training: bool = False):
#    if drop_prob == 0. or not training:
#        return x
#    keep_prob = 1 - drop_prob
#    shape = (x.shape[0], ) + (1, ) * (
#        x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
#    random_tensor = keep_prob + torch.rand(
#        shape, dtype=x.dtype, device=x.device)
#    random_tensor.floor_()  # binarize
#    output = x.div(keep_prob) * random_tensor
#    return output
#
#
#class DropPath(nn.Module):
#    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
#    """
#
#    def __init__(self, drop_prob=None):
#        super(DropPath, self).__init__()
#        self.drop_prob = drop_prob
#
#    def forward(self, x):
#        return drop_path(x, self.drop_prob, self.training)
#
#
#class MLP(nn.Module):
#
#    def __init__(self,
#                 in_features,
#                 hidden_features=None,
#                 out_features=None,
#                 act_layer=nn.GELU,
#                 drop=0.):
#        super().__init__()
#        out_features = out_features or in_features
#        hidden_features = hidden_features or in_features
#        self.fc1 = nn.Linear(in_features, hidden_features)
#        self.act = act_layer()
#        self.fc2 = nn.Linear(hidden_features, out_features)
#        self.drop = nn.Dropout(drop)
#
#    def forward(self, x):
#        x = self.fc1(x)
#        x = self.act(x)
#        x = self.drop(x)
#        x = self.fc2(x)
#        x = self.drop(x)
#        return x
#
#
#class Block(nn.Module):
#
#    def __init__(self,
#                 dim,
#                 num_heads,
#                 mlp_ratio=4.,
#                 qkv_bias=False,
#                 qk_scale=None,
#                 drop=0.,
#                 attn_drop=0.,
#                 drop_path=0.,
#                 act_layer=nn.GELU,
#                 norm_layer=nn.LayerNorm):
#        super().__init__()
#        self.norm1 = norm_layer(dim)
#        self.attn = Attention(dim,
#                              num_heads=num_heads,
#                              qkv_bias=qkv_bias,
#                              qk_scale=qk_scale,
#                              attn_drop=attn_drop,
#                              proj_drop=drop)
#        self.drop_path = DropPath(
#            drop_path) if drop_path > 0. else nn.Identity()
#        self.norm2 = norm_layer(dim)
#        mlp_hidden_dim = int(dim * mlp_ratio)
#        self.mlp = MLP(in_features=dim,
#                       hidden_features=mlp_hidden_dim,
#                       act_layer=act_layer,
#                       drop=drop)
#
#    def forward(self, x, return_attention=False):
#        y, attn = self.attn(self.norm1(x))
#        if return_attention:
#            return attn
#        x = x + self.drop_path(y)
#        residual = x
#
#        x = self.drop_path(self.mlp(self.norm2(x)))
#
#        x = residual + x
#        return x


class ViTBackbone(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """

    def __init__(self, global_pool=False, **kwargs):
        super(ViTBackbone, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm
        self.num_out = self.embed_dim

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(
            B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def forward(self, x):
        x = self.forward_features(x)
        return x


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


if __name__ == "__main__":
    import logging
    from torch.utils.data import DataLoader

    from torchsrh.datasets.db_improc import get_transformations
    from torchsrh.datasets.patch_dataset import PatchDataset
    logging.basicConfig(
        level=logging.DEBUG,
        format=
        "[%(levelname)-s|%(asctime)s|%(filename)s:%(lineno)d|%(funcName)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler()])
    logging.info("Vit backbone + patch data debug log")

    csv_path = "/nfs/turbo/umms-tocho/code/chengjia/torchsrh/torchsrh/train/data/srh7v1/srh7v1_toy.csv"
    data_root = "/nfs/turbo/umms-tocho/root_srh_db/"

    tx, vx = get_transformations({
        "data": {
            "train_augmentation": [{
                "which": "resize",
                "params": {
                    "size": 224
                }
            }],
            "valid_augmentation": "same",
            "augmentation_random_prob": 0
        }
    })
    dset = PatchDataset(data_root=data_root,
                        slides_file=csv_path,
                        segmentation_model="03207B00",
                        transform=tx,
                        balance_patch_per_class=True)
    dl = DataLoader(dset, batch_size=5)

    batch1 = next(iter(dl))
    model = get_vit_backbone()()
    assert model(torch.zeros_like(batch1["image"])).shape == torch.Size(
        [5, 768])

    import pdb
    pdb.set_trace()
