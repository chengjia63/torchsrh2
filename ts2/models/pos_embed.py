# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------

import numpy as np
import math
from typing import List, Tuple, Optional, Union

import torch
from torch import nn
import torch.nn.functional as F
import einops


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, prefix_len=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if prefix_len:
        pos_embed = np.concatenate(
            [np.zeros([prefix_len, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2,
                                              grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2,
                                              grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def resample_abs_pos_embed(posemb,
                           new_size: List[int],
                           old_size: Optional[List[int]] = None,
                           num_prefix_tokens: int = 1,
                           interpolation: str = 'bicubic',
                           antialias: bool = True,
                           verbose: bool = False):
    # sort out sizes, assume square if old size not provided
    num_pos_tokens = posemb.shape[1]
    num_new_tokens = new_size[0] * new_size[1] + num_prefix_tokens
    if num_new_tokens == num_pos_tokens and new_size[0] == new_size[1]:
        return posemb

    if old_size is None:
        hw = int(math.sqrt(num_pos_tokens - num_prefix_tokens))
        old_size = hw, hw

    if num_prefix_tokens:
        posemb_prefix, posemb = posemb[:, :
                                       num_prefix_tokens], posemb[:,
                                                                  num_prefix_tokens:]
    else:
        posemb_prefix, posemb = None, posemb

    # do the interpolation
    embed_dim = posemb.shape[-1]
    orig_dtype = posemb.dtype
    posemb = posemb.float()  # interpolate needs float32
    posemb = posemb.reshape(1, old_size[0], old_size[1],
                            -1).permute(0, 3, 1, 2)
    posemb = F.interpolate(posemb,
                           size=new_size,
                           mode=interpolation,
                           antialias=antialias)
    posemb = posemb.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)
    posemb = posemb.to(orig_dtype)

    # add back extra (class, etc) prefix tokens
    if posemb_prefix is not None:
        posemb = torch.cat([posemb_prefix, posemb], dim=1)

    return posemb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
#def interpolate_pos_embed(model, checkpoint_model):
#    if 'pos_embed' in checkpoint_model:
#        pos_embed_checkpoint = checkpoint_model['pos_embed']
#        embedding_size = pos_embed_checkpoint.shape[-1]
#        num_patches = model.patch_embed.num_patches
#        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
#        # height (== width) for the checkpoint position embedding
#        orig_size = int(
#            (pos_embed_checkpoint.shape[-2] - num_extra_tokens)**0.5)
#        # height (== width) for the new position embedding
#        new_size = int(num_patches**0.5)
#        # class_token and dist_token are kept unchanged
#        if orig_size != new_size:
#            print("Position interpolate from %dx%d to %dx%d" %
#                  (orig_size, orig_size, new_size, new_size))
#            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
#            # only the position tokens are interpolated
#            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
#            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size,
#                                            embedding_size).permute(
#                                                0, 3, 1, 2)
#            pos_tokens = torch.nn.functional.interpolate(pos_tokens,
#                                                         size=(new_size,
#                                                               new_size),
#                                                         mode='bicubic',
#                                                         align_corners=False)
#            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
#            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
#            checkpoint_model['pos_embed'] = new_pos_embed


class AbsolutePositionEmbedding(nn.Module):
    '''
    Attributes
        seq_len: sequence length which should be a square number
        embed_dim: dimension for positional embedding should be same as input feature dimension
        prefix_token: number of token which is not part of the input sequence, like cls_token or dist_token
        requires_grad: gradient for positional embeddings
    '''

    def __init__(self,
                 seq_len: int,
                 embed_dim: int,
                 prefix_len: int = 1,
                 pos_emb_grad: bool = False,
                 init_weight='sin'):
        super().__init__()
        self.seq_len = seq_len
        self.prefix_len = prefix_len
        self.pos_embed_grad = pos_emb_grad
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len + self.prefix_len,
                                                  embed_dim),
                                      requires_grad=pos_emb_grad)
        self.side_length = int(self.seq_len**0.5)
        if init_weight == 'sin':
            self.initialize_weights()

    def initialize_weights(self):
        '''Currently we only initialized in 2D sine-cosine
        '''
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1],
                                            int(self.seq_len**.5),
                                            prefix_len=self.prefix_len)
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))

    def interpolate_pos_encoding(self, seq_len):
        npatch = seq_len - self.prefix_len
        assert (int(npatch**0.5)**2 == npatch
                ), f"sequence length {npatch} is not a square number"
        N = self.seq_len
        if npatch == N:
            return self.pos_embed

        class_emb = self.pos_embed[:, 0:self.prefix_len]
        pos_embed = self.pos_embed[:, self.prefix_len:]
        dim = self.pos_embed.shape[-1]
        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)),
                              dim).permute(0, 3, 1, 2),
            scale_factor=math.sqrt(npatch / N),
            mode='bicubic',
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_emb, pos_embed), dim=1)

    def lookup_pos_encoding(self, coords):
        '''
        coords: [(0,0),...,(17,17)]
        '''
        coords = list(range(self.prefix_len)) + [
            self.prefix_len + i * self.side_length + j for (i, j) in coords
        ]
        return self.pos_embed[:, torch.tensor(coords), :]

    def forward(self, x, coords=None):
        if not self.pos_embed_grad:
            return x + self.interpolate_pos_encoding(x.shape[1])
        elif coords:
            return x + self.lookup_pos_encoding(coords)
        return x


class FourierFeaturePositionalEncoding(nn.Module):
    """Learnable fourier feature positional encoding.

    References:
    - https://arxiv.org/pdf/2106.02795.pdf
    - https://github.com/willGuimont/learnable_fourier_positional_encoding/blob/62528af9/learnable_fourier_pos_encoding.py#L6
    """

    def __init__(self,
                 embed_dim: int,
                 dim_ff: int = 96,
                 dim_mlp: int = 36,
                 gamma: float = .25,
                 prefix_len: int = 0,
                 pos_emb_grad: bool = True):
        super(FourierFeaturePositionalEncoding, self).__init__()
        self.dim_ff_ = dim_ff
        self.dim_mlp_ = dim_mlp
        self.gamma_ = gamma
        self.embed_dim_ = embed_dim
        self.prefix_len = prefix_len

        self.cls_pos_emb = nn.Parameter(torch.zeros(1, self.prefix_len,
                                                    embed_dim),
                                        requires_grad=True)

        self.num_pos_ = 1  # G
        self.pos_dim_ = 2  # M

        self._ff_embed = nn.Linear(self.pos_dim_, dim_ff // 2, bias=False)
        torch.nn.init.normal_(self._ff_embed.weight, mean=0, std=gamma)
        if not pos_emb_grad:
            for param in self._ff_embed.parameters():
                param.requires_grad = False

        self._mlp = nn.Sequential(
            nn.LayerNorm(dim_ff), nn.Linear(dim_ff, dim_mlp), nn.GELU(),
            nn.LayerNorm(dim_mlp),
            nn.Linear(dim_mlp, embed_dim // self.num_pos_))

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self._mlp.apply(init_weights)

    def forward_new(self, bsz, n_tokens, coords, return_ff: bool = False):
        n = n_tokens - self.prefix_len

        x = coords.unsqueeze(0).float().unsqueeze(-2)  # NxGxM (G=1, M=2)

        ff_vec = self._ff_embed(x)  # NxGx(F/2)

        f = torch.cat([torch.cos(ff_vec), torch.sin(ff_vec)], axis=-1)
        f = 1 / np.sqrt(self.dim_ff_) * f  # NxGxF

        if return_ff: return f
        pe = self._mlp(f).reshape(n, self.embed_dim_).unsqueeze(0).repeat(
            bsz, 1, 1)
        return einops.rearrange(pe, "inter intra d -> intra inter d")

    def forward(self, H, coords, return_ff: bool = False):
        bsz, n = H.shape[0], H.shape[1]
        n = n - self.prefix_len

        x = coords.unsqueeze(0).float().unsqueeze(-2)  # NxGxM (G=1, M=2)
        x = x.to(H.device)

        ff_vec = self._ff_embed(x)  # NxGx(F/2)

        f = torch.cat([torch.cos(ff_vec), torch.sin(ff_vec)], axis=-1)
        f = 1 / np.sqrt(self.dim_ff_) * f  # NxGxF

        if return_ff: return f

        pe = self._mlp(f).reshape(n, self.embed_dim_).unsqueeze(0).repeat(
            bsz, 1, 1)
        pe = torch.cat((self.cls_pos_emb, pe), dim=1)

        return H + pe
        # if self.global_avg_pool_:
        #     return H + pe
        # else:
        # cls_token, feat_token = H[:, 0:self.prefix_len], H[:, self.prefix_len:]
        # feat_token = feat_token + pe
        # return torch.cat((cls_token, feat_token), dim=1)


FFPEG = FourierFeaturePositionalEncoding

if __name__ == '__main__':
    coords = [(0, 0), (0, 1.2), (0, 2), (0, 3)]
    data = torch.rand(1, 4, 768)
    pos_emb = FFPEG(768, pos_emb_grad=True)
    #pos_emb = ABS_POS_EMB(196,768,pos_emb_grad=True)
    print(pos_emb(data, coords=coords).shape)
    import pdb
    pdb.set_trace()
