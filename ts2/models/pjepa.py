import copy
import math
from itertools import chain
from functools import partial
from collections import namedtuple
from typing import List, Optional, Callable, Dict

import torch
from torch import nn, Tensor
import einops

from torchvision import models

from ts2.models.ssl import instantiate_backbone
from ts2.models.pos_embed import FourierFeaturePositionalEncoding

import torch.nn.functional as F


class InterPatchJEPANetwork(torch.nn.Module):
    """A network consists of a backbone and projection head.
    Forward pass returns the normalized embeddings after a projection layer.
    """

    def __init__(self, backbone_cf: Dict, pred_params: Dict):
        super(InterPatchJEPANetwork, self).__init__()
        self.bb = instantiate_backbone(**backbone_cf)
        self.pred = InterPatchJEPAPredictor(embed_dim=self.bb.num_out,
                                            **pred_params)
        self.target_bb = copy.deepcopy(self.bb)
        #self.num_out = self.pred.num_out

    def forward(self, batch: Dict) -> torch.Tensor:
        bs, nc, nt, _, _, _ = batch["target_image"].shape

        context_im = einops.rearrange(batch["context_image"],
                                      "b nc c h w -> (b nc) c h w")

        target_im = einops.rearrange(batch["target_image"],
                                     "b nc nt c h w -> (b nc nt) c h w")

        target_delta = einops.rearrange(batch["target_delta"],
                                        "b nc nt delta -> (b nc) nt delta")

        context_emb = self.bb(context_im)
        context_emb = einops.rearrange(context_emb, "bnc d -> bnc 1 d")
        target_hat = self.pred(context_emb, target_delta)
        target_hat = einops.rearrange(target_hat, "b nc d -> (b nc) d")

        with torch.no_grad():
            target_emb = self.target_bb(target_im)
            target_emb = F.layer_norm(target_emb, (target_emb.size(-1), ))
            # normalize over feature-dim

        return target_hat, target_emb


class InterPatchJEPAPredictor(nn.Module):
    """ Vision Transformer """

    def __init__(self,
                 pos_emb_params,
                 block_params,
                 embed_dim=768,
                 predictor_embed_dim=384,
                 depth=6,
                 drop_path_rate=0.0,
                 init_std=0.02):
        super().__init__()

        self.predictor_embed = nn.Linear(embed_dim,
                                         predictor_embed_dim,
                                         bias=True)

        self.interpatch_pos_embed = FourierFeaturePositionalEncoding(
            embed_dim=predictor_embed_dim, **pos_emb_params)

        # same as timm vit
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)
               ]  # stochastic depth decay rule
        self.predictor_blocks = nn.ModuleList([
            Block(dim=predictor_embed_dim,
                  drop_path=dpr[i],
                  norm_layer=norm_layer,
                  **block_params) for i in range(depth)
        ])
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim,
                                        embed_dim,
                                        bias=True)
        # ------
        self.init_std = init_std
        #trunc_normal_(self.mask_token, std=self.init_std)
        #trunc_normal_(self.intrapatch_pos_embed, std=.02)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):

        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, deltas):
        # -- map from encoder-dim to pedictor-dim
        x = self.predictor_embed(x)

        # -- concat mask tokens to x
        target_patch_pos_emb = self.interpatch_pos_embed.forward_new(
            coords=deltas)

        _, n_ctxt, _ = x.shape
        x = torch.cat([x, target_patch_pos_emb], dim=1)

        # -- fwd prop
        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)

        # -- return preds for mask tokens
        x = x[:, n_ctxt:]
        x = self.predictor_proj(x)
        return x


# class InterPatchJEPANetwork(torch.nn.Module):
#     """A network consists of a backbone and projection head.
#     Forward pass returns the normalized embeddings after a projection layer.
#     """
#
#     def __init__(self, backbone_cf: Dict, pred_params: Dict):
#         super(InterPatchJEPANetwork, self).__init__()
#         self.bb = instantiate_backbone(**backbone_cf)
#         self.pred = InterPatchJEPAPredictor(embed_dim=self.bb.num_out,
#                                             **pred_params)
#         self.target_bb = copy.deepcopy(self.bb)
#         #self.num_out = self.pred.num_out
#
#     def forward(self, batch: Dict) -> torch.Tensor:
#         bs, nc, nt, _, _, _ = batch["target_image"].shape
#         context_im = einops.rearrange(batch["context_image"],
#                                       "b nc c h w -> (b nc) c h w")
#
#         target_im = einops.rearrange(batch["target_image"],
#                                      "b nc nt c h w -> (b nc nt) c h w")
#
#         target_delta = einops.rearrange(batch["target_delta"],
#                                         "b nc nt delta -> (b nc nt) delta")
#
#         context_emb = self.bb(context_im)
#         context_emb = einops.rearrange(context_emb,
#                                        "(b nc) p d -> b nc p d",
#                                        b=bs)
#         context_emb = einops.rearrange(context_emb.repeat(nt, 1, 1, 1, 1),
#                                        "nt b nc p d -> (b nc nt) p d")
#         target_hat = self.pred(context_emb, target_delta)
#
#         with torch.no_grad():
#             target_emb = self.target_bb(target_im)
#             target_emb = F.layer_norm(target_emb, (target_emb.size(-1), ))
#             # normalize over feature-dim
#
#         return target_hat, target_emb

# class InterPatchJEPAPredictor(nn.Module):
#     """ Vision Transformer """

#     def __init__(self,
#                  img_size: int,
#                  patch_size: int,
#                  pos_emb_params,
#                  block_params,
#                  embed_dim=768,
#                  predictor_embed_dim=384,
#                  depth=6,
#                  drop_path_rate=0.0,
#                  init_std=0.02):
#         super().__init__()

#         self.num_token_one_dim = img_size // patch_size

#         self.predictor_embed = nn.Linear(embed_dim,
#                                          predictor_embed_dim,
#                                          bias=True)

#         self.interpatch_pos_embed = FourierFeaturePositionalEncoding(
#             embed_dim=predictor_embed_dim, **pos_emb_params)

#         # same as timm vit
#         self.intrapatch_pos_embed = nn.Parameter(
#             torch.randn(1, self.num_token_one_dim * self.num_token_one_dim,
#                         predictor_embed_dim) * .02)

#         norm_layer = partial(nn.LayerNorm, eps=1e-6)
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)
#                ]  # stochastic depth decay rule
#         self.predictor_blocks = nn.ModuleList([
#             Block(dim=predictor_embed_dim,
#                   drop_path=dpr[i],
#                   norm_layer=norm_layer,
#                   **block_params) for i in range(depth)
#         ])
#         self.predictor_norm = norm_layer(predictor_embed_dim)
#         self.predictor_proj = nn.Linear(predictor_embed_dim,
#                                         embed_dim,
#                                         bias=True)
#         # ------
#         self.init_std = init_std
#         #trunc_normal_(self.mask_token, std=self.init_std)
#         trunc_normal_(self.intrapatch_pos_embed, std=.02)
#         self.apply(self._init_weights)
#         self.fix_init_weight()

#     def fix_init_weight(self):

#         def rescale(param, layer_id):
#             param.div_(math.sqrt(2.0 * layer_id))

#         for layer_id, layer in enumerate(self.predictor_blocks):
#             rescale(layer.attn.proj.weight.data, layer_id + 1)
#             rescale(layer.mlp.fc2.weight.data, layer_id + 1)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=self.init_std)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             trunc_normal_(m.weight, std=self.init_std)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, x, deltas):
#         # -- map from encoder-dim to pedictor-dim
#         x = self.predictor_embed(x)

#         # -- add positional embedding to x tokens
#         context_pos_emb = resample_abs_pos_embed(
#             self.intrapatch_pos_embed,
#             (self.num_token_one_dim, self.num_token_one_dim),
#             num_prefix_tokens=0)
#         x += context_pos_emb

#         # -- concat mask tokens to x
#         target_patch_pos_emb = self.interpatch_pos_embed.forward_new(
#             x.shape[1], deltas.shape[0], coords=deltas)
#         target_intrapatch_pos_emb = resample_abs_pos_embed(
#             self.intrapatch_pos_embed,
#             (self.num_token_one_dim, self.num_token_one_dim),
#             num_prefix_tokens=0)
#         target_patch_pos_emb += target_intrapatch_pos_emb

#         _, n_ctxt, _ = x.shape
#         x = torch.cat([x, target_patch_pos_emb], dim=1)

#         # -- fwd prop
#         for blk in self.predictor_blocks:
#             x = blk(x)
#         x = self.predictor_norm(x)

#         # -- return preds for mask tokens
#         x = x[:, n_ctxt:]
#         x = self.predictor_proj(x)
#         return x


# utils
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def resample_abs_pos_embed(
    posemb,
    new_size: List[int],
    old_size: Optional[List[int]] = None,
    num_prefix_tokens: int = 1,
    interpolation: str = 'bicubic',
    antialias: bool = True,
    verbose: bool = False,
):
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

    if not torch.jit.is_scripting() and verbose:
        _logger.info(f'Resized position embedding: {old_size} to {new_size}.')

    return posemb


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (
        x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(
        shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
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


class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              attn_drop=attn_drop,
                              proj_drop=drop)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans,
                              embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class ConvEmbed(nn.Module):
    """
    3x3 Convolution stems for ViT following ViTC models
    """

    def __init__(self,
                 channels,
                 strides,
                 img_size=224,
                 in_chans=3,
                 batch_norm=True):
        super().__init__()
        # Build the stems
        stem = []
        channels = [in_chans] + channels
        for i in range(len(channels) - 2):
            stem += [
                nn.Conv2d(channels[i],
                          channels[i + 1],
                          kernel_size=3,
                          stride=strides[i],
                          padding=1,
                          bias=(not batch_norm))
            ]
            if batch_norm:
                stem += [nn.BatchNorm2d(channels[i + 1])]
            stem += [nn.ReLU(inplace=True)]
        stem += [
            nn.Conv2d(channels[-2],
                      channels[-1],
                      kernel_size=1,
                      stride=strides[-1])
        ]
        self.stem = nn.Sequential(*stem)

        # Comptute the number of patches
        stride_prod = int(np.prod(strides))
        self.num_patches = (img_size[0] // stride_prod)**2

    def forward(self, x):
        p = self.stem(x)
        return p.flatten(2).transpose(1, 2)
