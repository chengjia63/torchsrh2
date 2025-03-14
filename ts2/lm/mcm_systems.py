import logging
import itertools
import copy
from functools import partial
from typing import Dict, Any, Optional
from abc import ABC
import gc
import psutil
import torch
from torch import nn
import torch.nn.functional as F
import einops

import pytorch_lightning as pl
import torchmetrics

import random
import math
import numpy as np
from torchsrh.losses.supcon import SupConLoss
from torchsrh.losses.vicreg import GeneralVICRegLoss
from ts2.models.ssl import (instantiate_backbone, MLP)
from ts2.optim.utils import get_optimizer_scheduler
from ts2.lm.ssl_systems import EvalBaseSystem
from ts2.models.pjepa import trunc_normal_
from ts2.models.ibot_vit import vit_small
from torchvision import transforms as transforms


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.LayerNorm):
        torch.nn.init.constant_(m.bias, 0)
        torch.nn.init.constant_(m.weight, 1.0)


class DataAugmentationiBOT(object):

    def __init__(self,
                 global_crops_scale=(0.14, 1.),
                 local_crops_scale=(0.05, 0.4),
                 global_crops_number=2,
                 local_crops_number=0):
        #def __init__(self, global_crops_scale=(0.2, 1.), local_crops_scale=(0.1, 0.4), global_crops_number=2, local_crops_number=0):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
            ],
                                   p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ])
        #normalize = transforms.Compose([
        #    transforms.ToTensor(),
        #    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        #])

        self.global_crops_number = global_crops_number
        # transformation for the first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(
                48,
                scale=global_crops_scale,
                interpolation=transforms.InterpolationMode.BICUBIC),
            flip_and_color_jitter,
            #utils.GaussianBlur(1.0),
            #normalize,
        ])
        # transformation for the rest of global crops
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(
                48,
                scale=global_crops_scale,
                interpolation=transforms.InterpolationMode.BICUBIC),
            flip_and_color_jitter,
            #utils.GaussianBlur(0.1),
            #utils.Solarization(0.2),
            #normalize,
        ])
        # transformation for the local crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(
                18,
                scale=local_crops_scale,
                interpolation=transforms.InterpolationMode.BICUBIC),
            flip_and_color_jitter,
            #utils.GaussianBlur(p=0.5),
            #normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(torch.nan_to_num(self.global_transfo1(image), nan=0))
        for _ in range(self.global_crops_number - 1):
            crops.append(torch.nan_to_num(self.global_transfo2(image), nan=0))
        for _ in range(self.local_crops_number):
            crops.append(torch.nan_to_num(self.local_transfo(image), nan=0))
        return crops


class IBotMaskGenerator():

    def __init__(self,
                 patch_size,
                 pred_ratio=0.3,
                 pred_ratio_var=0,
                 pred_aspect_ratio=(0.3, 1 / 0.3),
                 pred_shape='block',
                 pred_start_epoch=0):

        self.psz = patch_size
        self.pred_ratio = pred_ratio[0] if isinstance(pred_ratio, list) and \
            len(pred_ratio) == 1 else pred_ratio
        self.pred_ratio_var = pred_ratio_var[0] if isinstance(pred_ratio_var, list) and \
            len(pred_ratio_var) == 1 else pred_ratio_var
        if isinstance(self.pred_ratio,
                      list) and not isinstance(self.pred_ratio_var, list):
            self.pred_ratio_var = [self.pred_ratio_var] * len(self.pred_ratio)
        self.log_aspect_ratio = tuple(
            map(lambda x: math.log(x), pred_aspect_ratio))
        self.pred_shape = pred_shape
        self.pred_start_epoch = pred_start_epoch

    def get_pred_ratio(self):
        if hasattr(self, 'epoch') and self.epoch < self.pred_start_epoch:
            return 0

        if isinstance(self.pred_ratio, list):
            pred_ratio = []
            for prm, prv in zip(self.pred_ratio, self.pred_ratio_var):
                assert prm >= prv
                pr = random.uniform(prm - prv, prm + prv) if prv > 0 else prm
                pred_ratio.append(pr)
            pred_ratio = random.choice(pred_ratio)
        else:
            assert self.pred_ratio >= self.pred_ratio_var
            pred_ratio = random.uniform(self.pred_ratio - self.pred_ratio_var, self.pred_ratio + \
                self.pred_ratio_var) if self.pred_ratio_var > 0 else self.pred_ratio

        return pred_ratio

    def set_epoch(self, epoch):
        self.epoch = epoch

    def get_mask(self, images):
        masks = []
        H, W = images.shape[-2] // self.psz, images.shape[-1] // self.psz
        for img in images:

            high = self.get_pred_ratio() * H * W

            if self.pred_shape == 'block':
                # following BEiT (https://arxiv.org/abs/2106.08254), see at
                # https://github.com/microsoft/unilm/blob/b94ec76c36f02fb2b0bf0dcb0b8554a2185173cd/beit/masking_generator.py#L55
                mask = np.zeros((H, W), dtype=bool)
                mask_count = 0
                while mask_count < high:
                    max_mask_patches = high - mask_count

                    delta = 0
                    for attempt in range(10):
                        low = (min(H, W) // 3)**2
                        target_area = random.uniform(low, max_mask_patches)
                        aspect_ratio = math.exp(
                            random.uniform(*self.log_aspect_ratio))
                        h = int(round(math.sqrt(target_area * aspect_ratio)))
                        w = int(round(math.sqrt(target_area / aspect_ratio)))
                        if w < W and h < H:
                            top = random.randint(0, H - h)
                            left = random.randint(0, W - w)

                            num_masked = mask[top:top + h, left:left + w].sum()
                            if 0 < h * w - num_masked <= max_mask_patches:
                                for i in range(top, top + h):
                                    for j in range(left, left + w):
                                        if mask[i, j] == 0:
                                            mask[i, j] = 1
                                            delta += 1

                        if delta > 0:
                            break

                    if delta == 0:
                        break
                    else:
                        mask_count += delta

            elif self.pred_shape == 'rand':
                mask = np.hstack([
                    np.zeros(H * W - int(high)),
                    np.ones(int(high)),
                ]).astype(bool)
                np.random.shuffle(mask)
                mask = mask.reshape(H, W)

            else:
                # no implementation
                assert False

            masks.append(mask)

        return torch.tensor(np.array(masks))


class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """

    def __init__(self, backbone, head=None):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        if head is None:
            self.head = nn.Identity()
        else:
            self.head = head

    def forward(self, x, mask=None, return_backbone_feat=False, **kwargs):
        # convert to list
        if not isinstance(x, list):
            x = [x]
            mask = [mask] if mask is not None else None
        idx_crops = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in x]),
                return_counts=True,
            )[1], 0)
        start_idx = 0
        for end_idx in idx_crops:
            inp_x = torch.cat(x[start_idx:end_idx])

            if mask is not None:
                inp_m = torch.cat(mask[start_idx:end_idx])
                kwargs.update(dict(mask=inp_m))

            _out = self.backbone(inp_x, **kwargs)
            if start_idx == 0:
                output = _out
            else:
                output = torch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        output_ = self.head(output)
        if return_backbone_feat:
            return output, output_
        return output_


# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2)

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


class CustomSequential(nn.Sequential):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                nn.SyncBatchNorm)

    def forward(self, input):
        for module in self:
            dim = len(input.shape)
            if isinstance(module, self.bn_types) and dim > 2:
                perm = list(range(dim - 1))
                perm.insert(1, dim - 1)
                inv_perm = list(range(dim)) + [1]
                inv_perm.pop(1)
                input = module(input.permute(*perm)).permute(*inv_perm)
            else:
                input = module(input)
        return input


class DINOHead(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim,
                 norm=None,
                 act='gelu',
                 last_norm=None,
                 nlayers=3,
                 hidden_dim=2048,
                 bottleneck_dim=256,
                 norm_last_layer=True,
                 **kwargs):
        super().__init__()
        norm = self._build_norm(norm, hidden_dim)
        last_norm = self._build_norm(last_norm,
                                     out_dim,
                                     affine=False,
                                     **kwargs)
        act = self._build_act(act)

        nlayers = max(nlayers, 1)
        if nlayers == 1:
            if bottleneck_dim > 0:
                self.mlp = nn.Linear(in_dim, bottleneck_dim)
            else:
                self.mlp = nn.Linear(in_dim, out_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if norm is not None:
                layers.append(norm)
            layers.append(act)
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if norm is not None:
                    layers.append(norm)
                layers.append(act)
            if bottleneck_dim > 0:
                layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            else:
                layers.append(nn.Linear(hidden_dim, out_dim))
            self.mlp = CustomSequential(*layers)
        self.apply(self._init_weights)

        if bottleneck_dim > 0:
            self.last_layer = nn.utils.weight_norm(
                nn.Linear(bottleneck_dim, out_dim, bias=False))
            self.last_layer.weight_g.data.fill_(1)
            if norm_last_layer:
                self.last_layer.weight_g.requires_grad = False
        else:
            self.last_layer = None

        self.last_norm = last_norm

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        if self.last_layer is not None:
            x = nn.functional.normalize(x, dim=-1, p=2)
            x = self.last_layer(x)
        if self.last_norm is not None:
            x = self.last_norm(x)
        return x

    def _build_norm(self, norm, hidden_dim, **kwargs):
        if norm == 'bn':
            norm = nn.BatchNorm1d(hidden_dim, **kwargs)
        elif norm == 'syncbn':
            norm = nn.SyncBatchNorm(hidden_dim, **kwargs)
        elif norm == 'csyncbn':
            norm = CSyncBatchNorm(hidden_dim, **kwargs)
        elif norm == 'psyncbn':
            norm = PSyncBatchNorm(hidden_dim, **kwargs)
        elif norm == 'ln':
            norm = nn.LayerNorm(hidden_dim, **kwargs)
        else:
            assert norm is None, "unknown norm type {}".format(norm)
        return norm

    def _build_act(self, act):
        if act == 'relu':
            act = nn.ReLU()
        elif act == 'gelu':
            act = nn.GELU()
        else:
            assert False, "unknown act type {}".format(act)
        return act


class iBOTHead(DINOHead):

    def __init__(self,
                 *args,
                 patch_out_dim=8192,
                 norm=None,
                 act='gelu',
                 last_norm=None,
                 nlayers=3,
                 hidden_dim=2048,
                 bottleneck_dim=256,
                 norm_last_layer=True,
                 shared_head=False,
                 **kwargs):

        super(iBOTHead, self).__init__(*args,
                                       norm=norm,
                                       act=act,
                                       last_norm=last_norm,
                                       nlayers=nlayers,
                                       hidden_dim=hidden_dim,
                                       bottleneck_dim=bottleneck_dim,
                                       norm_last_layer=norm_last_layer,
                                       **kwargs)

        if not shared_head:
            if bottleneck_dim > 0:
                self.last_layer2 = nn.utils.weight_norm(
                    nn.Linear(bottleneck_dim, patch_out_dim, bias=False))
                self.last_layer2.weight_g.data.fill_(1)
                if norm_last_layer:
                    self.last_layer2.weight_g.requires_grad = False
            else:
                self.mlp2 = nn.Linear(hidden_dim, patch_out_dim)
                self.last_layer2 = None

            self.last_norm2 = self._build_norm(last_norm,
                                               patch_out_dim,
                                               affine=False,
                                               **kwargs)
        else:
            if bottleneck_dim > 0:
                self.last_layer2 = self.last_layer
            else:
                self.mlp2 = self.mlp[-1]
                self.last_layer2 = None

            self.last_norm2 = self.last_norm

    def forward(self, x):
        if len(x.shape) == 2:
            return super(iBOTHead, self).forward(x)

        if self.last_layer is not None:
            x = self.mlp(x)
            x = nn.functional.normalize(x, dim=-1, p=2)
            x1 = self.last_layer(x[:, 0])
            x2 = self.last_layer2(x[:, 1:])
        else:
            x = self.mlp[:-1](x)
            x1 = self.mlp[-1](x[:, 0])
            x2 = self.mlp2(x[:, 1:])

        if self.last_norm is not None:
            x1 = self.last_norm(x1)
            x2 = self.last_norm2(x2)

        return x1, x2


class iBOTLoss(nn.Module):

    def __init__(self,
                 out_dim,
                 patch_out_dim,
                 ngcrops,
                 nlcrops,
                 warmup_teacher_temp,
                 teacher_temp,
                 warmup_teacher_temp2,
                 teacher_temp2,
                 warmup_teacher_temp_epochs,
                 nepochs,
                 student_temp=0.1,
                 center_momentum=0.9,
                 center_momentum2=0.9,
                 lambda1=1.0,
                 lambda2=1.0,
                 mim_start_epoch=0):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.center_momentum2 = center_momentum2
        self.ngcrops = ngcrops
        self.nlcrops = nlcrops
        self.ncrops = ngcrops + nlcrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("center2", torch.zeros(1, 1, patch_out_dim))
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate(
            (np.linspace(warmup_teacher_temp, teacher_temp,
                         warmup_teacher_temp_epochs),
             np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp))
        self.teacher_temp2_schedule = np.concatenate(
            (np.linspace(warmup_teacher_temp2, teacher_temp2,
                         warmup_teacher_temp_epochs),
             np.ones(nepochs - warmup_teacher_temp_epochs) *
             teacher_temp2)) if mim_start_epoch == 0 else np.concatenate(
                 (np.ones(mim_start_epoch) * warmup_teacher_temp2,
                  np.linspace(warmup_teacher_temp2, teacher_temp2,
                              warmup_teacher_temp_epochs),
                  np.ones(nepochs - warmup_teacher_temp_epochs -
                          mim_start_epoch) * teacher_temp2))

    def forward(self, student_output, teacher_output, student_local_cls,
                student_mask, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_cls, student_patch = student_output
        teacher_cls, teacher_patch = teacher_output

        if student_local_cls is not None:
            student_cls = torch.cat([student_cls, student_local_cls])

        # [CLS] and patch for global patches
        student_cls = student_cls / self.student_temp
        student_cls_c = student_cls.chunk(self.ncrops)
        student_patch = student_patch / self.student_temp
        student_patch_c = student_patch.chunk(self.ngcrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        temp2 = self.teacher_temp2_schedule[epoch]
        teacher_cls_c = F.softmax((teacher_cls - self.center) / temp, dim=-1)
        teacher_cls_c = teacher_cls_c.detach().chunk(self.ngcrops)
        teacher_patch_c = F.softmax((teacher_patch - self.center2) / temp2,
                                    dim=-1)
        teacher_patch_c = teacher_patch_c.detach().chunk(self.ngcrops)

        total_loss1, n_loss_terms1 = 0, 0
        total_loss2, n_loss_terms2 = 0, 0
        for q in range(len(teacher_cls_c)):
            for v in range(len(student_cls_c)):
                if v == q:
                    loss2 = torch.sum(
                        -teacher_patch_c[q] *
                        F.log_softmax(student_patch_c[v], dim=-1),
                        dim=-1)
                    mask = student_mask[v].flatten(-2, -1)
                    loss2 = torch.sum(loss2 * mask.float(),
                                      dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
                    total_loss2 += loss2.mean()
                    n_loss_terms2 += 1
                else:
                    loss1 = torch.sum(-teacher_cls_c[q] *
                                      F.log_softmax(student_cls_c[v], dim=-1),
                                      dim=-1)
                    total_loss1 += loss1.mean()
                    n_loss_terms1 += 1

        total_loss1 = total_loss1 / n_loss_terms1 * self.lambda1
        total_loss2 = total_loss2 / n_loss_terms2 * self.lambda2
        total_loss = dict(cls=total_loss1,
                          patch=total_loss2,
                          loss=total_loss1 + total_loss2)
        self.update_center(teacher_cls, teacher_patch)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_cls, teacher_patch):
        """
        Update center used for teacher output.
        """
        cls_center = torch.sum(teacher_cls, dim=0, keepdim=True)
        cls_center = cls_center / (len(teacher_cls)
                                   )  # * dist.get_world_size())
        self.center = self.center * self.center_momentum + cls_center * (
            1 - self.center_momentum)

        patch_center = torch.sum(teacher_patch.mean(1), dim=0, keepdim=True)
        patch_center = patch_center / (len(teacher_patch)
                                       )  # * dist.get_world_size())
        self.center2 = self.center2 * self.center_momentum2 + patch_center * (
            1 - self.center_momentum2)


def cosine_scheduler(base_value,
                     final_value,
                     epochs,
                     niter_per_ep,
                     warmup_epochs=0,
                     start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value,
                                      warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


class MCMSystem(EvalBaseSystem):

    def __init__(self,
                 model_hyperparams,
                 loss_params: Optional[Dict] = None,
                 ema_beta: Optional[Dict] = None,
                 opt_cf: Optional[Dict] = None,
                 schd_cf: Optional[Dict] = None,
                 training_params: Optional[Dict] = None):
        super().__init__()

        self.beta = ema_beta
        self.opt_cf_ = opt_cf
        self.schd_cf_ = schd_cf

        # tile level encoder
        self.student_tile_encoder = vit_small(img_size=[48],
                                              patch_size=4,
                                              masked_im_modeling=True,
                                              return_all_tokens=True)
        self.teacher_tile_encoder = vit_small(img_size=[48],
                                              patch_size=4,
                                              return_all_tokens=True)

        self.student_tile_encoder = MultiCropWrapper(
            self.student_tile_encoder,
            iBOTHead(
                384,  #embed_dim,
                1024,  #args.out_dim,
                patch_out_dim=1024,  #args.patch_out_dim,
                norm=None,  #args.norm_in_head,
                act='gelu',  #args.act_in_head,
                norm_last_layer=True,  #args.norm_last_layer,
                shared_head=False,  #args.shared_head,
            ))

        self.teacher_tile_encoder = MultiCropWrapper(
            self.teacher_tile_encoder,
            iBOTHead(
                384,  #embed_dim, 
                1024,  #args.out_dim,
                patch_out_dim=1024,  #args.patch_out_dim,
                norm=None,  #args.norm_in_head,
                act="gelu",  #args.act_in_head,
                shared_head=True,  #args.shared_head_teacher,
            ),
        )

        # teacher and student start with the same weights
        self.teacher_tile_encoder.load_state_dict(
            self.student_tile_encoder.state_dict(), strict=False)
        # there is no backpropagation through the teacher, so no need for gradients
        for p in self.teacher_tile_encoder.parameters():
            p.requires_grad = False


#        self.predictor = None #vit_predictor( # TODO
#            num_patches=self.encoder.patch_embed.num_patches,
#            embed_dim=self.encoder.embed_dim,
#            num_heads=self.encoder.num_heads,
#            **pred_params)

        self.training_params_ = training_params
        print(training_params)
        if training_params:
            self.criterion = iBOTLoss(
                1024,  #args.out_dim,
                1024,  #args.out_dim if same_dim else args.patch_out_dim,
                2,  #args.global_crops_number,
                8,  #args.local_crops_number,
                0.04,  #args.warmup_teacher_temp,
                0.04,  #args.teacher_temp,
                0.07,  #args.warmup_teacher_patch_temp,
                0.07,  #args.teacher_patch_temp,
                int(training_params["num_ep_total"] *
                    0.3),  #args.warmup_teacher_temp_epochs,
                training_params["num_ep_total"],  # TODO #args.epochs,
                lambda1=1.0,  #args.lambda1,
                lambda2=1.0,  #args.lambda2,
                mim_start_epoch=0,  #args.pred_start_epoch,
            )
            self.train_loss = torchmetrics.MeanMetric()
            self.val_loss = torchmetrics.MeanMetric()
            #self.test_output = []

            self.tile_xform = DataAugmentationiBOT(global_crops_number=2,
                                                   local_crops_number=8)
            self.tile_mask_generator = IBotMaskGenerator(patch_size=4)
            #self.patch_mask_generator = IBotMaskGenerator(patch_size=1)

            self.momentum_scheduler = cosine_scheduler(
                0.9995, 1, training_params["num_ep_total"],
                self.training_params_["num_it_per_ep"])
        else:
            self.criterion = None

    def training_step(self, batch, _):

        with torch.no_grad():
            self.tile_mask_generator.set_epoch(self.current_epoch)
            #self.patch_mask_generator.set_epoch(self.current_epoch)

            # batch patches into tile regions
            batched_regions = einops.rearrange(
                batch["image"],
                "b a c (nh rh) (nw rw) -> (b a nh nw) c rh rw",
                rh=48,
                rw=48)

            # transform image, get crops
            region_crops = self.tile_xform(batched_regions)  # global, local

            # get mask for global crop
            tile_masks = [
                self.tile_mask_generator.get_mask(i).to(batch["image"].device)
                for i in region_crops[:2]
            ]

        # 48x48 tile encoding
        student_global_emb = self.student_tile_encoder(region_crops[:2],
                                                       mask=tile_masks)
        teacher_emb = self.teacher_tile_encoder(region_crops[:2])

        self.student_tile_encoder.backbone.masked_im_modeling = False
        student_local_emb = self.student_tile_encoder(region_crops[2:])
        self.student_tile_encoder.backbone.masked_im_modeling = True

        # reshape the student and teacher output for second stage modeling
        #studnet_emb_shaped0 = einops.rearrange(student_emb0, "(b a nh nw) d -> b a d nh nw", b=batch["image"].shape[0], nh=6, nw=6)
        #student_emb_shaped1 = einops.rearrange(studnet_emb1, "(b a nh nw) d -> b a d nh nw", b=batch["image"].shape[0], nh=6, nw=6)
        #
        #with torch.no_grad():
        #    teacher_emb0 = self.target_encoder(region0)
        #    teacher_emb1 = self.target_encoder(region1)

        #    teacher_emb_shaped0 = einops.rearrange(teacher_emb0, "(b a nh nw) d -> b a d nh nw", b=batch["image"].shape[0], nh=6, nw=6)
        #    teacher_emb_shaped1 = einops.rearrange(teacher_emb1, "(b a nh nw) d -> b a d nh nw", b=batch["image"].shape[0], nh=6, nw=6)

        all_loss = self.criterion(student_global_emb, teacher_emb,
                                  student_local_emb[0], tile_masks,
                                  self.current_epoch)

        for l in all_loss:
            self.log(f"train/{l}",
                     all_loss[l].detach().item(),
                     on_step=True,
                     on_epoch=True,
                     batch_size=batched_regions.shape[0],
                     rank_zero_only=True)

        loss = all_loss.pop('loss')
        self.train_loss.update(loss.detach().item(),
                               weight=batched_regions.shape[0])
        return loss

    @torch.inference_mode()
    def validation_step(self, batch, _):
        return

        batched_regions = einops.rearrange(
            batch["image"],
            "b a c (nh rh) (nw rw) -> (b a nh nw) c rh rw",
            rh=48,
            rw=48)
        batched_emb = self.encoder(batched_regions)
        emb = einops.rearrange(batched_emb,
                               "(b a nh nw) d -> b a d nh nw",
                               b=batch["image"].shape[0],
                               nh=6,
                               nw=6)

        import pdb
        pdb.set_trace()
        imgs, masks_enc, masks_pred = batch
        imgs = imgs['image'].squeeze()

        def forward_target():
            with torch.no_grad():
                h = self.model.target_encoder(imgs)
                h = F.layer_norm(h, (h.size(-1), ))
                B = len(h)
                # -- create targets (masked regions of h)
                h = apply_masks(h, masks_pred)
                h = repeat_interleave_batch(h, B, repeat=len(masks_enc))
                return h

        def forward_context():
            z = self.model.encoder(imgs, masks_enc)
            z = self.model.predictor(z, masks_enc, masks_pred)
            return z

        def loss_fn(z, h):
            loss = self.criterion(z, h)
            return loss

        h = forward_target()
        z = forward_context()
        loss = loss_fn(z, h)
        self.val_loss.update(loss, weight=imgs.shape[0])
        return loss

    @torch.inference_mode()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        assert batch["image"].shape[1] == 1

        emb = self.teacher_tile_encoder.backbone(batch["image"][:, 0,
                                                                ...])[:, 0, :]
        results = {
            "path": batch["path"],
            "label": batch["label"],
            "embeddings": emb
        }

        return results

    def on_train_epoch_end(self):
        train_loss = self.train_loss.compute()
        self.log("train/contrastive_manualepoch",
                 train_loss,
                 on_epoch=True,
                 sync_dist=True,
                 rank_zero_only=True)
        logging.info(f"train/contrastive_manualepoch {train_loss}")
        self.train_loss.reset()
        gc.collect()

    def on_validation_epoch_end(self):
        return
        val_loss = self.val_loss.compute()
        self.log("val/contrastive_manualepoch",
                 val_loss,
                 on_epoch=True,
                 sync_dist=True,
                 rank_zero_only=True)
        logging.info(f"val/contrastive_manualepoch {val_loss}")
        self.val_loss.reset()
        gc.collect()

    def on_before_zero_grad(self, _):
        names_q, params_q, names_k, params_k = [], [], [], []
        for name_q, param_q in self.student_tile_encoder.named_parameters():
            names_q.append(name_q)
            params_q.append(param_q)
        for name_k, param_k in self.teacher_tile_encoder.named_parameters():
            names_k.append(name_k)
            params_k.append(param_k)
        names_common = list(set(names_q) & set(names_k))
        params_q = [
            param_q for name_q, param_q in zip(names_q, params_q)
            if name_q in names_common
        ]
        params_k = [
            param_k for name_k, param_k in zip(names_k, params_k)
            if name_k in names_common
        ]

        with torch.no_grad():
            m = self.momentum_scheduler[self.trainer.global_step]
            for param_q, param_k in zip(params_q, params_k):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

    def configure_optimizers(self):
        if not self.training_params_: return None  # Not training

        opt, sch = get_optimizer_scheduler(self,
                                           opt_cf=self.opt_cf_,
                                           schd_cf=self.schd_cf_,
                                           **self.training_params_)

        if sch:
            return [opt], {
                "scheduler": sch,
                "interval": "step",
                "frequency": 1,
                "name": "lr"
            }
        else:
            return [opt]
