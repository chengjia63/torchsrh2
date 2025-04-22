# Copyright (c) 2025 University of Michigan
#
# This source code is licensed under the MIT License.
# See the LICENSE file in the root directory for details.

from functools import partial
import logging
import collections
import einops

import torch
from torch import nn

from dinov2.train.ssl_meta_arch import SSLMetaArch
from dinov2.fsdp import ShardedGradScaler
from dinov2.models import build_model
import dinov2.utils.utils as dinov2_utils

def sample_view(tokens, v_min=None, v_max=None):
    # tokens shape: (v, b, c, h, w)
    v, b, c, h, w = tokens.shape
    if v_min is None:
        v_min = 0
    if v_max is None:
        v_max = v  # typically, v_max should be <= v, not b

    # Permute to shape: (b, v, c, h, w) for easier gathering along the v dimension.
    tokens_perm = tokens.permute(1, 0, 2, 3, 4)  # shape: (b, v, c, h, w)

    # Generate random indices for each (b, h, w) location, shape: (b, h, w)
    idx = torch.randint(low=v_min,
                        high=v_max,
                        size=(b, h, w),
                        device=tokens.device)

    # Expand idx to shape (b, 1, h, w) so it can be used to gather along the v dimension.
    idx = idx.unsqueeze(1)  # shape: (b, 1, h, w)

    # Expand indices to cover the channel dimension.
    # The gather will be done along dim=1 (the v dimension).
    idx = idx.expand(b, 1, h, w)

    # Use gather to sample the selected view.
    # We need to expand indices to also match the channel dimension.
    # tokens_perm shape: (b, v, c, h, w)
    # We want to gather along dim=1, resulting in shape (b, 1, c, h, w).
    sampled = tokens_perm.gather(dim=1,
                                 index=idx.unsqueeze(2).expand(b, 1, c, h, w))

    # Squeeze out the singleton view dimension to get shape: (b, c, h, w)
    return sampled.squeeze(1)


def crop_views(xs, h0, w0):
    # x: (b, c, h, w)
    b, c, h, w = xs[0].shape
    device = xs[0].device

    # Random top and left coordinates for each image in the batch
    top = torch.randint(0, h - h0 + 1, (b, ), device=device)
    left = torch.randint(0, w - w0 + 1, (b, ), device=device)

    # Create a range for the crop dimensions and expand it to (b, h0, w0)
    row_offsets = torch.arange(h0, device=device).view(1, h0, 1)
    col_offsets = torch.arange(w0, device=device).view(1, 1, w0)

    # Compute absolute row and column indices for each crop
    # top and left are reshaped to (b, 1, 1) so they broadcast properly
    rows = top.view(b, 1, 1) + row_offsets  # shape: (b, h0, 1)
    cols = left.view(b, 1, 1) + col_offsets  # shape: (b, 1, w0)

    # Create a batch index for advanced indexing
    batch_idx = torch.arange(b, device=device).view(b, 1, 1)

    # Use advanced indexing to extract the crop for each image.
    # This returns a tensor of shape (b, h0, w0, c)
    cropped = [
        x[batch_idx, :, rows, cols].permute(0, 3, 1, 2).contiguous()
        for x in xs
    ]

    return cropped


# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from functools import partial
import logging
import collections

import torch
from torch import nn

from dinov2.loss import DINOLoss, iBOTPatchLoss, KoLeoLoss
from dinov2.models import build_model_from_cfg
from dinov2.layers import DINOHead
from dinov2.utils.utils import has_batchnorms
from dinov2.utils.param_groups import get_params_groups_with_decay, fuse_params_groups
from dinov2.fsdp import get_fsdp_wrapper, ShardedGradScaler, get_fsdp_modules, reshard_fsdp_model

from dinov2.models.vision_transformer import BlockChunk


try:
    from xformers.ops import fmha
except ImportError:
    raise AssertionError("xFormers is required for training")


logger = logging.getLogger("dinov2")


@torch.no_grad()
def load_uni_one_bbone(backbone, ckpt):
    backbone.cls_token.copy_(ckpt["cls_token"])
    backbone.pos_embed.copy_(ckpt["pos_embed"])
    backbone.patch_embed.load_state_dict(
        collections.OrderedDict([(k.removeprefix("patch_embed."), ckpt[k])
                                 for k in ckpt.keys()
                                 if k.startswith("patch_embed.")]))
    backbone.norm.load_state_dict(
        collections.OrderedDict([(k.removeprefix("norm."), ckpt[k])
                                 for k in ckpt.keys()
                                 if k.startswith("norm.")]))
    for b in backbone.blocks:
        expected_keys = b.state_dict().keys()
        b.load_state_dict(
            collections.OrderedDict([(k, ckpt[f"blocks.{k}"])
                                     for k in expected_keys]))

@torch.no_grad()
def load_dinov2_blocks_cls_only(backbone, ckpt):
    backbone.cls_token.copy_(ckpt["cls_token"])
    backbone.mask_token.copy_(ckpt["mask_token"])
    #backbone.pos_embed.copy_(ckpt["embeddings.position_embeddings"])
    #backbone.patch_embed.load_state_dict(
    #    collections.OrderedDict([(k.removeprefix("embeddings.patch_embeddings.").replace("projection", "proj"), ckpt[k])
    #                             for k in ckpt.keys()
    #                             if k.startswith("embeddings.patch_embeddings.")]))

    backbone.norm.load_state_dict(
        collections.OrderedDict([
        ("weight", ckpt["norm.weight"]),
        ("bias", ckpt["norm.bias"])]))

    for b in backbone.blocks:
        expected_keys = b.state_dict().keys()
        b.load_state_dict(
            collections.OrderedDict([(k, ckpt[f"blocks.{k}"])
                                     for k in expected_keys]))



class FullyShardedForwardArch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        student_model_dict = dict()

        student_backbone, teacher_backbone, embed_dim = build_model_from_cfg(cfg)
        student_model_dict["backbone"] = student_backbone
        logger.info(f"OPTIONS -- architecture : embed_dim: {embed_dim}")

        if cfg.student.pretrained_weights:
            if type(cfg.student.pretrained_weights) is str:
                chkpt = torch.load(cfg.student.pretrained_weights, map_location="cpu")
                logger.info(f"OPTIONS -- pretrained weights: loading from {cfg.student.pretrained_weights}")
                
                if "model" in chkpt.keys():
                    student_backbone.load_state_dict(chkpt["model"], strict=False)
                elif "teacher" in chkpt.keys():

                    dinohead_ckpt = {
                        k.removeprefix("dino_head."):chkpt["teacher"][k]
                        for k in chkpt["teacher"] if k.startswith("dino_head")}
                    ibothead_ckpt = {
                        k.removeprefix("ibot_head."):chkpt["teacher"][k]
                        for k in chkpt["teacher"] if k.startswith("ibot_head")}
                    
                    bbonehead_ckpt = {
                        k.removeprefix("backbone."):chkpt["teacher"][k]
                        for k in chkpt["teacher"]
                        if ((not k.startswith("ibot_head")) and 
                            (not k.startswith("dino_head")))}
                    student_backbone.load_state_dict(bbonehead_ckpt, strict=True)
                else:
                    raise ValueError()

            elif cfg.student.pretrained_weights.how == "dinov2_blocks_cls_only":
                ckpt = torch.load(cfg.student.pretrained_weights.path,
                                      map_location="cpu")
                load_dinov2_blocks_cls_only(student_model_dict["backbone"], ckpt)
                logger.info(f"OPTIONS -- pretrained weights: loading from {cfg.student.pretrained_weights.path}")

            elif cfg.student.pretrained_weights.how == "uni":
                uni_ckpt = torch.load(cfg.student.pretrained_weights.path,
                                      map_location="cpu")
                load_uni_one_bbone(student_model_dict["backbone"], uni_ckpt)
                logger.info(f"OPTIONS -- pretrained weights: loading from {cfg.student.pretrained_weights.path}")

            else:
                raise ValueError()

        self.student = nn.ModuleDict(student_model_dict)

        logger.info(f"Student is built: they are both {cfg.student.arch} network.")


    def forward_(self, images, return_student_emb=False):
        global_crops = images["collated_global_crops"].cuda(non_blocking=True)
        student_global_backbone_output_dict = self.student.backbone(
            [global_crops], masks=[None], is_training=True
        )[0]

        if return_student_emb:
            return student_global_backbone_output_dict["x_prenorm"]
        else:
            return None


    def train(self):
        super().train()

    def get_maybe_fused_params_for_submodel(self, m):
        params_groups = get_params_groups_with_decay(
            model=m,
            lr_decay_rate=self.cfg.optim.layerwise_decay,
            patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult,
        )
        fused_params_groups = fuse_params_groups(params_groups)
        logger.info("fusing param groups")

        for g in fused_params_groups:
            g["foreach"] = True
        return fused_params_groups

    def get_params_groups(self):
        all_params_groups = []
        for m in self.student.values():
            all_params_groups += self.get_maybe_fused_params_for_submodel(m)
        return all_params_groups

    def prepare_for_distributed_training(self):
        logger.info("DISTRIBUTED FSDP -- preparing model for distributed training")
        if has_batchnorms(self.student):
            raise NotImplementedError
        # below will synchronize all student subnetworks across gpus:
        for k, v in self.student.items():
            student_model_cfg = self.cfg.compute_precision.student[k]
            self.student[k] = get_fsdp_wrapper(student_model_cfg, modules_to_wrap={BlockChunk})(self.student[k])



class MCMPMetaArch(nn.Module):

    def __init__(self, tile_dinov2_fair_config,
                 patch_dinov2_fair_config) -> None:
        super().__init__()
        self.tile = FullyShardedForwardArch(tile_dinov2_fair_config)
        self.patch = SSLMetaArch(patch_dinov2_fair_config)

        #self.n_global_crop = 2
        #self.n_local_crop = tile_dinov2_fair_config.crops.local_crops_number
        self.num_tile_per_patch = 192 // tile_dinov2_fair_config.crops.global_crops_size # TODO

        self.student_sampler = partial(
            sample_view,
            v_min=patch_dinov2_fair_config.crops.student_sample_min,
            v_max=patch_dinov2_fair_config.crops.student_sample_max)

        self.teacher_sampler = partial(
            sample_view,
            v_min=patch_dinov2_fair_config.crops.teacher_sample_min,
            v_max=patch_dinov2_fair_config.crops.teacher_sample_max)

        self.global_cropper = partial(
            crop_views,
            h0=patch_dinov2_fair_config.crops.global_crops_size,
            w0=patch_dinov2_fair_config.crops.global_crops_size)
        self.local_cropper = partial(
            crop_views,
            h0=patch_dinov2_fair_config.crops.local_crops_size,
            w0=patch_dinov2_fair_config.crops.local_crops_size)

        self.num_patch_global_crops = patch_dinov2_fair_config.crops.global_crops_number
        self.num_patch_local_crops = patch_dinov2_fair_config.crops.local_crops_number
        
        if self.patch.fp16_scaler is not None:
            self.fp16_scaler = ShardedGradScaler()
            self.patch.fp16_scaler = None

    def forward(self, inputs):
        raise NotImplementedError

    def backprop_loss(self, loss):
        if self.fp16_scaler is not None:
            self.fp16_scaler.scale(loss).backward()
        else:
            loss.backward()

    def forward_backward(self, images, teacher_temp, loss_scale=1.0):

        student_global_out = self.tile.forward_(images,
                                           return_student_emb=True)
        tile_global = einops.rearrange(student_global_out[:, 0, :],
                                       "(v b nh nw) d -> v b d nh nw",
                                       v=1,#self.n_global_crop,
                                       nh=self.num_tile_per_patch,
                                       nw=self.num_tile_per_patch)

        #tile_local = einops.rearrange(student_local_out[:, 0, :],
        #                              "(v b nh nw) d -> v b d nh nw",
        #                              #v=self.n_local_crop,
        #                              nh=self.num_tile_per_patch,
        #                              nw=self.num_tile_per_patch)
#
        #all_student_tokens = torch.cat((tile_global, tile_local))
#
        #with torch.no_grad():
        #    all_teacher_tokens = einops.rearrange(
        #        teacher_out[:, 0, :],
        #        "(v b nh nw) d -> v b d nh nw",
        #        v=self.n_global_crop,
        #        nh=self.num_tile_per_patch,
        #        nw=self.num_tile_per_patch)
#
        #student_tokens = self.student_sampler(all_student_tokens)
#
        #with torch.no_grad():
        #    teacher_tokens = self.teacher_sampler(all_teacher_tokens)

        student_tokens = tile_global.squeeze()
        student_global_tokens = [
            self.global_cropper([student_tokens])[0]
            for _ in range(self.num_patch_global_crops)
        ]
        #student_global_tokens = torch.stack(
        #    [i[0] for i in cropped_global_tokens])
        #teacher_global_tokens = torch.stack(
        #    [i[1] for i in cropped_global_tokens])
#
        student_local_tokens = torch.stack([
            self.local_cropper([student_tokens])[0]
            for _ in range(self.num_patch_local_crops)
        ])

        #cropped_global_tokens = torch.stack(cropped_global_tokens)

        patch_images = {
            "collated_global_crops":
            einops.rearrange(student_global_tokens,
                             "v b c h w -> (v b) c h w").contiguous(),
            "collated_local_crops":
            einops.rearrange(student_local_tokens,
                             "v b c h w -> (v b) c h w").contiguous()
        }
        patch_images.update(images["patch_masks"])


        patch_loss_accumulator, patch_loss_dict = self.patch.forward_(
            patch_images, teacher_temp)

        loss_dict = {f'patch_{k}': v for k, v in patch_loss_dict.items()}
        #loss_dict.update({f'tile_{k}': v for k, v in tile_loss_dict.items()})

        #loss_accumulator = 0.7 * tile_loss_accumulator + 0.3 * patch_loss_accumulator
        loss_accumulator = patch_loss_accumulator

        loss_accumulator *= loss_scale
        self.backprop_loss(loss_accumulator)
        self.fsdp_synchronize_streams()

        return loss_dict

    def fsdp_synchronize_streams(self):
        self.patch.fsdp_synchronize_streams()

    def update_teacher(self, m):
        self.patch.update_teacher(m)

    def train(self):
        self.tile.train()
        self.patch.train()

    def get_params_groups(self):
        return self.patch.get_params_groups()

    def prepare_for_distributed_training(self):
        self.tile.prepare_for_distributed_training()
        self.patch.prepare_for_distributed_training()
