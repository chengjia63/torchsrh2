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


class MCMMetaArch(nn.Module):

    def __init__(self, tile_dinov2_fair_config,
                 patch_dinov2_fair_config) -> None:
        super().__init__()
        self.tile = SSLMetaArch(tile_dinov2_fair_config)
        self.patch = SSLMetaArch(patch_dinov2_fair_config)

        self.n_global_crop = 2
        self.n_local_crop = tile_dinov2_fair_config.crops.local_crops_number
        self.num_tile_per_patch = 288 // tile_dinov2_fair_config.crops.global_crops_size

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
        if ((self.tile.fp16_scaler is not None)
                or (self.patch.fp16_scaler is not None)):
            self.fp16_scaler = ShardedGradScaler()
            self.tile.fp16_scaler = None
            self.patch.fp16_scaler = None

    def forward(self, inputs):
        raise NotImplementedError

    def backprop_loss(self, loss):
        if self.fp16_scaler is not None:
            self.fp16_scaler.scale(loss).backward()
        else:
            loss.backward()

    def forward_backward(self, images, teacher_temp):
        #TODO

        (tile_loss_accumulator, tile_loss_dict, student_global_out,
         student_local_out,
         teacher_out) = self.tile.forward_(images,
                                           teacher_temp,
                                           return_student_emb=True)

        tile_global = einops.rearrange(student_global_out[:, 0, :],
                                       "(v b nh nw) d -> v b d nh nw",
                                       v=self.n_global_crop,
                                       nh=self.num_tile_per_patch,
                                       nw=self.num_tile_per_patch)

        tile_local = einops.rearrange(student_local_out[:, 0, :],
                                      "(v b nh nw) d -> v b d nh nw",
                                      v=self.n_local_crop,
                                      nh=self.num_tile_per_patch,
                                      nw=self.num_tile_per_patch)

        all_student_tokens = torch.cat((tile_global, tile_local))

        with torch.no_grad():
            all_teacher_tokens = einops.rearrange(
                teacher_out[:, 0, :],
                "(v b nh nw) d -> v b d nh nw",
                v=self.n_global_crop,
                nh=self.num_tile_per_patch,
                nw=self.num_tile_per_patch)

        student_tokens = self.student_sampler(all_student_tokens)

        with torch.no_grad():
            teacher_tokens = self.teacher_sampler(all_teacher_tokens)

        cropped_global_tokens = [
            self.global_cropper([student_tokens, teacher_tokens])
            for _ in range(self.num_patch_global_crops)
        ]
        student_global_tokens = torch.stack(
            [i[0] for i in cropped_global_tokens])
        teacher_global_tokens = torch.stack(
            [i[1] for i in cropped_global_tokens])

        student_local_tokens = torch.stack([
            self.local_cropper([student_tokens])[0]
            for _ in range(self.num_patch_local_crops)
        ])

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
        loss_dict.update({f'tile_{k}': v for k, v in tile_loss_dict.items()})

        #loss_accumulator = 0.7 * tile_loss_accumulator + 0.3 * patch_loss_accumulator
        loss_accumulator = patch_loss_accumulator

        self.backprop_loss(loss_accumulator)
        self.fsdp_synchronize_streams()

        return loss_dict

    def fsdp_synchronize_streams(self):
        self.tile.fsdp_synchronize_streams()
        self.patch.fsdp_synchronize_streams()

    def update_teacher(self, m):
        self.tile.update_teacher(m)
        self.patch.update_teacher(m)

    def train(self):
        self.tile.train()
        self.patch.train()

    def get_params_groups(self):
        return self.tile.get_params_groups() + self.patch.get_params_groups()

    def prepare_for_distributed_training(self):
        self.tile.prepare_for_distributed_training()
        self.patch.prepare_for_distributed_training()
