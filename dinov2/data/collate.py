# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import random
import einops
import logging
from typing import Sequence, Literal

from torchvision import transforms
from dinov2.data.transforms import (
    GaussianBlur,
    make_normalize_transform,
)
from dinov2.data.augmentations import GaussianNoise


def collate_data_and_cast(samples_list,
                          mask_ratio_tuple,
                          mask_probability,
                          dtype,
                          n_tokens=None,
                          mask_generator=None):
    # dtype = torch.half  # TODO: Remove
    #torch.save(samples_list, "noaug_sample_list.pt")
    #exit(0)
    n_global_crops = len(samples_list[0][0]["global_crops"])
    n_local_crops = len(samples_list[0][0]["local_crops"])

    collated_global_crops = torch.stack([
        s[0]["global_crops"][i] for i in range(n_global_crops)
        for s in samples_list
    ])

    if n_local_crops:
        collated_local_crops = torch.stack([
            s[0]["local_crops"][i] for i in range(n_local_crops)
            for s in samples_list
        ])
    else:
        collated_local_crops = torch.empty(collated_global_crops.shape)

    B = len(collated_global_crops)
    N = n_tokens
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    masks_list = []
    for i in range(0, n_samples_masked):
        prob_min = probs[i]
        prob_max = probs[i + 1]
        masks_list.append(
            torch.BoolTensor(
                mask_generator(int(N * random.uniform(prob_min, prob_max)))))
        upperbound += int(N * prob_max)
    for i in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)

    collated_masks = torch.stack(masks_list).flatten(1)
    mask_indices_list = collated_masks.flatten().nonzero().flatten()

    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)
                    ).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

    return {
        "collated_global_crops":
        collated_global_crops.to(dtype),
        "collated_local_crops":
        collated_local_crops.to(dtype),
        "collated_masks":
        collated_masks,
        "mask_indices_list":
        mask_indices_list,
        "masks_weight":
        masks_weight,
        "upperbound":
        upperbound,
        "n_masked_patches":
        torch.full((1, ),
                   fill_value=mask_indices_list.shape[0],
                   dtype=torch.long),
    }


def collate_data_and_cast_with_context(samples_list,
                                       mask_ratio_tuple,
                                       mask_probability,
                                       dtype,
                                       n_tokens=None,
                                       mask_generator=None):

    data = collate_data_and_cast(samples_list,
                                 mask_ratio_tuple,
                                 mask_probability,
                                 dtype,
                                 n_tokens=n_tokens,
                                 mask_generator=mask_generator)
    data["context"] = torch.stack([i[0]["context"] for i in samples_list])
    return data


def create_mask(B, N, mask_ratio_tuple, mask_probability, mask_generator):
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    masks_list = []
    for i in range(0, n_samples_masked):
        prob_min = probs[i]
        prob_max = probs[i + 1]
        masks_list.append(
            torch.BoolTensor(
                mask_generator(int(N * random.uniform(prob_min, prob_max)))))
        upperbound += int(N * prob_max)
    for i in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)

    collated_masks = torch.stack(masks_list).flatten(1)
    mask_indices_list = collated_masks.flatten().nonzero().flatten()

    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)
                    ).unsqueeze(-1).expand_as(collated_masks)[collated_masks]
    n_masked_patches = torch.full((1, ),
                                  fill_value=mask_indices_list.shape[0],
                                  dtype=torch.long)

    return {
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": n_masked_patches
    }


def create_mask_with_bgswap_mask(B, N, mask_ratio_tuple, mask_probability,
                                 mask_generator, bgswap_mask_generator):
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    masks_list = []
    for i in range(0, n_samples_masked):
        prob_min = probs[i]
        prob_max = probs[i + 1]
        masks_list.append(
            torch.BoolTensor(
                mask_generator(int(N * random.uniform(prob_min, prob_max)))))
        upperbound += int(N * prob_max)
    for i in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)

    collated_masks = torch.stack(masks_list).flatten(1)
    mask_indices_list = collated_masks.flatten().nonzero().flatten()
    bgs_fg_masks = 1 - torch.stack([
        bgswap_mask_generator(return_flat_mask=True)
        for _ in range(len(masks_list))
    ])
    masks_weight = (1 / (collated_masks * bgs_fg_masks).sum(-1).clamp(min=1.0)
                    ).unsqueeze(-1).expand_as(collated_masks)[collated_masks]
    n_masked_patches = torch.full((1, ),
                                  fill_value=mask_indices_list.shape[0],
                                  dtype=torch.long)
    return {
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": n_masked_patches,
        #"bgswap_masked_masks": collated_masks * bgs_fg_masks,
    }


def collate_tile_data_and_cast_fmi(samples_list,
                                   mask_ratio_tuple,
                                   mask_probability,
                                   dtype,
                                   n_tokens=None,
                                   mask_generator=None):
    # dtype = torch.half  # TODO: Remove
    n_global_crops = len(samples_list[0][0]["global_crops"])
    n_local_crops = len(samples_list[0][0]["local_crops"])

    collated_global_crops = torch.stack([
        torch.stack([s[0]["global_crops"][i] for i in range(n_global_crops)])
        for s in samples_list
    ])
    collated_global_crops = einops.rearrange(
        collated_global_crops, "b v t c h w -> (v b t) c h w").contiguous()

    collated_local_crops = torch.stack([
        torch.stack([s[0]["local_crops"][i] for i in range(n_local_crops)])
        for s in samples_list
    ])
    collated_local_crops = einops.rearrange(
        collated_local_crops, "b v t c h w -> (v b t) c h w").contiguous()

    images = {
        "collated_global_crops": collated_global_crops.to(dtype),
        "collated_local_crops": collated_local_crops.to(dtype),
    }
    B = len(collated_global_crops)
    N = n_tokens

    images.update(
        create_mask(B, N, mask_ratio_tuple, mask_probability, mask_generator))

    return images


def collate_tile_patch_data_and_cast_fmi(samples_list,
                                         mask_ratio_tuple,
                                         mask_probability,
                                         dtype,
                                         n_tokens=None,
                                         mask_generator=None,
                                         patch_n_tokens=None,
                                         patch_mask_generator=None):

    images = collate_tile_data_and_cast_fmi(samples_list,
                                            mask_ratio_tuple,
                                            mask_probability,
                                            dtype,
                                            n_tokens=n_tokens,
                                            mask_generator=mask_generator)

    images["patch_masks"] = create_mask(
        len(samples_list) * len(samples_list[0][0]["global_crops"]),
        patch_n_tokens, mask_ratio_tuple, mask_probability,
        patch_mask_generator)

    return images


class OuterBiasedMasker():

    def __init__(self,
                 mask_size: int,
                 token_size: int,
                 dist_power: float = 2.0,
                 sharpness_range=[1, 5],
                 n_mask_token_perct_range=[.5, .75],
                 device='cpu',
                 dtype=torch.float32):
        """
        Args:
            mask_size (int): size of the square mask (mask_size x mask_size).
            power (float): strength of the bias toward outer elements.
            device (str): device to use for tensors ('cpu' or 'cuda').
            dtype (torch.dtype): dtype for probability tensor.
        """
        self.mask_size = mask_size
        self.token_size = token_size
        self.device = device
        self.dtype = dtype
        self.sharpness_range = sharpness_range
        self.n_mask_token_min = int(
            round(n_mask_token_perct_range[0] * mask_size * mask_size))
        self.n_mask_token_max = int(
            round(n_mask_token_perct_range[1] * mask_size * mask_size))

        # Precompute distance
        x = torch.arange(mask_size, device=device, dtype=dtype)
        y = torch.arange(mask_size, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(x, y, indexing='ij')

        cx, cy = (mask_size - 1) / 2, (mask_size - 1) / 2
        dist2 = (xx - cx)**dist_power + (yy - cy)**dist_power
        self.norm_dist = dist2 / dist2.max()  # normalize to [0, 1]
        self.upsample_kernel = torch.ones((token_size, token_size),
                                          dtype=torch.uint8,
                                          device=device)

    def __call__(self, return_flat_mask=False) -> torch.Tensor:
        """
        Sample a k x k binary mask with num_masked entries = 1, biased toward outer region.

        Args:
            num_masked (int): number of tokens to mask.

        Returns:
            torch.Tensor: mask of shape (k, k), dtype=torch.uint8.
        """
        num_masked = random.randint(self.n_mask_token_min,
                                    self.n_mask_token_max)
        sharpness = random.uniform(*self.sharpness_range)
        prob = self.norm_dist.pow(sharpness)
        prob = (prob / prob.sum()).flatten()
        flat_mask = torch.zeros(self.mask_size * self.mask_size,
                                dtype=torch.uint8,
                                device=self.device)
        idx = torch.multinomial(prob,
                                num_samples=num_masked,
                                replacement=False)
        flat_mask[idx] = 1
        base_mask = flat_mask.view(self.mask_size, self.mask_size)
        upsampled_full_mask = torch.kron(base_mask, self.upsample_kernel)
        if return_flat_mask:
            return flat_mask
        else:
            return upsampled_full_mask


class OuterCircularMasker:

    def __init__(
        self,
        mask_size: int,
        token_size: int,
        n_mask_token_perct_range: Sequence[float] = (.5, .75),
        mode: Literal["strict", "shell"] = "strict",
        outer_ring_width: int = 2,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        """
        Outer circular mask with two modes and an optional fixed outer square ring
        that is always masked.

        - outer_ring_width:
            Number of tokens from each border forming a square ring that is
            always masked (1). The circular masking is applied only to the
            remaining inner region.

        - mode="strict":
            Enforce the sampled masking fraction as closely as possible by
            randomly selecting within the boundary radius (within the inner region).
        - mode="shell":
            Keep all tokens at a given radius together (no splitting within a shell),
            so the actual masked fraction may deviate from the target.

        Args:
            mask_size: number of tokens per spatial side (mask_size x mask_size).
            token_size: upsampling factor per token (token_size x token_size).
            n_mask_token_perct_range: (min, max) fraction of tokens to mask
                within the inner region (excluding the fixed outer ring).
            mode: "strict" or "shell".
            outer_ring_width: number of tokens from border always masked (square ring).
            device: tensor device.
            dtype: dtype for distance computation.
        """
        self.mask_size = mask_size
        self.token_size = token_size
        self.device = device
        self.dtype = dtype
        self.frac_min, self.frac_max = n_mask_token_perct_range
        self.mode = mode
        self.outer_ring_width = int(outer_ring_width)

        if self.outer_ring_width < 0:
            raise ValueError("outer_ring_width must be >= 0")
        if self.outer_ring_width * 2 >= mask_size:
            raise ValueError(
                "outer_ring_width is too large: inner region would be empty "
                f"(outer_ring_width={self.outer_ring_width}, mask_size={mask_size})"
            )

        self.num_tokens = mask_size * mask_size

        # Coordinates and squared distances to center (for circular ordering).
        coords = torch.arange(mask_size, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")

        cx = (mask_size - 1) / 2.0
        cy = (mask_size - 1) / 2.0

        dist2 = (xx - cx)**2 + (yy - cy)**2  # [mask_size, mask_size]
        self.dist2_flat = dist2.reshape(-1)  # [num_tokens]

        # Define inner region (not part of the fixed outer ring).
        if self.outer_ring_width > 0:
            w = self.outer_ring_width
            inner = ((xx >= w) & (xx < mask_size - w) & (yy >= w) &
                     (yy < mask_size - w))
        else:
            inner = torch.ones_like(xx, dtype=torch.bool, device=device)

        inner_flat = inner.reshape(-1)
        self.interior_mask_flat = inner_flat  # True for inner tokens
        self.interior_indices = torch.nonzero(
            inner_flat, as_tuple=False).squeeze(1)  # [N_inner]
        self.ring_indices = torch.nonzero(~inner_flat, as_tuple=False).squeeze(
            1)  # [N_ring]
        self.num_tokens_inner = int(self.interior_indices.numel())

        # Distances restricted to inner region
        self.dist2_inner = self.dist2_flat[self.interior_indices]  # [N_inner]
        sorted_dist2_inner, order_inner = torch.sort(self.dist2_inner,
                                                     descending=False,
                                                     stable=True)
        self.sorted_dist2_inner = sorted_dist2_inner  # [N_inner]
        self.sorted_idx_inner = self.interior_indices[
            order_inner]  # [N_inner], global indices

        if mode == "strict" and self.num_tokens_inner > 0:
            # Shells over inner tokens only
            unique_vals, counts = torch.unique(self.sorted_dist2_inner,
                                               return_counts=True)
            start = torch.zeros_like(counts)
            start[1:] = torch.cumsum(counts[:-1], dim=0)

            self.shell_radii = unique_vals  # [S_inner]
            self.shell_counts = counts  # [S_inner]
            self.shell_start = start  # [S_inner]

        # Upsampling kernel for tokens -> pixels
        self.upsample_kernel = torch.ones(
            (token_size, token_size),
            dtype=torch.uint8,
            device=device,
        )

    def __call__(self) -> torch.Tensor:
        """
        Generate a mask where:
          - an outer square ring of width `outer_ring_width` is always masked (1),
          - the remaining inner region is masked according to a sampled fraction
            in [frac_min, frac_max] using a circular ordering.

        Returns:
            torch.Tensor of shape
            (mask_size * token_size, mask_size * token_size), dtype=torch.uint8.
        """
        # Start with all unmasked (0)
        mask_flat = torch.zeros(self.num_tokens,
                                dtype=torch.uint8,
                                device=self.device)

        # Always-mask outer ring (if any)
        if self.outer_ring_width > 0 and self.ring_indices.numel() > 0:
            mask_flat[self.ring_indices] = 1

        # If no inner tokens exist, we are done (whole map is outer ring)
        if self.num_tokens_inner == 0:
            base_mask = mask_flat.view(self.mask_size, self.mask_size)
            return torch.kron(base_mask, self.upsample_kernel)

        # Sample desired masked fraction over the INNER region only
        frac = random.uniform(self.frac_min, self.frac_max)
        num_masked_target_inner = int(round(frac * self.num_tokens_inner))
        num_masked_target_inner = max(
            0, min(self.num_tokens_inner, num_masked_target_inner))
        num_unmasked_target_inner = self.num_tokens_inner - num_masked_target_inner

        inner_idx = self.interior_indices  # global indices of inner tokens

        # Edge cases for inner region
        if num_unmasked_target_inner <= 0:
            # All inner tokens masked
            mask_flat[inner_idx] = 1
        elif num_unmasked_target_inner >= self.num_tokens_inner:
            # All inner tokens unmasked (outer ring already masked)
            # mask_flat[inner_idx] already 0, so nothing to do
            pass
        else:
            if self.mode == "shell":
                # Choose radius threshold so that all points with same radius are treated identically.
                k = num_unmasked_target_inner - 1
                thr_dist2 = self.sorted_dist2_inner[k]
                # Inner region: dist2 >= thr_dist2 -> masked (1), else unmasked (0)
                mask_inner = (self.dist2_inner >= thr_dist2).to(torch.uint8)
                mask_flat[inner_idx] = mask_inner

            elif self.mode == "strict":
                # Initialize inner region as fully masked
                mask_flat[inner_idx] = 1

                remaining = num_unmasked_target_inner
                unmasked_indices = []

                if self.num_tokens_inner > 0:
                    for shell_start, shell_count in zip(
                            self.shell_start, self.shell_counts):
                        shell_start = int(shell_start.item())
                        shell_count = int(shell_count.item())

                        if remaining <= 0:
                            break

                        shell_idx = self.sorted_idx_inner[
                            shell_start:shell_start +
                            shell_count]  # global indices for this shell

                        if remaining >= shell_count:
                            # Take the entire shell as unmasked
                            unmasked_indices.append(shell_idx)
                            remaining -= shell_count
                        else:
                            # Boundary shell: randomly select 'remaining' tokens in this shell
                            perm = torch.randperm(
                                shell_count, device=self.device)[:remaining]
                            chosen = shell_idx[perm]
                            unmasked_indices.append(chosen)
                            remaining = 0
                            break

                if unmasked_indices:
                    unmasked_indices = torch.cat(unmasked_indices, dim=0)
                    mask_flat[unmasked_indices] = 0  # 0 = unmasked
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

        base_mask = mask_flat.view(self.mask_size, self.mask_size)
        return torch.kron(base_mask, self.upsample_kernel)


class CellCollator():

    def __init__(self,
                 mask_ratio_tuple,
                 mask_probability,
                 dtype,
                 omb_params,
                 omb_which="OuterBiasedMasker",
                 n_tokens=None,
                 mask_generator=None,
                 mask_outerbias_weight=False):

        self.mask_ratio_tuple = mask_ratio_tuple
        self.mask_probability = mask_probability
        self.dtype = dtype
        self.n_tokens = n_tokens
        self.mask_generator = mask_generator

        self.color_jittering = transforms.Compose(
            [  # for local views, deferred
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(brightness=0.4,
                                               contrast=0.4,
                                               saturation=0.2,
                                               hue=0.1)
                    ],
                    p=0.8,
                ),
                transforms.RandomApply(
                    [GaussianNoise(min_var=0, max_var=0.05)], p=0.5),
                GaussianBlur(p=0.5, radius_max=1),
                transforms.RandomSolarize(threshold=0.5, p=.2),
                transforms.RandomGrayscale(p=0.2),
            ])
        #self.color_jittering = transforms.Compose([])

        if omb_which == "OuterBiasedMasker":
            self.obm = OuterBiasedMasker(**omb_params)
        elif omb_which == "OuterCircularMasker":
            self.obm = OuterCircularMasker(**omb_params)
        else:
            assert False

        self.mask_outerbias_weight = mask_outerbias_weight

    def __call__(self, samples_list):
        n_global_crops = len(samples_list[0][0]["global_crops"])
        n_local_crops = len(samples_list[0][0]["local_crops"])

        collated_global_crops = torch.stack([
            s[0]["global_crops"][i] for i in range(n_global_crops)
            for s in samples_list
        ])
        collated_local_crops = torch.stack([
            s[0]["local_crops"][i] for i in range(n_local_crops)
            for s in samples_list
        ])

        bg_swap_masks = torch.stack([
            self.obm() for _ in range(len(collated_local_crops))
        ]).unsqueeze(1)

        collated_local_crops = (
            collated_local_crops *
            (1 - bg_swap_masks) + collated_local_crops[torch.randperm(
                collated_local_crops.size(0))] * bg_swap_masks)
        collated_local_crops = torch.stack(
            [self.color_jittering(i) for i in collated_local_crops])

        images = {
            "collated_global_crops": collated_global_crops.to(self.dtype),
            "collated_local_crops": collated_local_crops.to(self.dtype),
            "bg_swap_masks": bg_swap_masks,
        }
        B = len(collated_global_crops)
        N = self.n_tokens

        if self.mask_outerbias_weight:
            images.update(
                create_mask_with_bgswap_mask(B, N, self.mask_ratio_tuple,
                                             self.mask_probability,
                                             self.mask_generator, self.obm))
        else:
            images.update(
                create_mask(
                    B,
                    N,
                    self.mask_ratio_tuple,
                    self.mask_probability,
                    self.mask_generator,
                ))

        return images
