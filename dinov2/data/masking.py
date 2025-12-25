# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import random
import math
import numpy as np
import torch

class MaskingGenerator:
    def __init__(
        self,
        input_size,
        num_masking_patches=None,
        min_num_patches=4,
        max_num_patches=None,
        min_aspect=0.3,
        max_aspect=None,
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height,
            self.width,
            self.min_num_patches,
            self.max_num_patches,
            self.num_masking_patches,
            self.log_aspect_ratio[0],
            self.log_aspect_ratio[1],
        )
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for _ in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top : top + h, left : left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self, num_masking_patches=0):
        mask = np.zeros(shape=self.get_shape(), dtype=bool)
        mask_count = 0
        while mask_count < num_masking_patches:
            max_mask_patches = num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        return mask

class OuterBiasedTokenMasker():
    def __init__(self,
                 mask_size:int,
                 dist_power:float=2.0,
                 sharpness_range=[1, 5],
                 device='cpu', dtype=torch.float32):
        """
        Args:
            mask_size (int): size of the square mask (mask_size x mask_size).
            power (float): strength of the bias toward outer elements.
            device (str): device to use for tensors ('cpu' or 'cuda').
            dtype (torch.dtype): dtype for probability tensor.
        """
        self.mask_size = mask_size
        self.device = device
        self.dtype = dtype
        self.sharpness_range = sharpness_range

        # Precompute distance
        x = torch.arange(mask_size, device=device, dtype=dtype)
        y = torch.arange(mask_size, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(x, y, indexing='ij')

        cx, cy = (mask_size - 1) / 2, (mask_size - 1) / 2
        dist2 = (xx - cx) ** dist_power + (yy - cy) ** dist_power
        self.norm_dist = dist2 / dist2.max()  # normalize to [0, 1]

    def __call__(self, num_masked) -> torch.Tensor:
        """
        Sample a k x k binary mask with num_masked entries = 1, biased toward outer region.

        Args:
            num_masked (int): number of tokens to mask.

        Returns:
            torch.Tensor: mask of shape (k, k), dtype=torch.uint8.
        """
        sharpness = random.uniform(*self.sharpness_range)
        prob = self.norm_dist.pow(sharpness)
        prob = (prob / prob.sum()).flatten()
        flat_mask = torch.zeros(self.mask_size * self.mask_size, dtype=torch.uint8, device=self.device)
        if num_masked == 0:
            return (flat_mask.view(self.mask_size, self.mask_size)).to(bool)   
        else:
            idx = torch.multinomial(prob, num_samples=num_masked, replacement=False)
            flat_mask[idx] = 1
            return (flat_mask.view(self.mask_size, self.mask_size)).to(bool)


