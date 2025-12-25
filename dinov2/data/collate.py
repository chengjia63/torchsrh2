# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import random
import einops
import logging

from torchvision import transforms
from dinov2.data.transforms import (
    GaussianBlur,
    make_normalize_transform,
)
from dinov2.data.augmentations import GaussianNoise

def collate_data_and_cast(samples_list, mask_ratio_tuple, mask_probability, dtype, n_tokens=None, mask_generator=None):
    # dtype = torch.half  # TODO: Remove
    #torch.save(samples_list, "noaug_sample_list.pt")
    #exit(0)
    n_global_crops = len(samples_list[0][0]["global_crops"])
    n_local_crops = len(samples_list[0][0]["local_crops"])

    collated_global_crops = torch.stack([s[0]["global_crops"][i] for i in range(n_global_crops) for s in samples_list])

    if n_local_crops:
        collated_local_crops = torch.stack([s[0]["local_crops"][i] for i in range(n_local_crops) for s in samples_list])
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
        masks_list.append(torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max)))))
        upperbound += int(N * prob_max)
    for i in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)

    collated_masks = torch.stack(masks_list).flatten(1)
    mask_indices_list = collated_masks.flatten().nonzero().flatten()

    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

    return {
        "collated_global_crops": collated_global_crops.to(dtype),
        "collated_local_crops": collated_local_crops.to(dtype),
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
    }



def collate_data_and_cast_with_context(samples_list, mask_ratio_tuple, mask_probability, dtype, n_tokens=None, mask_generator=None):
    
    data = collate_data_and_cast(samples_list, mask_ratio_tuple,
        mask_probability, dtype, n_tokens=n_tokens,
        mask_generator=mask_generator)
    data["context"] = torch.stack([i[0]["context"] for i in samples_list])
    return data


def create_mask(B,N, mask_ratio_tuple, mask_probability, mask_generator):
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    masks_list = []
    for i in range(0, n_samples_masked):
        prob_min = probs[i]
        prob_max = probs[i + 1]
        masks_list.append(torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max)))))
        upperbound += int(N * prob_max)
    for i in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)

    collated_masks = torch.stack(masks_list).flatten(1)
    mask_indices_list = collated_masks.flatten().nonzero().flatten()

    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

    return {
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
    }


def collate_tile_data_and_cast_fmi(samples_list, mask_ratio_tuple, mask_probability, dtype, n_tokens=None, mask_generator=None):
    # dtype = torch.half  # TODO: Remove
    n_global_crops = len(samples_list[0][0]["global_crops"])
    n_local_crops = len(samples_list[0][0]["local_crops"])

    collated_global_crops = torch.stack([torch.stack([s[0]["global_crops"][i] for i in range(n_global_crops)]) for s in samples_list])
    collated_global_crops = einops.rearrange(collated_global_crops, "b v t c h w -> (v b t) c h w").contiguous()

    collated_local_crops = torch.stack([torch.stack([s[0]["local_crops"][i] for i in range(n_local_crops)]) for s in samples_list])
    collated_local_crops = einops.rearrange(collated_local_crops, "b v t c h w -> (v b t) c h w").contiguous()

    images = {
        "collated_global_crops": collated_global_crops.to(dtype),
        "collated_local_crops": collated_local_crops.to(dtype),
    }
    B = len(collated_global_crops)
    N = n_tokens

    images.update(create_mask(B,N,mask_ratio_tuple, mask_probability, mask_generator))

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
                 mask_size:int,
                 token_size:int,
                 dist_power:float=2.0,
                 sharpness_range=[1,5],
                 n_mask_token_perct_range=[.5, .75],
                 device='cpu', dtype=torch.float32):
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
        self.n_mask_token_min = int(round(n_mask_token_perct_range[0] * mask_size * mask_size))
        self.n_mask_token_max = int(round(n_mask_token_perct_range[1] * mask_size * mask_size))

        # Precompute distance
        x = torch.arange(mask_size, device=device, dtype=dtype)
        y = torch.arange(mask_size, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(x, y, indexing='ij')

        cx, cy = (mask_size - 1) / 2, (mask_size - 1) / 2
        dist2 = (xx - cx) ** dist_power + (yy - cy) ** dist_power
        self.norm_dist = dist2 / dist2.max()  # normalize to [0, 1]
        self.upsample_kernel = torch.ones((token_size, token_size), dtype=torch.uint8, device=device)

    def __call__(self) -> torch.Tensor:
        """
        Sample a k x k binary mask with num_masked entries = 1, biased toward outer region.

        Args:
            num_masked (int): number of tokens to mask.

        Returns:
            torch.Tensor: mask of shape (k, k), dtype=torch.uint8.
        """
        num_masked = random.randint(self.n_mask_token_min, self.n_mask_token_max)
        sharpness = random.uniform(*self.sharpness_range)
        prob = self.norm_dist.pow(sharpness)
        prob = (prob / prob.sum()).flatten()
        flat_mask = torch.zeros(self.mask_size * self.mask_size, dtype=torch.uint8, device=self.device)
        idx = torch.multinomial(prob, num_samples=num_masked, replacement=False)
        flat_mask[idx] = 1
        base_mask = flat_mask.view(self.mask_size, self.mask_size)
        return torch.kron(base_mask, self.upsample_kernel)


class CellCollator():
    def __init__(self, mask_ratio_tuple, mask_probability, dtype, omb_params, n_tokens=None, mask_generator=None):
        self.mask_ratio_tuple = mask_ratio_tuple
        self.mask_probability = mask_probability
        self.dtype = dtype
        self.n_tokens = n_tokens
        self.mask_generator = mask_generator
        
        self.color_jittering = transforms.Compose([
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
                ],
                p=0.8,
            ),
            transforms.RandomApply(
                [GaussianNoise(min_var=0, max_var=0.05)],p=0.5),
            GaussianBlur(p=0.5, radius_max=1),            
            transforms.RandomSolarize(threshold=0.5, p=.2),
            transforms.RandomGrayscale(p=0.2),
        ])
        self.obm = OuterBiasedMasker(**omb_params)

    def __call__(self, samples_list):
        n_global_crops = len(samples_list[0][0]["global_crops"])
        n_local_crops = len(samples_list[0][0]["local_crops"])

        collated_global_crops = torch.stack([
            s[0]["global_crops"][i]
            for i in range(n_global_crops)
            for s in samples_list
        ])
        collated_local_crops = torch.stack([
            s[0]["local_crops"][i]
            for i in range(n_local_crops)
            for s in samples_list
        ])
        
        bg_swap_masks = torch.stack([self.obm()
                                     for _ in range(len(collated_local_crops))]).unsqueeze(1)
        collated_local_crops = (collated_local_crops * (1-bg_swap_masks) + 
                                collated_local_crops[torch.randperm(collated_local_crops.size(0))] * bg_swap_masks)
        collated_local_crops = torch.stack([self.color_jittering(i)
                                            for i in collated_local_crops])
        
        images = {
            "collated_global_crops": collated_global_crops.to(self.dtype),
            "collated_local_crops": collated_local_crops.to(self.dtype),
            "bg_swap_masks": bg_swap_masks,
        }
        B = len(collated_global_crops)
        N = self.n_tokens
        images.update(create_mask(B,N, self.mask_ratio_tuple, self.mask_probability, self.mask_generator))
        
        return images