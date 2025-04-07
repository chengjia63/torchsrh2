# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

from torchvision import transforms

from .transforms import (
    GaussianBlur,
    make_normalize_transform,
)
import einops
import torch

logger = logging.getLogger("dinov2")




class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")

        # random resized crop and flip
        self.geometric_augmentation_global = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        self.geometric_augmentation_local = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        # color distorsions / blurring
        color_jittering = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        global_transfo1_extra = GaussianBlur(p=1.0)

        global_transfo2_extra = transforms.Compose(
            [
                GaussianBlur(p=0.1),
                #transforms.RandomSolarize(threshold=128, p=0.2),
                transforms.RandomSolarize(threshold=0.5, p=0.2),
            ]
        )

        local_transfo_extra = GaussianBlur(p=0.5)

        # normalization
        self.normalize = transforms.Compose(
            [
                #transforms.ToTensor(),
                transforms.ConvertImageDtype(dtype=torch.float16),
                make_normalize_transform(),
            ]
        )

        self.global_transfo1 = transforms.Compose([color_jittering, global_transfo1_extra])#, self.normalize])
        self.global_transfo2 = transforms.Compose([color_jittering, global_transfo2_extra])#, self.normalize])
        self.local_transfo = transforms.Compose([color_jittering, local_transfo_extra])#, self.normalize])



    def __call__(self, image):
        output = {}
        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1 = torch.nan_to_num(self.global_transfo1(im1_base), nan=0)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2 = torch.nan_to_num(self.global_transfo2(im2_base), nan=0)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # local crops:
        local_crops = [
            torch.nan_to_num(self.local_transfo(self.geometric_augmentation_local(image)), nan=0) for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output

class TileDataAugmentationDINO(DataAugmentationDINO):
    def __init__(self, tile_height, tile_width, **kwargs):
        super().__init__(**kwargs)
        self.rh = tile_height
        self.rw = tile_width

    def __call__(self, image):
        batched_regions = einops.rearrange(
                image,
                "c (nh rh) (nw rw) -> (nh nw) c rh rw",
                rh=self.rh,
                rw=self.rw)
        return super().__call__(batched_regions)
