# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import torch
from torchvision import transforms
import einops
import random
from .transforms import (
    GaussianBlur,
    make_normalize_transform,
)

logger = logging.getLogger("dinov2")
from functools import partial

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
        self.geometric_augmentation_global = transforms.Compose([
            transforms.RandomResizedCrop(
                global_crops_size,
                scale=global_crops_scale,
                interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
        ])

        self.geometric_augmentation_local = transforms.Compose([
            transforms.RandomResizedCrop(
                local_crops_size,
                scale=local_crops_scale,
                interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
        ])

        # color distorsions / blurring
        color_jittering = transforms.Compose([
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
                ],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
        ])

        global_transfo1_extra = GaussianBlur(p=1.0)

        global_transfo2_extra = transforms.Compose([
            GaussianBlur(p=0.1),
            transforms.RandomSolarize(threshold=128, p=0.2),
        ])

        local_transfo_extra = GaussianBlur(p=0.5)

        # normalization
        self.normalize = transforms.Compose([
            #transforms.ToTensor(),
            transforms.ConvertImageDtype(dtype=float),
            make_normalize_transform(),
        ])

        self.global_transfo1 = transforms.Compose(
            [color_jittering, global_transfo1_extra, self.normalize])
        self.global_transfo2 = transforms.Compose(
            [color_jittering, global_transfo2_extra, self.normalize])
        self.local_transfo = transforms.Compose(
            [color_jittering, local_transfo_extra, self.normalize])

    def __call__(self, image):
        output = {}

        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1 = self.global_transfo1(im1_base)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2 = self.global_transfo2(im2_base)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # local crops:
        local_crops = [
            self.local_transfo(self.geometric_augmentation_local(image))
            for _ in range(self.local_crops_number)
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


class TileContextMultiCropDataAugmentationDINO(DataAugmentationDINO):
    def __init__(self, tile_size, context_size, **kwargs):
        super().__init__(**kwargs)
        self.tile_crop = transforms.RandomCrop(size=tile_size)
        self.context_crop = transforms.RandomCrop(size=context_size)

    def __call__(self, image):
        aug_tile = super().__call__(self.tile_crop(image))
        aug_tile["context"] = self.context_crop(image)
        return aug_tile



class DataAugmentationDINONoNormalize(object):

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
        self.geometric_augmentation_global = transforms.Compose([
            transforms.RandomResizedCrop(
                global_crops_size,
                scale=global_crops_scale,
                interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
        ])

        self.geometric_augmentation_local = transforms.Compose([
            transforms.RandomResizedCrop(
                local_crops_size,
                scale=local_crops_scale,
                interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
        ])

        # color distorsions / blurring
        color_jittering = transforms.Compose([
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
                ],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
        ])

        global_transfo1_extra = GaussianBlur(p=1.0)

        global_transfo2_extra = transforms.Compose([
            GaussianBlur(p=0.1),
            transforms.RandomSolarize(threshold=0.5, p=0.2),
        ])

        local_transfo_extra = GaussianBlur(p=0.5)

        # normalization
        #self.normalize = transforms.Compose([
        #    #transforms.ToTensor(),
        #    transforms.ConvertImageDtype(dtype=float),
        #    make_normalize_transform(),
        #])

        self.global_transfo1 = transforms.Compose(
            [color_jittering, global_transfo1_extra]) #, self.normalize])
        self.global_transfo2 = transforms.Compose(
            [color_jittering, global_transfo2_extra]) #, self.normalize])
        self.local_transfo = transforms.Compose(
            [color_jittering, local_transfo_extra]) #, self.normalize])

        self.dropna = partial(torch.nan_to_num, nan=1.0e-6, posinf=1, neginf=1.0e-6)
    def __call__(self, image):
        output = {}

        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1 = self.dropna(self.global_transfo1(im1_base))

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2 = self.dropna(self.global_transfo2(im2_base))

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # local crops:
        local_crops = [
            self.dropna(self.local_transfo(self.geometric_augmentation_local(image)))
            for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output


class GaussianNoise(torch.nn.Module):
    """object to add guassian noise to images."""

    def __init__(self, min_var: float = 0.01, max_var: float = 0.1):
        super().__init__()
        self.min_var = min_var
        self.max_var = max_var

    def forward(self, tensor):
        min_, max_ = tensor.min(), tensor.max()
        var = random.uniform(self.min_var, self.max_var)
        noisy = tensor + torch.randn(tensor.size()) * var
        return torch.clamp(noisy, min=min_, max=max_)


class AggressiveDataAugmentationDINONoNormalize(object):

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
        self.geometric_augmentation_global = transforms.Compose([
            transforms.RandomResizedCrop(
                global_crops_size,
                scale=global_crops_scale,
                interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
        ])

        self.geometric_augmentation_local = transforms.Compose([
            transforms.RandomResizedCrop(
                local_crops_size,
                scale=local_crops_scale,
                interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
        ])

        # color distorsions / blurring
        color_jittering = transforms.Compose([
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
                ],
                p=0.8,
            ),
            transforms.RandomApply(
                [GaussianNoise(min_var=0, max_var=0.1)],p=0.5), 
            transforms.RandomGrayscale(p=0.2),
        ])

        global_transfo1_extra = transforms.Compose([GaussianBlur(p=1.0,radius_max=1)])

        global_transfo2_extra = transforms.Compose([
            GaussianBlur(p=0.1),
            transforms.RandomSolarize(threshold=0.5, p=0.2),
        ])

        local_transfo_extra = transforms.Compose([GaussianBlur(p=0.5,radius_max=1)])

        # normalization
        #self.normalize = transforms.Compose([
        #    #transforms.ToTensor(),
        #    transforms.ConvertImageDtype(dtype=float),
        #    make_normalize_transform(),
        #])

        self.global_transfo1 = transforms.Compose(
            [color_jittering, global_transfo1_extra]) #, self.normalize])
        self.global_transfo2 = transforms.Compose(
            [color_jittering, global_transfo2_extra]) #, self.normalize])
        self.local_transfo = transforms.Compose(
            [color_jittering, local_transfo_extra]) #, self.normalize])

        self.instance_norm = torch.nn.InstanceNorm2d(num_features=3)
        self.mean = torch.tensor([0.0872, 0.1546, 0.1604]).unsqueeze(-1).unsqueeze(-1)
        self.std = torch.tensor([0.0444, 0.0939, 0.0560]).unsqueeze(-1).unsqueeze(-1)
    def __call__(self, image):
        output = {}

        image = (self.instance_norm(image) * self.std + self.mean).clamp(0, 1)
 
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
            torch.nan_to_num(self.local_transfo(self.geometric_augmentation_local(image)), nan=0)
            for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output


class TileContextMultiCropDataAugmentationNoNormalizeDINO(DataAugmentationDINONoNormalize):
    def __init__(self, tile_size, context_size, **kwargs):
        super().__init__(**kwargs)
        self.tile_crop = transforms.RandomCrop(size=tile_size)
        self.context_crop = transforms.RandomCrop(size=context_size)


    def __call__(self, image):
        aug_tile = super().__call__(self.tile_crop(image))
        aug_tile["context"] = self.context_crop(image)
        return aug_tile

class DataAugmentationHiDiscDINO(object):

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
        self.geometric_augmentation_global = transforms.Compose([
            transforms.RandomResizedCrop(
                global_crops_size,
                scale=global_crops_scale,
                interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
        ])

        self.geometric_augmentation_local = transforms.Compose([
            transforms.RandomResizedCrop(
                local_crops_size,
                scale=local_crops_scale,
                interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
        ])

        # color distorsions / blurring
        color_jittering = transforms.Compose([
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
                ],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
        ])

        global_transfo_extra = transforms.Compose([
            GaussianBlur(p=0.1),
            transforms.RandomSolarize(threshold=128, p=0.2),
        ])

        local_transfo_extra = GaussianBlur(p=0.5)

        # normalization
        self.normalize = transforms.Compose([
            #transforms.ToTensor(),
            transforms.ConvertImageDtype(dtype=float),
            make_normalize_transform(),
        ])

        self.global_transfo = transforms.Compose(
            [color_jittering, global_transfo_extra, self.normalize])
        self.local_transfo = transforms.Compose(
            [color_jittering, local_transfo_extra, self.normalize])

    def __call__(self, image):
        output = {}

        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1 = self.global_transfo(im1_base)

        output["global_crops"] = [global_crop_1]

        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1]

        # local crops:
        local_crops = [
            self.local_transfo(self.geometric_augmentation_local(image))
            for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output

class RandomRangeCrop(torch.nn.Module):
    """
    Always crop a square patch of side `size` from a fixed-size image (HxH),
    where both the row and column start are sampled independently from
    the same [min_start, max_start] interval (inclusive).
    Designed to crop 64x64 to 48x48. top left range would be 0,0 - 16,16. shift factor should be between [0-7].
    0 would be most aggressive.
    Args:
        size (int): side length of the square crop.
        shift_factor (int): how much shift to exclude
        img_size (int): the constant height/width H of all input images.
    """
    def __init__(self, size: int, shift_factor: int):
        super().__init__()
        self.size = size
        mn, mx = shift_factor, 16-shift_factor

        self.min_start = mn
        self.max_start = mx

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img (Tensor): shape (C, H, H)

        Returns:
            Tensor: cropped image of shape (C, size, size)
        """
        # draw row and column starts independently from [min_start, max_start]
        i = torch.randint(self.min_start, self.max_start + 1, ()).item()
        j = torch.randint(self.min_start, self.max_start + 1, ()).item()

        return img[:, i : i + self.size, j : j + self.size]
    
class CellAugmentationDINO(object):

    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=48,
        local_crops_size=48,
        gaussian_noise_params={"min_var":0,"max_var":0.05},

    ):
        del global_crops_scale
        del local_crops_scale
        assert global_crops_size == local_crops_size

        self.local_crops_number = local_crops_number
        self.crops_size = local_crops_size

        logging.info("###################################")
        logging.info("Using data augmentation parameters:")
        logging.info(f"local_crops_number: {local_crops_number}")
        logging.info(f"crops_size: {local_crops_size}")
        logging.info("###################################")

        # random resized crop and flip
        geometric_augmentation_global = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            RandomRangeCrop(size=self.crops_size, shift_factor=4)
        ])


        # color distorsions / blurring
        color_jittering = transforms.Compose([
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
                ],
                p=0.8,
            ),
            transforms.RandomApply(
                [GaussianNoise(**gaussian_noise_params)],p=0.5),
            GaussianBlur(p=0.5, radius_max=1),            
            transforms.RandomSolarize(threshold=0.5, p=.2),
            transforms.RandomGrayscale(p=0.2),
        ])

       
        self.global_transfo = transforms.Compose(
            [geometric_augmentation_global, color_jittering]) #, self.normalize])
        self.local_transfo = transforms.Compose(
            [geometric_augmentation_global]) #, self.normalize]) # need to defer color jittering
        self.dropna = partial(torch.nan_to_num, nan=1.0e-6, posinf=1, neginf=1.0e-6)
        
        
    def __call__(self, image):
        output = {}

        global_crops = [self.dropna(self.global_transfo(image))
                        for _ in range(2)]
        local_crops = [self.dropna(self.local_transfo(image))
                        for _ in range(self.local_crops_number)]
        
        output["local_crops"] = local_crops
        output["global_crops"] = global_crops
        output["global_crops_teacher"] = global_crops
        output["offsets"] = ()

        return output
