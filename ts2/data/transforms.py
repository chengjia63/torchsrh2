import random
import logging
from functools import partial
from typing import List, Tuple, Dict, Optional, Callable, Any

import numpy as np
from PIL import Image
from tifffile import imread
from skimage.filters import gaussian

import torch
from torch.nn import ModuleList
from torch.fft import fft2, fftshift, ifft2, ifftshift

from torchvision.transforms import (
    Compose, Resize, RandomCrop, Normalize, RandomAffine, RandomApply,
    RandomHorizontalFlip, RandomVerticalFlip, ColorJitter, GaussianBlur,
    RandomErasing, RandomAutocontrast, RandomSolarize, RandomAdjustSharpness,
    Grayscale, RandomResizedCrop, RandomGrayscale)
from torchvision.transforms import RandomEqualize, RandomPosterize, ConvertImageDtype
from torchvision.transforms import functional as F

from torch import Tensor

from dinov2.data.augmentations import DataAugmentationDINO
from dinov2.data.transforms import (make_normalize_transform)


class HistologyTransform(torch.nn.Module):
    """Transformation module for histology data training"""

    def __init__(self, which_set, base_aug_params, strong_aug_params):
        super().__init__()
        base_augs = {
            "srh": SRHBaseTransform,
            "he": NoBaseTransform,
            "cifar": VisionBaseTransform
        }
        self.base_aug = base_augs[which_set](**base_aug_params)
        self.strong_aug = StrongTransform(**strong_aug_params)

    def forward(self, x: Tensor) -> Tensor:  # pylint: disable=missing-function-docstring
        return self.strong_aug(self.base_aug(x))


class SRHBaseTransform(torch.nn.Module):
    """Base transformations for SRH training."""

    def __init__(self, laser_noise_config=None, base_aug="three_channels"):
        super().__init__()
        u16_min = (0, 0)
        u16_max = (65536, 65536)  # 2^16

        layers = [Normalize(mean=u16_min, std=u16_max)]

        if laser_noise_config is not None:
            layers.append(
                RandomApply(
                    ModuleList([LaserNoise(**laser_noise_config.params)]),
                    laser_noise_config.prob))

        layers += [GetThirdChannel(mode=base_aug), MinMaxChop()]
        self.model = Compose(layers)

    def forward(self, x: Tensor):  # pylint: disable=missing-function-docstring
        return self.model(x)


class VisionBaseTransform(torch.nn.Module):
    """Base transformations for natural images."""

    def __init__(self, do_imnet_norm=False):
        super().__init__()

        imnet_norm_params = {
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225)
        }
        u8_norm_params = {"mean": (0, 0, 0), "std": (255, 255, 255)}

        layers = [Normalize(**u8_norm_params)]
        if do_imnet_norm:
            layers.append(Normalize(**imnet_norm_params))
        self.model = Compose(layers)

    def forward(self, x: torch.Tensor):  # pylint: disable=missing-function-docstring
        return self.model(x)


class NoBaseTransform(torch.nn.Module):
    """No base transformation used."""

    def __init__(self, do_normalize=True):  # pylint: disable=missing-function-docstring
        super().__init__()
        if do_normalize:
            u8_min = (0, 0, 0)
            u8_max = (255, 255, 255)  # 2^8
            self.model = Normalize(mean=u8_min, std=u8_max)
        else:
            self.model = lambda x: x.to(torch.uint8)

    def forward(self, x: torch.Tensor):  # pylint: disable=missing-function-docstring
        return self.model(x)


class Dinov2Normalization(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.normalize = Compose([
            ConvertImageDtype(dtype=float),
            make_normalize_transform(),
        ])

    def forward(self, x: torch.Tensor):
        return self.normalize(x).to(torch.float)


class StrongTransform(torch.nn.Module):
    """Strong transformations for all image data."""

    def __init__(self, aug_list: List[Dict[str, Any]], aug_prob: float):
        super().__init__()

        rand_apply_p = partial(self.rand_apply, prob=aug_prob)
        callable_dict = {
            "inpaint_rows_always_apply": InpaintRows,
            "inpaint_rows": partial(rand_apply_p, which=InpaintRows),
            "resize_always_apply": Resize,
            "dinov2_always_apply": DataAugmentationDINO,
            "resize": Resize,
            "normalize_always_apply": Normalize,
            "dinov2_normalize_always_apply": Dinov2Normalization,
            "random_horiz_flip": partial(RandomHorizontalFlip, p=aug_prob),
            "random_vert_flip": partial(RandomVerticalFlip, p=aug_prob),
            "gaussian_noise": partial(rand_apply_p, which=GaussianNoise),
            "color_jitter": partial(rand_apply_p, which=ColorJitter),
            "random_autocontrast": partial(RandomAutocontrast, p=aug_prob),
            "random_solarize": partial(RandomSolarize, p=aug_prob),
            "random_sharpness": partial(RandomAdjustSharpness, p=aug_prob),
            "drop_color": partial(rand_apply_p, which=Grayscale),
            "gaussian_blur": partial(rand_apply_p, GaussianBlur),
            "random_erasing": partial(RandomErasing, p=aug_prob),
            "random_affine": partial(rand_apply_p, RandomAffine),
            "random_crop": partial(RandomCrop),
            "random_resized_crop": partial(rand_apply_p, RandomResizedCrop),
            "random_grayscale": partial(RandomGrayscale, p=aug_prob),
            "fft_low_pass_filter": partial(rand_apply_p, FFTLowPassFilter),
            "fft_high_pass_filter": partial(rand_apply_p, FFTHighPassFilter),
            "fft_band_pass_filter": partial(rand_apply_p, FFTBandPassFilter),
        }

        self.transforms_ = Compose(
            [callable_dict[aug["which"]](**aug["params"]) for aug in aug_list])

    @staticmethod
    def rand_apply(which, prob, **kwargs):
        """Random apply an augmentation with probability"""
        return RandomApply(ModuleList([which(**kwargs)]), p=prob)

    def forward(self, x):  # pylint: disable=missing-function-docstring
        return self.transforms_(x)


# Base augmentation modules
class GetThirdChannel(torch.nn.Module):
    """computes the third channel of srh image

    compute the third channel of srh images by subtracting ch3 and ch2. the
    channel difference is added to the subtracted_base.

    """

    def __init__(self,
                 mode: str = "three_channels",
                 subtracted_base: float = 5000 / 65536.0):
        super().__init__()

        self.subtracted_base = subtracted_base
        aug_func_dict = {
            "three_channels": self.get_third_channel_,
            "ch2_only": self.get_ch2_,
            "ch3_only": self.get_ch3_,
            "diff_only": self.get_diff_
        }
        if mode in aug_func_dict:
            self.aug_func = aug_func_dict[mode]
        else:
            raise ValueError("base_augmentation must be in " +
                             f"{aug_func_dict.keys()}")

    def get_third_channel_(self, im2: Tensor) -> Tensor:
        ch2 = im2[0, :, :]
        ch3 = im2[1, :, :]
        ch1 = ch3 - ch2 + self.subtracted_base
        return torch.stack((ch1, ch2, ch3), dim=0)

    def get_ch2_(self, im2: Tensor) -> Tensor:
        return im2[0, :, :].unsqueeze(0)

    def get_ch3_(self, im2: Tensor) -> Tensor:
        return im2[1, :, :].unsqueeze(0)

    def get_diff_(self, im2: Tensor) -> Tensor:
        ch2 = im2[0, :, :]
        ch3 = im2[1, :, :]
        ch1 = ch3 - ch2 + self.subtracted_base

        return ch1.unsqueeze(0)

    def forward(self, two_channel_image: Tensor) -> Tensor:
        """
        args:
            two_channel_image: a 2 channel np array in the shape h * w * 2
            subtracted_base: an integer to be added to (ch3 - ch2)

        returns:
            a 1 or 3 channel tensor in the shape 1xhxw or 3xhxw
        """
        return self.aug_func(two_channel_image)


class MinMaxChop(torch.nn.Module):

    def __init__(self, min_val: float = 0.0, max_val: float = 1.0):
        super().__init__()
        self.min_ = min_val
        self.max_ = max_val

    def forward(self, image: Tensor) -> Tensor:
        return image.clamp(self.min_, self.max_)


# Strong augmentation modules
class InpaintRows(torch.nn.Module):

    def __init__(self, y_skip: int = 2, image_size: int = 300):
        self.y_skip = y_skip
        self.image_size = image_size

    def __call__(self, img):
        self.original_y = img.shape[1]
        mask = np.arange(0, self.original_y, self.y_skip)
        img_trans = img[:, mask, :]
        img_trans = Resize(size=(self.image_size, self.image_size),
                           interpolation=F.InterpolationMode.BILINEAR,
                           antialias=True)(img_trans)
        return img_trans

    def __repr__(self):
        return self.__class__.__name__ + '()'


class GaussianNoise(torch.nn.Module):
    """object to add guassian noise to images."""

    def __init__(self, min_var: float = 0.01, max_var: float = 0.1):
        super().__init__()
        self.min_var = min_var
        self.max_var = max_var

    def forward(self, tensor):
        var = random.uniform(self.min_var, self.max_var)
        noisy = tensor + torch.randn(tensor.size()) * var
        noisy = torch.clamp(noisy, min=0., max=1.)
        return noisy


class LaserNoise(torch.nn.Module):
    """object to add laser noise to images."""

    def __init__(self,
                 shot_noise_min_rate: float = 0.0,
                 shot_noise_max_rate: float = 0.2,
                 scatter_min_var: float = 0.0,
                 scatter_max_var: float = 1.0):
        super().__init__()
        self.shot_noise_min_rate = shot_noise_min_rate
        self.shot_noise_max_rate = shot_noise_max_rate
        self.scatter_min_var = scatter_min_var
        self.scatter_max_var = scatter_max_var

    def forward(self, img: Tensor):  # 2 channels
        # sample a sigma value for gaussian blur of noise
        sigma_val = random.uniform(2, 3)

        # additive shot noise
        var_shot = random.uniform(self.shot_noise_min_rate,
                                  self.shot_noise_max_rate)
        shot_noise = torch.randn(img.size()) * var_shot
        shot_noise = gaussian(shot_noise, sigma=sigma_val, channel_axis=None)

        # multiplicative scatter noise
        var_mul = random.uniform(self.scatter_min_var, self.scatter_max_var)
        scatter_noise = torch.randn(img.size()) * var_mul
        # scatter_noise = gaussian(scatter_noise, sigma=sigma_val, multichannel=false)

        # apply noise to image
        noisy = (img + shot_noise) + (img * scatter_noise)
        # noisy = torch.clamp(noisy, min=0., max=1.)

        return noisy


class FFTLowPassFilter(torch.nn.Module):

    def __init__(self, circ_radius: List[int] = [50, 300]):
        super().__init__()
        self.circ_radius_ = circ_radius

    def circ_mask(self, im_size: torch.Size()):
        im_size = np.array(im_size)
        r = torch.linspace(0, im_size[-2] - 1, steps=im_size[-2])
        c = torch.linspace(0, im_size[-1] - 1, steps=im_size[-1])
        center = im_size[-2:] // 2
        R, C = torch.meshgrid(r, c, indexing="ij")
        R_diff = R - center[0]
        C_diff = C - center[1]
        radius = random.randint(self.circ_radius_[0], self.circ_radius_[1])
        return (R_diff * R_diff + C_diff * C_diff) < (radius * radius)

    def forward(self, img: Tensor):  # 3 channels
        fft_img = fftshift(fft2(img))
        fft_img = fft_img * self.circ_mask(img.shape)

        # invert the fft to get the reconstructed images
        freq_filt_img = ifft2(ifftshift(fft_img))
        freq_filt_img = torch.abs(freq_filt_img)
        #freq_filt_img /= freq_filt_img.max()
        return freq_filt_img


class FFTHighPassFilter(torch.nn.Module):

    def __init__(self, circ_radius: List[int] = [50, 300]):
        super().__init__()
        self.circ_radius_ = circ_radius

    def circ_mask(self, im_size: torch.Size()):
        im_size = np.array(im_size)
        r = torch.linspace(0, im_size[-2] - 1, steps=im_size[-2])
        c = torch.linspace(0, im_size[-1] - 1, steps=im_size[-1])
        center = im_size[-2:] // 2
        R, C = torch.meshgrid(r, c, indexing="ij")
        R_diff = R - center[0]
        C_diff = C - center[1]
        radius = random.randint(self.circ_radius_[0], self.circ_radius_[1])
        return (R_diff * R_diff + C_diff * C_diff) >= (radius * radius)

    def forward(self, img: Tensor):  # 3 channels
        fft_img = fftshift(fft2(img))
        fft_img = fft_img * self.circ_mask(img.shape)

        # invert the fft to get the reconstructed images
        freq_filt_img = ifft2(ifftshift(fft_img))
        freq_filt_img = torch.abs(freq_filt_img)
        #freq_filt_img /= freq_filt_img.max()
        return freq_filt_img


class FFTBandPassFilter(torch.nn.Module):

    def __init__(self,
                 lower_radius: List[int] = [50, 100],
                 higher_radius: List[int] = [150, 300]):
        super().__init__()
        self.lower_radius_ = lower_radius
        self.higher_radius_ = higher_radius

    def circ_mask(self, im_size: torch.Size()):
        im_size = np.array(im_size)
        r = torch.linspace(0, im_size[-2] - 1, steps=im_size[-2])
        c = torch.linspace(0, im_size[-1] - 1, steps=im_size[-1])

        center = im_size[-2:] // 2
        R, C = torch.meshgrid(r, c, indexing="ij")
        R_diff = R - center[0]
        C_diff = C - center[1]
        radius_mat = (R_diff * R_diff + C_diff * C_diff)

        low_radius = random.randint(self.lower_radius_[0],
                                    self.lower_radius_[1])
        high_radius = random.randint(self.higher_radius_[0],
                                     self.higher_radius_[1])

        hpf_mask = radius_mat >= (low_radius * low_radius)
        lpf_mask = radius_mat < (high_radius * high_radius)
        return lpf_mask & hpf_mask

    def forward(self, img: Tensor):  # 3 channels
        fft_img = fftshift(fft2(img))
        fft_img = fft_img * self.circ_mask(img.shape)

        # invert the fft to get the reconstructed images
        freq_filt_img = ifft2(ifftshift(fft_img))
        freq_filt_img = torch.abs(freq_filt_img)
        #freq_filt_img /= freq_filt_img.max()
        return freq_filt_img


# Augmentations including masks (for ground truth or other supervision)
class RandomHorizFlipWithMask(torch.nn.Module):

    def __init__(self, p=0.5):
        super().__init__()
        self.p_ = p

    def forward(self, im: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        if torch.rand(1).item() < self.p_:
            return (F.hflip(im[0]), F.hflip(im[1]))
        return im


class RandomVertFlipWithMask(torch.nn.Module):

    def __init__(self, p=0.5):
        super().__init__()
        self.p_ = p

    def forward(self, im: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        if torch.rand(1).item() < self.p_:
            return (F.vflip(im[0]), F.vflip(im[1]))
        return im


class RandAugImWithMask(torch.nn.Module):

    def __init__(self, aug: Callable, p: float = 0.5, **kwargs):
        super().__init__()
        self.aug_ = aug(**kwargs)
        self.p_ = p

    def forward(self, im: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        if torch.rand(1).item() < self.p_:
            return (self.aug_.forward(im[0]), im[1])
        else:
            return im

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}({self.aug_.__class__.__name__}, " +
                f"p={self.p_})")


class Autocontrast(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, img: Tensor) -> Tensor:
        return F.autocontrast(img)

    def __repr__(self) -> str:
        return self.__class__.__name__


class Solarize(torch.nn.Module):

    def __init__(self, threshold: float):
        super().__init__()
        self.threshold_ = threshold

    def forward(self, img: Tensor) -> Tensor:
        return F.solarize(img, self.threshold_)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(threshold={self.threshold})"


class AdjustSharpness(torch.nn.Module):

    def __init__(self, sharpness_factor: float):
        super().__init__()
        self.sharpness_factor_ = sharpness_factor

    def forward(self, img: Tensor) -> Tensor:
        return F.adjust_sharpness(img, self.sharpness_factor_)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}" +
                f"(sharpness_factor={self.sharpness_factor_})")
