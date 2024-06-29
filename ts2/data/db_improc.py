import numpy as np
from PIL import Image
from tifffile import imread
from typing import Optional
import torch
import torchvision
from torchvision.transforms.functional import to_tensor
import einops
import logging


def instantiate_process_read(which: str, which_set: Optional[str] = "srh"):
    """Returns the proper process read function"""
    if which == "memmap":
        return MemmapReader(which_set=which_set)
    elif which == "memmap_multi":
        return MemmapMultiReader(which_set=which_set)

    return {
        "srh": process_read_srh,
        "png": process_read_png,
    }[which]


class MemmapReader():

    def __init__(self, which_set):
        dtype_map = {"srh": "uint16"}
        self.dtype_ = dtype_map.get(which_set, "uint8")

    def __call__(self, mm_path, tensor_shape, patch_idx):
        """Read in two channel image

        Returns:
            A 2 channel torch Tensor in the shape 2 * H * W
        """
        fd = np.memmap(mm_path,
                       dtype=self.dtype_,
                       mode="r",
                       shape=tensor_shape)
        im = np.array(fd[patch_idx, ...])
        fd._mmap.close()
        del fd
        return einops.rearrange(
            torch.from_numpy(im).to(torch.float32),
            "h w c -> c h w").contiguous()


class MemmapMultiReader(MemmapReader):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, mm_path, tensor_shape, patch_idx):
        """Read in two channel image

        Returns:
            A 2 channel torch Tensor in the shape 2 * H * W
        """
        fd = np.memmap(mm_path,
                       dtype=self.dtype_,
                       mode="r",
                       shape=tensor_shape)
        im = np.array(fd[patch_idx, ...])
        fd._mmap.close()
        del fd
        return einops.rearrange(
            torch.from_numpy(im).to(torch.float32),
            "b h w c -> b c h w").contiguous()


def process_read_srh(imp: str) -> torch.Tensor:
    """Read in two channel image

    Args:
        imp: a string that is the path to the tiff image

    Returns:
        A 2 channel torch Tensor in the shape 2 * H * W
    """

    return torch.from_numpy(imread(imp).astype(np.float32)).contiguous()


def process_read_png(imp: str) -> torch.Tensor:
    """Read in 3 channel image

    Args:
        imp: a string that is the path to the tiff image

    Returns:
        A TODO
    """
    # https://pytorch.org/vision/main/generated/torchvision.transforms.ToTensor.html
    return to_tensor(Image.open(imp))[0:3, ...]
