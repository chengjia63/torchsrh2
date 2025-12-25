import numpy as np
from PIL import Image
from tifffile import imread
from typing import Optional
import torch
import torchvision
from torchvision.transforms.functional import to_tensor
import einops
import logging
import time

def instantiate_process_read(which: str, which_set: Optional[str] = "srh"):
    """Returns the proper process read function"""
    if which == "memmap":
        return MemmapReader(which_set=which_set)
    elif which == "memmap_tile":
        return MemmapTileReader(which_set=which_set)
    elif which == "memmap_multi":
        return MemmapMultiReader(which_set=which_set)
    elif which == "memmap_multi_fm":
        return MemmapMultiReaderWithFoundation(which_set=which_set)
    elif which == "cell_memmap":
        return CellMemmapReader(which_set=which_set)

    return {
        "srh": process_read_srh,
        "png": process_read_png,
    }[which]


class CellMemmapReader():

    def __init__(self, which_set):
        dtype_map = {"scsrh": "uint16", "srh": "uint16"}
        self.dtype_ = dtype_map.get(which_set, "uint8")

    def __call__(self, mm_path, tensor_shape, patch_idx):
        """Read in two channel image

        Returns:
            A 2 channel torch Tensor in the shape 2 * H * W
        """

        #start = time.time()
        fd = np.memmap(mm_path,
                       dtype=self.dtype_,
                       mode="r",
                       shape=tensor_shape)
        #t1 = time.time()
        im = np.array(fd[patch_idx, ...])
        fd._mmap.close()
        del fd
        #t2 = time.time()
        data = torch.from_numpy(im).to(torch.float32).contiguous()
        #t3 = time.time()
        return data#, (t1-start,t2-start,t3-start)


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


class MemmapTileReader(MemmapReader):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, mm_path, tensor_shape, patch_idx, rc_idx, tile_size):
        """Read in two channel image

        Returns:
            A 2 channel torch Tensor in the shape 2 * H * W
        """
        fd = np.memmap(mm_path,
                       dtype=self.dtype_,
                       mode="r",
                       shape=tensor_shape)
        im = np.array(fd[patch_idx, 
                         rc_idx[0]:rc_idx[0]+tile_size,
                         rc_idx[1]:rc_idx[1]+tile_size, 
                         :])
        fd._mmap.close()
        del fd
        return einops.rearrange(
            torch.from_numpy(im).to(torch.float32),
            "b h w c -> b c h w").contiguous()
    

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

class MemmapMultiReaderWithFoundation(MemmapMultiReader):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_emb(self, mm_path, tensor_shape, patch_idx):
        """Read in two channel image

        Returns:
            A 2 channel torch Tensor in the shape 2 * H * W
        """
        fd = np.memmap(mm_path, dtype="float32",
                       mode="r").reshape(tensor_shape)
        im = np.array(fd[patch_idx, ...])
        fd._mmap.close()
        del fd
        return torch.from_numpy(im).to(torch.float32).contiguous()

    def __call__(self, mm_path, tensor_shape, patch_idx):
        """Read in two channel image

        Returns:
            A 2 channel torch Tensor in the shape 2 * H * W
        """
        im = super().__call__(mm_path=mm_path[0],
                              tensor_shape=tensor_shape[0],
                              patch_idx=patch_idx)
        embs = [
            self.get_emb(i, tensor_shape[1], patch_idx) for i in mm_path[1]
        ]
        return im, embs


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
