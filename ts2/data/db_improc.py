import numpy as np
from PIL import Image
from tifffile import imread

import torch
import torchvision
from torchvision.transforms.functional import to_tensor

def instantiate_process_read(which: str):
    """Returns the proper process read function"""
    return {"srh": process_read_srh, "he": process_read_png}[which]


def process_read_srh(imp: str) -> torch.Tensor:
    """Read in two channel image

    Args:
        imp: a string that is the path to the tiff image

    Returns:
        A 2 channel torch Tensor in the shape 2 * H * W
    """

    # reference: https://github.com/pytorch/vision/blob/49468279/torchvision/transforms/functional.py#L133
    return torch.from_numpy(imread(imp).astype(np.float32)).contiguous()


def process_read_png(imp: str) -> torch.Tensor:
    """Read in 3 channel image

    Args:
        imp: a string that is the path to the tiff image

    Returns:
        A TODO
    """
    # https://pytorch.org/vision/main/generated/torchvision.transforms.ToTensor.html
    return to_tensor(Image.open(imp))[0:3,...]
