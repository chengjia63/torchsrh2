"""Utilities for preparing SRH tensors for visualization.

This module centralizes the small image helpers that are currently duplicated
in visualization code. The expected input is a torch tensor in ``C x H x W``
layout with values in ``[0, 1]``.
"""

import einops
import torch
from torchvision.transforms.functional import adjust_brightness, adjust_contrast


def get_third_channel(
    x: torch.Tensor,
    subtracted_base: float = 5000 / 65536.0,
) -> torch.Tensor:
    """Build a 3-channel SRH visualization tensor from a 2-channel input."""

    ch2 = x[0]
    ch3 = x[1]
    ch1 = ch3 - ch2 + subtracted_base
    return torch.stack((ch1, ch2, ch3), dim=0)


def normalize_viz_image(x: torch.Tensor) -> torch.Tensor:
    """Scale an SRH tensor for display as an 8-bit image.

    The transform matches the existing visualization behavior used elsewhere in
    the codebase: apply a fixed contrast boost, then a fixed brightness boost,
    then convert the result to ``uint8`` in the ``[0, 255]`` range.
    """

    return (adjust_brightness(adjust_contrast(x, 2), 2) * 255).to(torch.uint8)


def rearrange_viz_image(x: torch.Tensor) -> torch.Tensor:
    """Convert a channel-first image tensor from ``C x H x W`` to ``H x W x C``."""

    return einops.rearrange(x, "c h w -> h w c")


def prepare_three_channel_viz_image(x: torch.Tensor) -> torch.Tensor:
    """Normalize an SRH tensor and convert it to channel-last layout.

    This is the common visualization path when a model or transform produces a
    ``C x H x W`` tensor and downstream code needs an image-like ``H x W x C``
    tensor for plotting, PIL conversion, or serialization.
    """

    return rearrange_viz_image(normalize_viz_image(x))


def prepare_two_channel_viz_image(
    x: torch.Tensor,
    subtracted_base: float = 5000 / 65536.0,
) -> torch.Tensor:
    """Prepare a raw 16-bit two-channel HWC SRH patch for visualization.

    This mirrors the base SRH preprocessing path used in
    ``ts2.data.transforms.SRHBaseTransform``:
    normalize by ``2^16``, construct the third channel, clamp to ``[0, 1]``,
    then apply the visualization brightness/contrast conversion.
    """

    x = einops.rearrange(x, "h w c -> c h w").float() / 65536.0
    x = get_third_channel(x, subtracted_base=subtracted_base).clamp(0, 1)

    return prepare_three_channel_viz_image(x)
