"""Minimal image processing helpers for standalone strip patching."""

import cv2
import numpy as np


def field_flatten(
    array: np.ndarray,
    blur_type: str = "gaussian",
    kernel_size: int = 301,
) -> np.ndarray:
    if blur_type.lower() == "gaussian":
        filter_img = cv2.GaussianBlur(
            array,
            ksize=(kernel_size, kernel_size),
            sigmaX=kernel_size,
        )
    elif blur_type.lower() == "average":
        filter_img = cv2.blur(array, (kernel_size, kernel_size))
    else:
        raise ValueError("blur_type must be 'gaussian' or 'average'.")

    flat_image = array / filter_img
    flat_image *= 10000
    return flat_image


def get_16bit_patch(patch: np.ndarray, flatten_field: bool = True) -> np.ndarray:
    patch = patch.astype(float)
    ch2 = patch[:, :, 0]
    ch3 = patch[:, :, 1]

    if flatten_field:
        ch2 = field_flatten(ch2)
        ch3 = field_flatten(ch3)

    stack = np.zeros((ch2.shape[0], ch2.shape[1], 2), dtype=float)
    stack[:, :, 0] = ch2
    stack[:, :, 1] = ch3
    return stack.clip(min=0, max=65535)


def percentile_rescaling(array: np.ndarray, percentile_clip: int = 3) -> np.ndarray:
    p_low, p_high = np.percentile(array, (3, 100 - percentile_clip))
    array = array.clip(min=p_low, max=p_high)
    return (array - p_low) / (p_high - p_low)


def _srh_8bit_preprocess(
    patch: np.ndarray,
    percentile_clip: int = 3,
    subtracted_channel_recenter: float = 0.2,
) -> np.ndarray:
    patch = patch.astype(float)
    ch2 = percentile_rescaling(patch[:, :, 0], percentile_clip)
    ch3 = percentile_rescaling(patch[:, :, 1], percentile_clip)
    subtracted_array = np.subtract(ch3, ch2).clip(min=0, max=1)

    stack = np.zeros((ch2.shape[0], ch2.shape[1], 3), dtype=float)
    stack[:, :, 0] = subtracted_array + subtracted_channel_recenter
    stack[:, :, 1] = ch2
    stack[:, :, 2] = ch3
    stack *= 255.0
    return stack


def blood_check(
    image: np.ndarray,
    upper_percentile: int = 20,
    intensity_threshold: int = 60,
) -> np.ndarray:
    two_channel = image[:, :, 1:3]
    for percentile in range(3, upper_percentile):
        post_image = _srh_8bit_preprocess(two_channel, percentile)
        if post_image.mean() > intensity_threshold:
            return post_image
    return image


def get_three_channel_8bit_patch(
    patch: np.ndarray,
    percentile_clip: int = 3,
    subtracted_channel_recenter: float = 0.2,
    check_blood: bool = True,
    intensity_threshold: int = 60,
) -> np.ndarray:
    stack = _srh_8bit_preprocess(
        patch,
        percentile_clip,
        subtracted_channel_recenter,
    )
    if check_blood:
        stack = blood_check(stack, percentile_clip, intensity_threshold)
    return stack.clip(min=0, max=255)
