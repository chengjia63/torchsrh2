"""Helpers for extracting aligned patches from paired SRH strip DICOMs.

This module operates on paired CH2 and CH3 strip images and produces a
dictionary of patches keyed by their global strip coordinates. The patching
logic is designed to support long strip images where direct full-image
registration may be impractical, so the strip is first divided into larger
overlapping substrips and patch starts are then recovered from those local
windows while remaining aligned to the requested global patch grid.
"""

import os
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pydicom as pyd
import torch
from PIL import Image
from tqdm import tqdm

from ts2.utils.strip_patching.image_utils import (
    get_16bit_patch,
    get_three_channel_8bit_patch,
)
from ts2.utils.strip_patching.registration import fft_register
from ts2.utils.srh_viz import prepare_two_channel_viz_image

IntOrTuple = Union[int, Tuple[int, int]]
PatchProcessor = Callable[[np.ndarray], np.ndarray]
PatchProcessorArg = Optional[Union[str, PatchProcessor]]


def _as_pair(value: IntOrTuple, name: str) -> Tuple[int, int]:
    """Normalizes a scalar-or-pair argument into a validated integer pair.

    Args:
        value: Integer or length-2 iterable to normalize.
        name: Argument name used in validation error messages.

    Returns:
        A validated ``(y, x)`` integer tuple.

    Raises:
        ValueError: If ``value`` is not length 2 or contains non-positive
            entries.
    """

    if isinstance(value, int):
        pair = (value, value)
    else:
        pair = tuple(value)
    if len(pair) != 2:
        raise ValueError(f"{name} must be an int or a 2-item tuple.")
    if pair[0] <= 0 or pair[1] <= 0:
        raise ValueError(f"{name} values must be positive.")
    return int(pair[0]), int(pair[1])


def _as_offset(value: IntOrTuple, name: str) -> Tuple[int, int]:
    """Normalizes a scalar-or-pair offset into a validated non-negative pair.

    Args:
        value: Integer or length-2 iterable to normalize.
        name: Argument name used in validation error messages.

    Returns:
        A validated ``(y, x)`` integer tuple.

    Raises:
        ValueError: If ``value`` is not length 2 or contains negative entries.
    """

    if isinstance(value, int):
        pair = (value, value)
    else:
        pair = tuple(value)
    if len(pair) != 2:
        raise ValueError(f"{name} must be an int or a 2-item tuple.")
    if pair[0] < 0 or pair[1] < 0:
        raise ValueError(f"{name} values must be non-negative.")
    return int(pair[0]), int(pair[1])


def _window_starts(
    side_length: int,
    window_size: int,
    stride: int,
    start: int,
) -> Tuple[int, ...]:
    """Enumerates 1D window start positions.

    The returned starts follow ``start, start + stride, ...`` and stop before a
    window would extend past ``side_length``.

    Args:
        side_length: Full axis length.
        window_size: Size of the sliding window.
        stride: Step size between windows.
        start: Initial start position.

    Returns:
        A tuple of valid window start indices.
    """

    if start + window_size > side_length:
        return tuple()

    starts = []
    position = start
    while position + window_size <= side_length:
        starts.append(position)
        position += stride
    return tuple(starts)


def _substrip_starts(
    side_length: int,
    substrip_size: int,
    substrip_stride: int,
    substrip_start: int,
) -> Tuple[int, ...]:
    """Returns substrip origins for a single axis, including a tail window.

    The tail window ensures the final extent of a strip is still covered even
    when the main stride sequence does not land exactly on the end.

    Args:
        side_length: Full axis length.
        substrip_size: Size of each substrip.
        substrip_stride: Step size between substrips.
        substrip_start: Initial substrip origin.

    Returns:
        A tuple of substrip start indices.
    """

    starts = list(
        _window_starts(side_length, substrip_size, substrip_stride, substrip_start)
    )
    if not starts:
        return tuple()

    tail_start = side_length - substrip_size
    if tail_start >= 0 and starts[-1] != tail_start:
        starts.append(tail_start)
    return tuple(dict.fromkeys(starts))


def _derived_substrip_layout(
    patch_size: int,
    patch_stride: int,
    patch_start: int,
    substrip_size: int,
) -> Tuple[int, int]:
    """Derives a substrip start and stride aligned to the patch grid.

    ``substrip_size`` is larger than ``patch_size``. This helper chooses a
    substrip stride such that consecutive substrips expose consecutive groups of
    valid patch origins without skipping patch rows or columns. That alignment
    is important for long strips, for example to avoid missing starts like
    ``900`` or ``1800`` in a 300-stride patch grid.

    Args:
        patch_size: Patch size along one axis.
        patch_stride: Patch stride along one axis.
        patch_start: First global patch origin along one axis.
        substrip_size: Substrip size along one axis.

    Returns:
        A ``(substrip_start, substrip_stride)`` tuple.

    Raises:
        ValueError: If ``patch_size`` is larger than ``substrip_size``.
    """

    max_local_patch_start = substrip_size - patch_size
    if max_local_patch_start < 0:
        raise ValueError("patch_size cannot be larger than substrip_size.")

    num_patch_steps = max_local_patch_start // patch_stride
    substrip_start = max(0, patch_start - (num_patch_steps * patch_stride))
    substrip_stride = (num_patch_steps + 1) * patch_stride
    return substrip_start, substrip_stride


def _aligned_patch_starts(
    window_origin: int,
    window_size: int,
    patch_size: int,
    patch_stride: int,
    patch_start: int,
) -> Tuple[int, ...]:
    """Computes patch starts inside a substrip while preserving global alignment.

    The returned values are local coordinates inside the current substrip, but
    they correspond to the global patch grid defined by ``patch_start`` and
    ``patch_stride``.

    Args:
        window_origin: Global origin of the current substrip.
        window_size: Size of the current substrip along one axis.
        patch_size: Patch size along one axis.
        patch_stride: Patch stride along one axis.
        patch_start: First global patch origin along one axis.

    Returns:
        A tuple of local patch start indices within the substrip.
    """

    max_global_start = window_origin + window_size - patch_size
    if max_global_start < window_origin:
        return tuple()

    if patch_start >= window_origin:
        first_global_start = patch_start
    else:
        delta = window_origin - patch_start
        steps = (delta + patch_stride - 1) // patch_stride
        first_global_start = patch_start + (steps * patch_stride)

    starts = []
    global_start = first_global_start
    while global_start <= max_global_start:
        starts.append(global_start - window_origin)
        global_start += patch_stride
    return tuple(starts)


def _process_patch(
    patch: np.ndarray,
    patch_processor: PatchProcessorArg = "16bit",
) -> np.ndarray:
    """Applies optional post-processing to an extracted patch.

    Args:
        patch: Extracted patch in ``H x W x C`` layout.
        patch_processor: Patch processor specification. ``"16bit"`` applies
            :func:`get_16bit_patch`, ``"8bit"`` applies
            :func:`get_three_channel_8bit_patch`, ``None`` returns the raw
            patch, and a callable is applied directly.

    Returns:
        The processed patch array.

    Raises:
        ValueError: If ``patch_processor`` is a string other than ``"16bit"``
            or ``"8bit"``.
    """

    if patch_processor is None:
        return patch
    if isinstance(patch_processor, str):
        mode = patch_processor.lower()
        if mode == "16bit":
            return get_16bit_patch(patch, flatten_field=True)
        if mode == "8bit":
            return get_three_channel_8bit_patch(patch, flatten_field=True)
        raise ValueError(
            "patch_processor must be one of: None, '16bit', '8bit', or a callable."
        )
    return patch_processor(patch)


def generate_paired_strip_patches(
    ch2_dicom_path: str,
    ch3_dicom_path: str,
    patch_size: IntOrTuple = 300,
    patch_stride: IntOrTuple = 300,
    patch_start: IntOrTuple = (0, 50),
    coordinate_offset: IntOrTuple = (0, 0),
    substrip_size: IntOrTuple = 1000,
    register: bool = True,
    patch_processor: PatchProcessorArg = "16bit",
    series_name: str = None,
) -> Dict[str, np.ndarray]:
    """Generates aligned patches from paired CH2 and CH3 strip DICOMs.

    Args:
        ch2_dicom_path: Path to the CH2 strip DICOM.
        ch3_dicom_path: Path to the CH3 strip DICOM.
        patch_size: Patch size as an integer or ``(height, width)`` pair.
        patch_stride: Patch stride as an integer or ``(y_stride, x_stride)``
            pair.
        patch_start: Patch-grid origin within the current strip as an integer
            or ``(y, x)`` pair.
        coordinate_offset: Two-dimensional offset added to the final output
            patch coordinates, given as an integer or ``(y, x)`` pair. This is
            useful when multiple strips need to be placed into a shared global
            coordinate system.
        substrip_size: Size of the registration or extraction window used
            before taking patches. This may be larger than ``patch_size``.
        register: If ``True``, registers CH3 to CH2 within each substrip before
            extracting patches.
        patch_processor: ``"16bit"``, ``"8bit"``, ``None``, or a callable
            patch processor.
        series_name: Optional prefix for patch names. If omitted, the literal
            word ``"patch"`` is used as the prefix.

    Returns:
        A dictionary mapping patch names to numpy arrays. Names are formatted as
        ``"{prefix}-{y}_{x}"`` where ``prefix`` is either ``series_name`` or
        ``"patch"``, and ``y`` and ``x`` are global strip coordinates.

    Raises:
        ValueError: If the input strip shapes do not match, the strips are not
            2D, the requested patch size exceeds the substrip size, or no valid
            substrip layout can cover the requested patch grid.
    """

    patch_height, patch_width = _as_pair(patch_size, "patch_size")
    patch_stride_y, patch_stride_x = _as_pair(patch_stride, "patch_stride")
    patch_start_y, patch_start_x = _as_offset(patch_start, "patch_start")
    coordinate_offset_y, coordinate_offset_x = _as_offset(
        coordinate_offset,
        "coordinate_offset",
    )
    substrip_height, substrip_width = _as_pair(substrip_size, "substrip_size")
    substrip_start_y, substrip_stride_y = _derived_substrip_layout(
        patch_height, patch_stride_y, patch_start_y, substrip_height
    )
    substrip_start_x, substrip_stride_x = _derived_substrip_layout(
        patch_width, patch_stride_x, patch_start_x, substrip_width
    )

    ch2_strip = pyd.dcmread(ch2_dicom_path).pixel_array.astype(float)
    ch3_strip = pyd.dcmread(ch3_dicom_path).pixel_array.astype(float)

    if ch2_strip.shape != ch3_strip.shape:
        raise ValueError("Paired CH2/CH3 strips must have the same dimensions.")
    if ch2_strip.ndim != 2:
        raise ValueError("Expected single-frame 2D strip DICOM inputs.")
    if patch_height > substrip_height or patch_width > substrip_width:
        raise ValueError("patch_size cannot be larger than substrip_size.")

    strip_height, strip_width = ch2_strip.shape
    substrip_starts_y = _substrip_starts(
        strip_height, substrip_height, substrip_stride_y, substrip_start_y
    )
    substrip_starts_x = _substrip_starts(
        strip_width, substrip_width, substrip_stride_x, substrip_start_x
    )

    if not substrip_starts_y or not substrip_starts_x:
        raise ValueError(
            "No valid substrips fit in the input strips with the requested "
            "substrip_size and derived substrip layout."
        )

    patch_prefix = series_name or "patch"
    patch_dict: Dict[str, np.ndarray] = {}
    seen_patch_origins = set()

    for substrip_y in substrip_starts_y:
        for substrip_x in substrip_starts_x:
            ch2_substrip = ch2_strip[
                substrip_y : substrip_y + substrip_height,
                substrip_x : substrip_x + substrip_width,
            ]
            ch3_substrip = ch3_strip[
                substrip_y : substrip_y + substrip_height,
                substrip_x : substrip_x + substrip_width,
            ]

            if register:
                registered_substrip = fft_register(ch2_substrip, ch3_substrip)
            else:
                registered_substrip = np.stack((ch2_substrip, ch3_substrip), axis=-1)

            patch_starts_y = _aligned_patch_starts(
                substrip_y,
                substrip_height,
                patch_height,
                patch_stride_y,
                patch_start_y,
            )
            patch_starts_x = _aligned_patch_starts(
                substrip_x,
                substrip_width,
                patch_width,
                patch_stride_x,
                patch_start_x,
            )

            for local_y in patch_starts_y:
                for local_x in patch_starts_x:
                    patch = registered_substrip[
                        local_y : local_y + patch_height,
                        local_x : local_x + patch_width,
                        :,
                    ]
                    patch = _process_patch(
                        patch,
                        patch_processor=patch_processor,
                    )

                    global_y = substrip_y + local_y + coordinate_offset_y
                    global_x = substrip_x + local_x + coordinate_offset_x
                    patch_origin = (global_y, global_x)
                    if patch_origin in seen_patch_origins:
                        continue

                    seen_patch_origins.add(patch_origin)
                    patch_name = f"{patch_prefix}-{global_y}_{global_x}"
                    patch_dict[patch_name] = patch

    return patch_dict


def generate_paired_strip_patches_from_lists(
    ch2_dicom_paths: Sequence[str],
    ch3_dicom_paths: Sequence[str],
    strip_offset_interval: int = 900,
    **kwargs,
) -> Dict[str, np.ndarray]:
    """Generates one merged patch dictionary from paired CH2 and CH3 strip lists.

    Each CH2 and CH3 path at the same list index is treated as one strip pair.
    The helper calls :func:`generate_paired_strip_patches` for each pair and
    merges all patch dictionaries into a single output dictionary. Patch names
    are prefixed with ``"strip_{i}"`` where ``i`` is the zero-based strip
    index.

    Args:
        ch2_dicom_paths: Ordered CH2 strip DICOM paths.
        ch3_dicom_paths: Ordered CH3 strip DICOM paths.
        strip_offset_interval: Column offset interval between consecutive
            strips in the merged coordinate system.
        **kwargs: Keyword arguments forwarded to
            :func:`generate_paired_strip_patches` for every strip pair.

    Returns:
        One merged patch dictionary for all strip pairs.

    Raises:
        ValueError: If the CH2 and CH3 lists do not have the same length.
    """

    if len(ch2_dicom_paths) != len(ch3_dicom_paths):
        raise ValueError(
            "ch2_dicom_paths and ch3_dicom_paths must have the same length."
        )

    patch_dict: Dict[str, np.ndarray] = {}
    for strip_index, (ch2_dicom_path, ch3_dicom_path) in enumerate(
        tqdm(
            zip(ch2_dicom_paths, ch3_dicom_paths),
            total=len(ch2_dicom_paths),
            desc="Generating strip patches",
        )
    ):
        strip_patch_dict = generate_paired_strip_patches(
            ch2_dicom_path=ch2_dicom_path,
            ch3_dicom_path=ch3_dicom_path,
            coordinate_offset=(0, strip_index * strip_offset_interval),
            series_name=f"strip_{strip_index}",
            **kwargs,
        )
        patch_dict.update(strip_patch_dict)

    return patch_dict


def save_stitched_patch_visualization(
    patch_dict: Dict[str, np.ndarray],
    output_path: str,
) -> np.ndarray:
    """Builds and saves a stitched visualization image from a patch dictionary.

    Args:
        patch_dict: Dictionary produced by
            :func:`generate_paired_strip_patches`. Keys must end with a
            coordinate suffix formatted as ``"{y}_{x}"`` after the final
            ``"-"``.
        output_path: Destination path for the stitched image.

    Returns:
        The stitched ``H x W x 3`` uint8 visualization image that was saved.
    """

    patch_items = []
    max_bottom = 0
    max_right = 0

    for patch_name, patch_array in tqdm(
        patch_dict.items(),
        total=len(patch_dict),
        desc="Preparing stitched patches",
    ):
        coord_str = patch_name.rsplit("-", 1)[1]
        y_str, x_str = coord_str.split("_", 1)
        top = int(y_str)
        left = int(x_str)
        viz_patch = prepare_two_channel_viz_image(torch.as_tensor(patch_array).float())
        patch_height, patch_width, _ = viz_patch.shape

        patch_items.append((top, left, viz_patch.numpy().astype(np.float32)))
        max_bottom = max(max_bottom, top + patch_height)
        max_right = max(max_right, left + patch_width)

    stitched_sum = np.zeros((max_bottom, max_right, 3), dtype=np.float32)
    stitched_count = np.zeros((max_bottom, max_right, 1), dtype=np.float32)

    for top, left, patch_np in tqdm(
        patch_items,
        total=len(patch_items),
        desc="Stitching patches",
    ):
        patch_height, patch_width, _ = patch_np.shape
        stitched_sum[top : top + patch_height, left : left + patch_width] += patch_np
        stitched_count[top : top + patch_height, left : left + patch_width] += 1

    stitched_image = np.divide(
        stitched_sum,
        np.maximum(stitched_count, 1),
    ).astype(np.uint8)
    Image.fromarray(stitched_image).save(output_path)
    return stitched_image


def save_tiled_patch_visualization(
    patch_dict: Dict[str, np.ndarray],
    output_path: str,
) -> np.ndarray:
    """Builds and saves a tiled visualization image from a patch dictionary.

    This version ignores the coordinate suffix in patch names and instead tiles
    patches into a compact grid using the coordinates encoded in the patch
    names to determine row-major order.

    Args:
        patch_dict: Dictionary produced by a strip patch generation helper.
        output_path: Destination path for the tiled image.

    Returns:
        The tiled ``H x W x 3`` uint8 visualization image that was saved.

    Raises:
        ValueError: If ``patch_dict`` is empty.
    """

    if not patch_dict:
        raise ValueError("patch_dict must not be empty.")

    patch_rows = {}
    for patch_name, patch_array in tqdm(
        patch_dict.items(),
        total=len(patch_dict),
        desc="Preparing tiled patches",
    ):
        coord_str = patch_name.rsplit("-", 1)[1]
        y_str, x_str = coord_str.split("_", 1)
        y_coord = int(y_str)
        x_coord = int(x_str)
        viz_patch = prepare_two_channel_viz_image(torch.as_tensor(patch_array).float())
        patch_rows.setdefault(y_coord, []).append((x_coord, viz_patch.numpy()))

    ordered_rows = []
    max_columns = 0
    for y_coord in sorted(patch_rows):
        row_patches = [patch for _, patch in sorted(patch_rows[y_coord], key=lambda item: item[0])]
        ordered_rows.append(row_patches)
        max_columns = max(max_columns, len(row_patches))

    patch_height, patch_width, num_channels = ordered_rows[0][0].shape
    num_rows = len(ordered_rows)
    tiled_image = np.zeros(
        (num_rows * patch_height, max_columns * patch_width, num_channels),
        dtype=np.uint8,
    )

    for row_index, row_patches in enumerate(
        tqdm(ordered_rows, total=len(ordered_rows), desc="Tiling patches")
    ):
        top = row_index * patch_height
        for col_index, patch_np in enumerate(row_patches):
            left = col_index * patch_width
            tiled_image[top : top + patch_height, left : left + patch_width] = patch_np

    Image.fromarray(tiled_image).save(output_path)
    return tiled_image


def main() -> None:
    """Generates demo patches and saves one stitched visualization image.

    This entry point is a local debugging helper. It runs patch extraction on a
    fixed list of CH2 and CH3 strips and writes the stitched visualization to
    ``stitched.png``.
    """

    ch2_dicom_paths = [
        "/nfs/turbo/umms-tocho/data/db_nio_mouse/mouse/nio_mouse_1/1/strips/CH2/IMG00006.dcm",
        "/nfs/turbo/umms-tocho/data/db_nio_mouse/mouse/nio_mouse_1/1/strips/CH2/IMG00011.dcm",
        "/nfs/turbo/umms-tocho/data/db_nio_mouse/mouse/nio_mouse_1/1/strips/CH2/IMG00016.dcm",
        "/nfs/turbo/umms-tocho/data/db_nio_mouse/mouse/nio_mouse_1/1/strips/CH2/IMG00021.dcm",
        "/nfs/turbo/umms-tocho/data/db_nio_mouse/mouse/nio_mouse_1/1/strips/CH2/IMG00026.dcm",
        "/nfs/turbo/umms-tocho/data/db_nio_mouse/mouse/nio_mouse_1/1/strips/CH2/IMG00031.dcm",
    ]
    ch3_dicom_paths = [
        "/nfs/turbo/umms-tocho/data/db_nio_mouse/mouse/nio_mouse_1/1/strips/CH3/IMG00007.dcm",
        "/nfs/turbo/umms-tocho/data/db_nio_mouse/mouse/nio_mouse_1/1/strips/CH3/IMG00012.dcm",
        "/nfs/turbo/umms-tocho/data/db_nio_mouse/mouse/nio_mouse_1/1/strips/CH3/IMG00017.dcm",
        "/nfs/turbo/umms-tocho/data/db_nio_mouse/mouse/nio_mouse_1/1/strips/CH3/IMG00022.dcm",
        "/nfs/turbo/umms-tocho/data/db_nio_mouse/mouse/nio_mouse_1/1/strips/CH3/IMG00027.dcm",
        "/nfs/turbo/umms-tocho/data/db_nio_mouse/mouse/nio_mouse_1/1/strips/CH3/IMG00032.dcm",
    ]
    patch_dict = generate_paired_strip_patches_from_lists(
        ch2_dicom_paths=ch2_dicom_paths,
        ch3_dicom_paths=ch3_dicom_paths,
        strip_offset_interval=900,
        patch_stride=150,
    )
    save_tiled_patch_visualization(patch_dict, "stitched.png")


if __name__ == "__main__":
    main()
