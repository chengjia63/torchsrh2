from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import torch
import torch

from ts2.utils.ds_inf.inference import (
    DEFAULT_CLASSES,
    run_inference,
)
from ts2.utils.ds_inf.model import get_model
from ts2.utils.srh_viz import prepare_two_channel_viz_image
from ts2.utils.strip_patching.patching import generate_paired_strip_patches_from_lists


def _patch_dict_to_batch(
    patch_dict: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, List[str]]:
    patch_names = list(patch_dict)
    if not patch_names:
        raise ValueError("No patches were generated from the provided strips.")

    images = np.stack(
        [np.moveaxis(np.asarray(patch_dict[name]), -1, 0) for name in patch_names],
        axis=0,
    )
    return images, patch_names


def _parse_patch_metadata(patch_name: str) -> Tuple[Optional[int], int, int]:
    prefix, coords = patch_name.rsplit("-", 1)
    top_str, left_str = coords.split("_", 1)

    strip_index = None
    if prefix.startswith("strip_"):
        try:
            strip_index = int(prefix.split("_", 1)[1])
        except ValueError:
            strip_index = None

    return strip_index, int(top_str), int(left_str)


def _add_global_coordinates(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results

    metadata = results["patch"].map(_parse_patch_metadata)
    metadata_df = pd.DataFrame(
        metadata.tolist(),
        columns=["strip_index", "patch_top", "patch_left"],
        index=results.index,
    )
    results = pd.concat([results, metadata_df], axis=1)
    results["global_bbox_cx"] = results["patch_left"] + results["bbox_cx"]
    results["global_bbox_cy"] = results["patch_top"] + results["bbox_cy"]
    results["global_centroid_r"] = results["patch_top"] + results["centroid_r"]
    results["global_centroid_c"] = results["patch_left"] + results["centroid_c"]
    return results


def save_cell_prediction_visualization(
    patch_dict: Dict[str, np.ndarray],
    cell_predictions: pd.DataFrame,
    output_path: str,
    outline_color: Tuple[int, int, int] = (255, 255, 0),
    outline_width: int = 2,
) -> np.ndarray:
    patch_items = []
    max_bottom = 0
    max_right = 0

    for patch_name, patch_array in patch_dict.items():
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

    for top, left, patch_np in patch_items:
        patch_height, patch_width, _ = patch_np.shape
        stitched_sum[top : top + patch_height, left : left + patch_width] += patch_np
        stitched_count[top : top + patch_height, left : left + patch_width] += 1

    stitched_image = np.divide(
        stitched_sum,
        np.maximum(stitched_count, 1),
    ).astype(np.uint8)

    image = Image.fromarray(stitched_image)
    if not cell_predictions.empty:
        draw = ImageDraw.Draw(image)
        for _, row in cell_predictions.iterrows():
            half_width = row["bbox_w"] / 2.0
            half_height = row["bbox_h"] / 2.0
            left = row["global_bbox_cx"] - half_width
            top = row["global_bbox_cy"] - half_height
            right = row["global_bbox_cx"] + half_width
            bottom = row["global_bbox_cy"] + half_height
            draw.rectangle(
                [(left, top), (right, bottom)],
                outline=outline_color,
                width=outline_width,
            )

    image.save(output_path)
    return np.asarray(image)


def main() -> None:
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
    checkpoint_path = "/nfs/turbo/umms-tocho-snr/exp/chengjia/ds_reretrain/1e4/439971b1_Apr30-17-27-52_sd1000_ALLDATA_/models/ckpt-epoch19-step5500-loss0.00.ckpt"

    #output_csv = "/path/to/cell_predictions.csv"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if len(ch2_dicom_paths) != len(ch3_dicom_paths):
        raise ValueError(
            "ch2_dicom_paths and ch3_dicom_paths must have the same length."
        )

    patch_dict = generate_paired_strip_patches_from_lists(
        ch2_dicom_paths=ch2_dicom_paths,
        ch3_dicom_paths=ch3_dicom_paths,
    )
    images, patch_names = _patch_dict_to_batch(patch_dict)

    model = get_model(
        checkpoint_path=checkpoint_path,
        num_classes=len(DEFAULT_CLASSES),
        device=device,
    )

    results = pd.DataFrame(
        run_inference(
            images=images,
            model=model,
            classes=DEFAULT_CLASSES,
            patch_names=patch_names,
        )
    )

    results = _add_global_coordinates(results)

    save_cell_prediction_visualization(
        patch_dict=patch_dict,
        cell_predictions=results,
        output_path="cell_predictions.png",
    )

    #output_path = Path(output_csv).expanduser()
    #output_path.parent.mkdir(parents=True, exist_ok=True)
    #results.to_csv(output_path, index=False)
#
    #print(
    #    f"Saved {len(results)} detections from {len(patch_names)} patches to {output_path}"
    #)


if __name__ == "__main__":
    main()
