from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from omegaconf import OmegaConf
import torch
from tqdm.auto import tqdm

from ts2.data.transforms import HistologyTransform
from ts2.lm.dinov2_eval_system import Dinov2EvalSystem
from ts2.utils.ds_inf.inference import (
    DEFAULT_CLASSES,
    run_inference,
)
from ts2.utils.silica_sc_eval.generate_gmm_visualization import generate_visualization
from ts2.utils.silica_sc_eval.gmm_inference import instantiate_gmm, run_gmm_inference
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


def dilate_ims(ims, bbox_spec, box_h: int = 48, box_w: int = 48):
    x, y, w, h = bbox_spec
    is_edge = 0

    if y - (box_h // 2) < 0:
        start_y = torch.tensor(0)
        is_edge = 1
    elif y + (box_h // 2) > ims[0].shape[-2]:
        start_y = torch.tensor(ims[0].shape[-2] - box_h)
        is_edge = 1
    else:
        start_y = y - (box_h // 2)

    if x - (box_w // 2) < 0:
        start_x = torch.tensor(0)
        is_edge = 1
    elif x + (box_w // 2) > ims[0].shape[-1]:
        start_x = torch.tensor(ims[0].shape[-1] - box_w)
        is_edge = 1
    else:
        start_x = x - (box_w // 2)

    specs = torch.stack([start_x, start_y, torch.tensor(is_edge)])

    x0_r, y0_r = int(start_x.round()), int(start_y.round())
    cropped_ims = [im[..., y0_r:y0_r + box_h, x0_r:x0_r + box_w] for im in ims]

    return cropped_ims, specs

def run_cell_representation_inference(
    patch_dict: Dict[str, np.ndarray],
    cell_predictions: pd.DataFrame,
    data_xform_cf,
    lightning_module_params,
    device: str,
    batch_size: int = 256,
) -> dict:
    transform = HistologyTransform(**data_xform_cf)
    tile_size = 48

    cells = []
    for row in cell_predictions.itertuples(index=False):
        if row.celltype not in {"nuclei", "mp"} or row.score <= 0.5:
            continue

        patch_array = patch_dict.get(row.patch)
        if patch_array is None:
            continue

        centroid_r = int(round(row.centroid_r))
        centroid_c = int(round(row.centroid_c))
        patch_tensor = torch.as_tensor(np.moveaxis(patch_array, -1, 0))
        cropped_ims, _ = dilate_ims(
            [patch_tensor],
            bbox_spec=(
                torch.tensor(centroid_c),
                torch.tensor(centroid_r),
                torch.tensor(row.bbox_w),
                torch.tensor(row.bbox_h),
            ),
            box_h=tile_size,
            box_w=tile_size,
        )
        crop_tensor = cropped_ims[0]
        if crop_tensor.shape[-2:] != (tile_size, tile_size):
            continue

        cells.append(
            {
                "image": transform(crop_tensor).to(torch.float32),
                "path": f"{row.patch}#{centroid_r}_{centroid_c}",
                "label": row.celltype,
            }
        )

    if not cells:
        return {"path": [], "label": [], "embeddings": torch.empty((0, 0))}


    model = Dinov2EvalSystem(**lightning_module_params).to(device)
    model.eval()

    pred_raw = {"path": [], "label": [], "embeddings": []}
    with torch.inference_mode():
        for batch_idx, start_idx in enumerate(
            tqdm(
                range(0, len(cells), batch_size),
                total=(len(cells) + batch_size - 1) // batch_size,
                desc="Cell representation inference",
            )
        ):
            batch_cells = cells[start_idx : start_idx + batch_size]
            batch = {
                "image": torch.stack([cell["image"] for cell in batch_cells], dim=0)
                .unsqueeze(1)
                .to(torch.float32)
                .to(device),
                "path": [cell["path"] for cell in batch_cells],
                "label": [cell["label"] for cell in batch_cells],
            }
            batch_pred = model.predict_step(batch, batch_idx)
            pred_raw["path"].extend(batch_pred["path"])
            pred_raw["label"].extend(batch_pred["label"])
            pred_raw["embeddings"].append(batch_pred["embeddings"].detach().cpu())

    pred_raw["embeddings"] = torch.cat(pred_raw["embeddings"], dim=0)
    return pred_raw


def run_single_slide_gmm_inference(
    cell_representations: dict,
    gmm_cf: dict,
) -> pd.DataFrame:
    gmm_metadata = instantiate_gmm(gmm_cf)
    gmm_output = run_gmm_inference(
        inf_embeddings=cell_representations["embeddings"],
        gmm=gmm_metadata["gmm"],
        db_mean=gmm_metadata["db_mean"],
        mixture_score=gmm_metadata["mixture_score"],
        mixture_hard=False,
        point_hard=False,
    )

    return pd.DataFrame(
        {
            "path": cell_representations["path"],
            "label": cell_representations["label"],
            "prediction": gmm_output["prediction"],
            "assignment": gmm_output["assignment"].tolist(),
        }
    )


def main() -> None:
    mouse_id = "nio_mouse_1"
    mosaic_id = "10"

    ch2_dicom_paths = [
        f"/nfs/turbo/umms-tocho/data/db_nio_mouse/mouse/{mouse_id}/{mosaic_id}/strips/CH2/IMG00006.dcm",
        f"/nfs/turbo/umms-tocho/data/db_nio_mouse/mouse/{mouse_id}/{mosaic_id}/strips/CH2/IMG00011.dcm",
        f"/nfs/turbo/umms-tocho/data/db_nio_mouse/mouse/{mouse_id}/{mosaic_id}/strips/CH2/IMG00016.dcm",
        f"/nfs/turbo/umms-tocho/data/db_nio_mouse/mouse/{mouse_id}/{mosaic_id}/strips/CH2/IMG00021.dcm",
        f"/nfs/turbo/umms-tocho/data/db_nio_mouse/mouse/{mouse_id}/{mosaic_id}/strips/CH2/IMG00026.dcm",
        f"/nfs/turbo/umms-tocho/data/db_nio_mouse/mouse/{mouse_id}/{mosaic_id}/strips/CH2/IMG00031.dcm",
    ]
    ch3_dicom_paths = [
        f"/nfs/turbo/umms-tocho/data/db_nio_mouse/mouse/{mouse_id}/{mosaic_id}/strips/CH3/IMG00007.dcm",
        f"/nfs/turbo/umms-tocho/data/db_nio_mouse/mouse/{mouse_id}/{mosaic_id}/strips/CH3/IMG00012.dcm",
        f"/nfs/turbo/umms-tocho/data/db_nio_mouse/mouse/{mouse_id}/{mosaic_id}/strips/CH3/IMG00017.dcm",
        f"/nfs/turbo/umms-tocho/data/db_nio_mouse/mouse/{mouse_id}/{mosaic_id}/strips/CH3/IMG00022.dcm",
        f"/nfs/turbo/umms-tocho/data/db_nio_mouse/mouse/{mouse_id}/{mosaic_id}/strips/CH3/IMG00027.dcm",
        f"/nfs/turbo/umms-tocho/data/db_nio_mouse/mouse/{mouse_id}/{mosaic_id}/strips/CH3/IMG00032.dcm",
    ]
    checkpoint_path = "/nfs/turbo/umms-tocho-snr/exp/chengjia/ds_reretrain/1e4/439971b1_Apr30-17-27-52_sd1000_ALLDATA_/models/ckpt-epoch19-step5500-loss0.00.ckpt"
    mosaic_dicom_path = f"/nfs/turbo/umms-tocho/data/db_nio_mouse/mouse/{mouse_id}/{mosaic_id}/mosaics/IMG00001.dcm"
    viz_candidate_name = f"{mouse_id}-{mosaic_id}"

    data_xform_cf = OmegaConf.create({
        "which_set": "srh",
        "base_aug_params": {
            "laser_noise_config": None,
            "get_third_channel_params": {
                "mode": "three_channels",
                "subtracted_base": 0.07629394531,
            },
            "to_uint8": False,
        },
        "strong_aug_params": {
            "aug_list": [
                {
                    "which": "center_crop_always_apply",
                    "params": {"size": 48},
                }
            ],
            "aug_prob": 1,
        },
    })
    lightning_module_params = OmegaConf.create({
        "model_hyperparams": {
            "arch": "vit_small",
            "img_size": 48,
            "patch_size": 4,
            "block_chunks": 4,
            "ffn_bias": True,
            "ffn_layer": "mlp",
            "num_register_tokens": 0,
            "interpolate_antialias": False,
            "interpolate_offset": 0.1,
            "layerscale": 1.0e-05,
            "proj_bias": True,
            "qkv_bias": True,
        },
        "pretrained_weights": "/nfs/turbo/umms-tocho-snr/exp/chengjia/ts2/fmi_dinov2_cc_new/6778e5d1_May27-15-59-58_sd1000_dev_tune0/models/eval/training_124999/teacher_checkpoint.pth",
    })
    gmm_cf = {
        "db_mean_path": "/nfs/turbo/umms-tocho/code/chengjia/torchsrh2/ts2/playgrounds/silica_gmm_reproduce/stats/db_mean.pt",
        "metrics_path": "/nfs/turbo/umms-tocho/code/chengjia/torchsrh2/ts2/playgrounds/silica_gmm_reproduce/stats/gmm_g2m_metrics.csv",
        "model_path": "/nfs/turbo/umms-tocho/code/chengjia/torchsrh2/ts2/playgrounds/silica_gmm_reproduce/models/gmm_g2m_m32.pkl",
        "k": 32,
        "zero_components": [],
    }

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
    cell_representations = run_cell_representation_inference(
        patch_dict=patch_dict,
        cell_predictions=results,
        data_xform_cf=data_xform_cf,
        lightning_module_params=lightning_module_params,
        device=device,
        batch_size=256,
    )
    gmm_predictions = run_single_slide_gmm_inference(
        cell_representations=cell_representations,
        gmm_cf=gmm_cf,
    )
    #torch.save(cell_representations, "cell_representations.pt")
    #gmm_predictions.to_csv("cell_gmm_predictions.csv", index=False)

    generate_visualization(
        predictions=gmm_predictions,
        mosaic_dicom_path=mosaic_dicom_path,
        out_dir=".",
        candidate_name=viz_candidate_name,
        strip_padding=50,
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
