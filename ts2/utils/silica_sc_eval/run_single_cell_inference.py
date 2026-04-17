import argparse
import logging
import os
import glob
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
from ts2.utils.silica_sc_eval.portal_assets import export_slide_portal_assets
from ts2.utils.ds_inf.model import get_model
from ts2.utils.srh_viz import prepare_two_channel_viz_image
from ts2.utils.strip_patching.patching import generate_paired_strip_patches_from_lists

logger = logging.getLogger(__name__)


def _resolve_standard_gmm_paths(gmm_cf: dict) -> dict:
    resolved = dict(gmm_cf)

    if all(
        key in resolved for key in ("model_path", "db_mean_path", "metrics_path", "k")
    ):
        return resolved

    artifact_dir = (
        resolved.get("gmm_ckpt")
        or resolved.get("artifact_dir")
        or resolved.get("gmm_path")
    )
    assert artifact_dir, (
        "Expected either explicit GMM artifact paths (`model_path`, `db_mean_path`, "
        "`metrics_path`, `k`) or a standard GMM output directory via `gmm_ckpt`, "
        "`artifact_dir`, or `gmm_path`."
    )

    artifact_dir = os.path.abspath(os.path.expanduser(str(artifact_dir)))
    assert os.path.exists(
        artifact_dir
    ), f"GMM artifact directory not found: {artifact_dir}"

    k = resolved.get("k")
    model_dir = os.path.join(artifact_dir, "models")
    stats_dir = os.path.join(artifact_dir, "stats")

    if k is None:
        candidate_models = sorted(glob.glob(os.path.join(model_dir, "gmm_g2m_m*.pkl")))
        assert candidate_models, f"No GMM model files found under {model_dir}"
        if len(candidate_models) != 1:
            candidate_names = ", ".join(
                os.path.basename(path) for path in candidate_models[:10]
            )
            raise ValueError(
                "Could not infer `k` because multiple standard GMM model files were "
                f"found under {model_dir}: {candidate_names}. Please set `k` "
                "explicitly in gmm_cf."
            )
        model_name = os.path.splitext(os.path.basename(candidate_models[0]))[0]
        k_str = model_name.removeprefix("gmm_g2m_m")
        assert k_str.isdigit(), f"Could not infer k from model filename: {model_name}"
        k = int(k_str)
        resolved["k"] = k

    resolved["gmm_ckpt"] = artifact_dir
    resolved["artifact_dir"] = artifact_dir
    resolved["model_path"] = os.path.join(model_dir, f"gmm_g2m_m{k}.pkl")
    resolved["db_mean_path"] = os.path.join(stats_dir, "db_mean.pt")
    resolved["metrics_path"] = os.path.join(stats_dir, "gmm_g2m_metrics.csv")

    for key in ("model_path", "db_mean_path", "metrics_path"):
        path = resolved[key]
        assert os.path.exists(path), f"Missing GMM {key}: {path}"

    return resolved


def infer_mosaic_io_paths(
    mouse_id: str,
    mosaic_id: str,
    data_root: str,
) -> dict:
    mosaic_root = os.path.join(data_root, mouse_id, str(mosaic_id))
    assert os.path.exists(mosaic_root), f"Mosaic root not found: {mosaic_root}"

    ch2_dir = os.path.join(mosaic_root, "strips", "CH2")
    ch3_dir = os.path.join(mosaic_root, "strips", "CH3")
    mosaics_dir_candidates = [
        os.path.join(mosaic_root, "mosaics"),
        os.path.join(mosaic_root, "mosaic"),
    ]
    mosaics_dir = next(
        (path for path in mosaics_dir_candidates if os.path.isdir(path)),
        None,
    )
    assert mosaics_dir is not None, (
        f"Could not find mosaic directory under {mosaic_root}. "
        "Expected either `mosaics/` or `mosaic/`."
    )

    ch2_dicom_paths = sorted(glob.glob(os.path.join(ch2_dir, "*.dcm")))
    ch3_dicom_paths = sorted(glob.glob(os.path.join(ch3_dir, "*.dcm")))
    mosaic_dicom_paths = sorted(glob.glob(os.path.join(mosaics_dir, "*.dcm")))

    assert ch2_dicom_paths, f"No CH2 DICOM files found under {ch2_dir}"
    assert ch3_dicom_paths, f"No CH3 DICOM files found under {ch3_dir}"
    assert len(ch2_dicom_paths) == len(ch3_dicom_paths), (
        f"CH2/CH3 strip count mismatch for {mouse_id}-{mosaic_id}: "
        f"{len(ch2_dicom_paths)} vs {len(ch3_dicom_paths)}"
    )
    assert mosaic_dicom_paths, f"No mosaic DICOM files found under {mosaics_dir}"

    return {
        "ch2_dicom_paths": ch2_dicom_paths,
        "ch3_dicom_paths": ch3_dicom_paths,
        "mosaic_dicom_path": mosaic_dicom_paths[0],
    }


def parse_mosaic_run(mosaic_run) -> dict:
    if isinstance(mosaic_run, str):
        parts = mosaic_run.split("/", 1)
        assert (
            len(parts) == 2
        ), f"Expected mosaic run in 'mouse_id/mosaic_id' format, got: {mosaic_run!r}"
        mouse_id, mosaic_id = parts
        assert mouse_id, f"Missing mouse_id in mosaic run: {mosaic_run!r}"
        assert mosaic_id, f"Missing mosaic_id in mosaic run: {mosaic_run!r}"
        return {"mouse_id": mouse_id, "mosaic_id": mosaic_id}

    assert isinstance(
        mosaic_run, dict
    ), f"Expected mosaic run to be a string or dict, got: {type(mosaic_run)}"
    assert "mouse_id" in mosaic_run, f"Missing mouse_id in mosaic run: {mosaic_run}"
    assert "mosaic_id" in mosaic_run, f"Missing mosaic_id in mosaic run: {mosaic_run}"
    return dict(mosaic_run)


def select_mosaic_runs_for_task(mosaic_runs, num_mosaic_per_task: int | None):
    mosaic_runs = list(mosaic_runs)
    if num_mosaic_per_task is None:
        return mosaic_runs

    num_mosaic_per_task = int(num_mosaic_per_task)
    assert (
        num_mosaic_per_task > 0
    ), f"Expected num_mosaic_per_task > 0, got {num_mosaic_per_task}"

    slurm_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if slurm_task_id is None:
        logger.warning(
            "infra.num_mosaic_per_task is set, but SLURM_ARRAY_TASK_ID is missing. "
            "Defaulting to task index 0."
        )
        task_idx = 0
    else:
        task_idx = int(slurm_task_id)
    assert task_idx >= 0, f"Expected non-negative SLURM_ARRAY_TASK_ID, got {task_idx}"

    start_idx = task_idx * num_mosaic_per_task
    end_idx = start_idx + num_mosaic_per_task
    selected_runs = mosaic_runs[start_idx:end_idx]
    assert selected_runs, (
        f"Task index {task_idx} selected no mosaic runs from "
        f"{len(mosaic_runs)} total runs with num_mosaic_per_task={num_mosaic_per_task}."
    )
    logger.info(
        "Selected mosaic runs [%d:%d) for SLURM_ARRAY_TASK_ID=%d (%d runs)",
        start_idx,
        max(end_idx, len(mosaic_runs)),
        task_idx,
        len(selected_runs),
    )
    return selected_runs


def _patch_dict_to_batch(
    patch_dict: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, List[str]]:
    patch_names = list(patch_dict)
    assert patch_names, "No patches were generated from the provided strips."

    images = np.stack(
        [np.moveaxis(np.asarray(patch_dict[name]), -1, 0) for name in patch_names],
        axis=0,
    )
    assert images.ndim == 4, f"Expected patch batch to be 4D, got shape {images.shape}"
    logger.info(
        "Prepared %d patches with batch shape %s", len(patch_names), tuple(images.shape)
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


def _representation_path_from_detection_row(row) -> str:
    centroid_r = int(round(float(row["centroid_r"])))
    centroid_c = int(round(float(row["centroid_c"])))
    return f"{row['patch']}#{centroid_r}_{centroid_c}"


def _drop_duplicate_representation_cells(
    results: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    if results.empty:
        return results, 0

    required_columns = {"patch", "celltype", "score", "centroid_r", "centroid_c"}
    missing_columns = required_columns.difference(results.columns)
    assert not missing_columns, (
        "Cannot deduplicate representation cells because detection results are "
        f"missing columns: {sorted(missing_columns)}"
    )

    dedup_df = results.copy()
    dedup_df["_original_order"] = np.arange(len(dedup_df), dtype=np.int64)
    eligible_mask = dedup_df["celltype"].isin({"nuclei", "mp"}) & (
        dedup_df["score"].astype(float) > 0.5
    )
    eligible = dedup_df.loc[eligible_mask].copy()
    if eligible.empty:
        return results, 0

    eligible["_representation_path"] = eligible.apply(
        _representation_path_from_detection_row,
        axis=1,
    )
    duplicate_mask = eligible.duplicated(subset=["_representation_path"], keep=False)
    if not duplicate_mask.any():
        return results, 0

    duplicate_count = int(duplicate_mask.sum())
    logger.info(
        "Dropping %d duplicate representation detections that would otherwise "
        "produce identical 48x48 crops",
        duplicate_count,
    )
    eligible = eligible.sort_values(
        ["_representation_path", "score", "_original_order"],
        ascending=[True, False, True],
        kind="stable",
    )
    eligible = eligible.drop_duplicates(
        subset=["_representation_path"],
        keep="first",
    )
    deduped = pd.concat(
        [dedup_df.loc[~eligible_mask], eligible.drop(columns=["_representation_path"])],
        axis=0,
    )
    deduped = deduped.sort_values("_original_order", kind="stable").drop(
        columns=["_original_order"]
    )
    dropped_count = len(results) - len(deduped)
    assert (
        dropped_count > 0
    ), "Expected duplicate representation deduplication to remove at least one row."
    return deduped, dropped_count


def _ensure_output_dir(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    assert os.path.isdir(output_dir), f"Failed to create output directory: {output_dir}"


def resolve_mosaic_output_dirs(
    mosaic_run: dict,
    out_root: str,
    static_infra_out_root: str,
    viz_candidate_name: str,
) -> dict[str, str]:
    assert out_root, "Expected a non-empty out_root."
    assert static_infra_out_root, "Expected a non-empty static_infra_out_root."

    output_dir = mosaic_run.get("output_dir", os.path.join(out_root, viz_candidate_name))
    static_output_dir = mosaic_run.get(
        "static_output_dir",
        os.path.join(static_infra_out_root, viz_candidate_name),
    )
    portal_dir = os.path.join(output_dir, "portal")
    dzi_dir = os.path.join(output_dir, "dzi")
    return {
        "output_dir": output_dir,
        "static_output_dir": static_output_dir,
        "portal_dir": portal_dir,
        "dzi_dir": dzi_dir,
    }


def _normalize_cache_stages(cache_stages: str) -> set[str]:
    if cache_stages is None:
        return set()
    normalized = str(cache_stages).replace(",", "").replace(" ", "").upper()
    invalid_stages = sorted(set(normalized) - set("PCRGV"))
    assert not invalid_stages, (
        f"Invalid cache stage letters: {invalid_stages}. "
        "Expected only letters from {P, C, R, G, V}."
    )
    return set(normalized)


def _can_use_cache(
    cache_stages: str,
    stage: str,
    artifact_paths,
) -> bool:
    stage = str(stage).upper()
    assert stage in {"P", "C", "R", "G", "V"}, f"Unexpected cache stage: {stage}"
    paths = (
        [artifact_paths] if isinstance(artifact_paths, str) else list(artifact_paths)
    )
    assert paths, f"No artifact paths were provided for cache stage {stage}."
    enabled_stages = _normalize_cache_stages(cache_stages)
    return stage in enabled_stages and all(os.path.exists(path) for path in paths)


def save_cell_prediction_visualization(
    patch_dict: Dict[str, np.ndarray],
    cell_predictions: pd.DataFrame,
    output_path: str,
    outline_color: Tuple[int, int, int] = (255, 255, 0),
    outline_width: int = 2,
) -> np.ndarray:
    assert (
        patch_dict
    ), "patch_dict is empty; cannot render cell prediction visualization."
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

    assert patch_items, "No patch items were available for visualization stitching."

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
    cropped_ims = [im[..., y0_r : y0_r + box_h, x0_r : x0_r + box_w] for im in ims]

    return cropped_ims, specs


def run_cell_representation_inference(
    patch_dict: Dict[str, np.ndarray],
    cell_predictions: pd.DataFrame,
    data_xform,
    representation_model: Dinov2EvalSystem,
    device: str,
    batch_size: int = 256,
) -> dict:
    logger.info("Running cell representation inference")
    transform = HistologyTransform(**data_xform)
    tile_size = 48

    cells = []
    for row in tqdm(
        cell_predictions.itertuples(index=False),
        total=len(cell_predictions),
        desc="Preparing cell crops",
    ):
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
        logger.info("No cells passed representation inference filters")
        return {"path": [], "label": [], "embeddings": torch.empty((0, 0))}

    logger.info("Prepared %d cells for representation inference", len(cells))

    representation_model.eval()

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
            batch_pred = representation_model.predict_step(batch, batch_idx)
            pred_raw["path"].extend(batch_pred["path"])
            pred_raw["label"].extend(batch_pred["label"])
            pred_raw["embeddings"].append(batch_pred["embeddings"].detach().cpu())

    assert pred_raw[
        "embeddings"
    ], "No embedding batches were produced during cell representation inference."
    pred_raw["embeddings"] = torch.cat(pred_raw["embeddings"], dim=0)
    assert pred_raw["embeddings"].shape[0] == len(
        pred_raw["path"]
    ), "Embedding count does not match number of paths."
    logger.info(
        "Completed cell representation inference with embedding shape %s",
        tuple(pred_raw["embeddings"].shape),
    )
    return pred_raw


def run_single_slide_gmm_inference(
    cell_representations: dict,
    gmm_metadata: dict,
    mixture_hard: bool = False,
    point_hard: bool = False,
) -> pd.DataFrame:
    logger.info("Running single-slide GMM inference")
    gmm_output = run_gmm_inference(
        inf_embeddings=cell_representations["embeddings"],
        gmm=gmm_metadata["gmm"],
        db_mean=gmm_metadata["db_mean"],
        mixture_score=gmm_metadata["mixture_score"],
        mixture_hard=mixture_hard,
        point_hard=point_hard,
    )

    predictions = pd.DataFrame(
        {
            "path": cell_representations["path"],
            "label": cell_representations["label"],
            "prediction": gmm_output["prediction"],
            "assignment": gmm_output["assignment"].tolist(),
        }
    )
    assert len(predictions) > 0, "GMM inference produced no predictions."
    logger.info("Generated %d GMM predictions", len(predictions))
    return predictions


def run_single_mosaic_pipeline(
    mosaic_run: dict,
    detection_model,
    representation_model: Dinov2EvalSystem,
    gmm_metadata: dict,
    cf,
    cell_batch_size: int = 256,
) -> dict:
    mouse_id = mosaic_run["mouse_id"]
    mosaic_id = mosaic_run["mosaic_id"]
    inferred_paths = infer_mosaic_io_paths(
        mouse_id=mouse_id,
        mosaic_id=mosaic_id,
        data_root=cf.data.data_root,
    )
    ch2_dicom_paths = inferred_paths["ch2_dicom_paths"]
    ch3_dicom_paths = inferred_paths["ch3_dicom_paths"]
    mosaic_dicom_path = inferred_paths["mosaic_dicom_path"]
    viz_candidate_name = f"{mouse_id}-{mosaic_id}"
    output_dirs = resolve_mosaic_output_dirs(
        mosaic_run=mosaic_run,
        out_root=cf.infra.out_root,
        static_infra_out_root=cf.infra.static_infra_out_root,
        viz_candidate_name=viz_candidate_name,
    )
    output_dir = output_dirs["output_dir"]
    static_output_dir = output_dirs["static_output_dir"]
    portal_dir = output_dirs["portal_dir"]
    dzi_dir = output_dirs["dzi_dir"]

    assert ch2_dicom_paths, f"No CH2 DICOM paths provided for {viz_candidate_name}."
    assert ch3_dicom_paths, f"No CH3 DICOM paths provided for {viz_candidate_name}."
    assert len(ch2_dicom_paths) == len(
        ch3_dicom_paths
    ), f"CH2/CH3 strip count mismatch for {viz_candidate_name}."

    logger.info(
        "Starting single-cell silica eval pipeline for %s on device=%s with %d strip pairs",
        viz_candidate_name,
        cf.infra.device,
        len(ch2_dicom_paths),
    )
    _ensure_output_dir(output_dir)
    _ensure_output_dir(static_output_dir)
    logger.info("Writing all outputs for %s under %s", viz_candidate_name, output_dir)
    logger.info(
        "Writing static infra outputs for %s under %s",
        viz_candidate_name,
        static_output_dir,
    )

    artifact_prefix = f"{viz_candidate_name}-"
    patch_dict_path = os.path.join(
        static_output_dir, f"{artifact_prefix}generated_patches.pt"
    )
    detected_cells_csv_path = os.path.join(
        static_output_dir, f"{artifact_prefix}detected_cells.csv"
    )
    cell_predictions_image_path = os.path.join(
        static_output_dir, f"{artifact_prefix}cell_predictions.png"
    )
    cell_representations_path = os.path.join(
        output_dir, f"{artifact_prefix}cell_representations.pt"
    )
    gmm_predictions_csv_path = os.path.join(
        output_dir, f"{artifact_prefix}cell_gmm_predictions.csv"
    )
    gmm_overlay_path = os.path.join(output_dir, f"{viz_candidate_name}.png")
    gmm_sharpened_overlay_path = os.path.join(
        output_dir, f"{viz_candidate_name}-sharpened.png"
    )
    gmm_score_overlay_path = os.path.join(output_dir, f"{viz_candidate_name}-score.png")
    gmm_mixture_overlay_path = os.path.join(
        output_dir, f"{viz_candidate_name}-mixture.png"
    )
    gmm_cluster_pct_pdf_path = os.path.join(
        output_dir, f"{viz_candidate_name}-cluster-pct.pdf"
    )
    gmm_cluster_pct_png_path = os.path.join(
        output_dir, f"{viz_candidate_name}-cluster-pct.png"
    )
    gmm_hist_path = os.path.join(output_dir, f"{viz_candidate_name}-hist.pdf")
    portal_manifest_path = os.path.join(portal_dir, "slide_manifest.json")
    portal_cells_path = os.path.join(portal_dir, "cells.json")

    logger.info("[%s] Begin step 1: generate strip patches", viz_candidate_name)
    if _can_use_cache(cf.infra.cache_stages, "P", patch_dict_path):
        logger.info(
            "[%s] Loading cached patches from %s",
            viz_candidate_name,
            patch_dict_path,
        )
        patch_dict = torch.load(patch_dict_path, weights_only=False)
    else:
        patch_dict = generate_paired_strip_patches_from_lists(
            ch2_dicom_paths=ch2_dicom_paths,
            ch3_dicom_paths=ch3_dicom_paths,
        )
        assert patch_dict, "Patch generation returned no patches."
        torch.save(patch_dict, patch_dict_path)
        assert os.path.exists(
            patch_dict_path
        ), f"Expected generated patches at {patch_dict_path}"
    assert patch_dict, "Patch dictionary is empty after step 1."
    images, patch_names = _patch_dict_to_batch(patch_dict)
    logger.info("[%s] End step 1: generate strip patches", viz_candidate_name)

    logger.info("[%s] Begin step 2: run cell detection inference", viz_candidate_name)
    if _can_use_cache(
        cf.infra.cache_stages,
        "C",
        [detected_cells_csv_path, cell_predictions_image_path],
    ):
        logger.info(
            "[%s] Loading cached detection outputs from %s and %s",
            viz_candidate_name,
            detected_cells_csv_path,
            cell_predictions_image_path,
        )
        results = pd.read_csv(detected_cells_csv_path)
    else:
        results = pd.DataFrame(
            run_inference(
                images=images,
                model=detection_model,
                classes=DEFAULT_CLASSES,
                patch_names=patch_names,
            )
        )
        assert not results.empty, "Cell detection inference returned no detections."
        logger.info(
            "[%s] Cell detection produced %d detections",
            viz_candidate_name,
            len(results),
        )
        results = _add_global_coordinates(results)
        results.to_csv(detected_cells_csv_path, index=False)
        logger.info(
            "[%s] Saving cell detection visualization to %s",
            viz_candidate_name,
            cell_predictions_image_path,
        )
        save_cell_prediction_visualization(
            patch_dict=patch_dict,
            cell_predictions=results,
            output_path=cell_predictions_image_path,
        )

    results, dropped_duplicate_repr_cells = _drop_duplicate_representation_cells(results)
    if dropped_duplicate_repr_cells > 0:
        logger.info(
            "[%s] Removed %d duplicate detected cells before representation inference",
            viz_candidate_name,
            dropped_duplicate_repr_cells,
        )
        results.to_csv(detected_cells_csv_path, index=False)
        logger.info(
            "[%s] Refreshing cell detection visualization after deduplication at %s",
            viz_candidate_name,
            cell_predictions_image_path,
        )
        save_cell_prediction_visualization(
            patch_dict=patch_dict,
            cell_predictions=results,
            output_path=cell_predictions_image_path,
        )

    assert not results.empty, "Detection results are empty after step 2."
    required_columns = {
        "patch",
        "celltype",
        "score",
        "centroid_r",
        "centroid_c",
        "bbox_w",
        "bbox_h",
    }
    missing_columns = required_columns.difference(results.columns)
    assert (
        not missing_columns
    ), f"Missing expected detection columns: {sorted(missing_columns)}"

    logger.info("[%s] End step 2: run cell detection inference", viz_candidate_name)

    logger.info(
        "[%s] Begin step 3: run cell representation inference", viz_candidate_name
    )
    use_repr_cache = (
        _can_use_cache(cf.infra.cache_stages, "R", cell_representations_path)
        and dropped_duplicate_repr_cells == 0
    )
    if use_repr_cache:
        logger.info(
            "[%s] Loading cached cell representations from %s",
            viz_candidate_name,
            cell_representations_path,
        )
        cell_representations = torch.load(cell_representations_path, weights_only=False)
    else:
        if dropped_duplicate_repr_cells > 0 and os.path.exists(cell_representations_path):
            logger.info(
                "[%s] Ignoring cached cell representations because detection "
                "deduplication changed the inference set",
                viz_candidate_name,
            )
        cell_representations = run_cell_representation_inference(
            patch_dict=patch_dict,
            cell_predictions=results,
            data_xform=cf.data_xform,
            representation_model=representation_model,
            device=cf.infra.device,
            batch_size=cell_batch_size,
        )
        torch.save(cell_representations, cell_representations_path)
    assert (
        "embeddings" in cell_representations
    ), "Cached cell representations missing `embeddings`."
    logger.info(
        "[%s] End step 3: run cell representation inference", viz_candidate_name
    )

    logger.info("[%s] Begin step 4: run GMM inference", viz_candidate_name)
    use_gmm_cache = (
        _can_use_cache(cf.infra.cache_stages, "G", gmm_predictions_csv_path)
        and dropped_duplicate_repr_cells == 0
    )
    if use_gmm_cache:
        logger.info(
            "[%s] Loading cached GMM predictions from %s",
            viz_candidate_name,
            gmm_predictions_csv_path,
        )
        gmm_predictions = pd.read_csv(gmm_predictions_csv_path)
    else:
        if dropped_duplicate_repr_cells > 0 and os.path.exists(gmm_predictions_csv_path):
            logger.info(
                "[%s] Ignoring cached GMM predictions because detection "
                "deduplication changed the inference set",
                viz_candidate_name,
            )
        gmm_predictions = run_single_slide_gmm_inference(
            cell_representations=cell_representations,
            gmm_metadata=gmm_metadata,
            mixture_hard=cf.gmm.mixture_hard,
            point_hard=cf.gmm.point_hard,
        )
        gmm_predictions.to_csv(gmm_predictions_csv_path, index=False)
    assert not gmm_predictions.empty, "GMM predictions are empty after step 4."
    logger.info("[%s] Generated GMM predictions", viz_candidate_name)
    logger.info("[%s] End step 4: run GMM inference", viz_candidate_name)

    logger.info("[%s] Begin step 5: render GMM visualizations", viz_candidate_name)
    use_viz_cache = (
        _can_use_cache(
            cf.infra.cache_stages,
            "V",
            [
                gmm_overlay_path,
                gmm_sharpened_overlay_path,
                gmm_score_overlay_path,
                gmm_mixture_overlay_path,
                gmm_cluster_pct_pdf_path,
                gmm_cluster_pct_png_path,
                gmm_hist_path,
            ],
        )
        and dropped_duplicate_repr_cells == 0
    )
    if use_viz_cache:
        logger.info(
            "[%s] Using cached GMM visualization outputs at %s, %s, %s, %s, %s, %s, and %s",
            viz_candidate_name,
            gmm_overlay_path,
            gmm_sharpened_overlay_path,
            gmm_score_overlay_path,
            gmm_mixture_overlay_path,
            gmm_cluster_pct_pdf_path,
            gmm_cluster_pct_png_path,
            gmm_hist_path,
        )
    else:
        if dropped_duplicate_repr_cells > 0:
            logger.info(
                "[%s] Regenerating GMM visualizations because detection "
                "deduplication changed downstream predictions",
                viz_candidate_name,
            )
        logger.info("[%s] Rendering GMM visualizations", viz_candidate_name)
        generate_visualization(
            predictions=gmm_predictions,
            mosaic_dicom_path=mosaic_dicom_path,
            out_dir=output_dir,
            candidate_name=viz_candidate_name,
            strip_padding=50,
        )
    logger.info("[%s] End step 5: render GMM visualizations", viz_candidate_name)
    logger.info("[%s] Begin step 6: export portal metadata", viz_candidate_name)
    _ensure_output_dir(portal_dir)
    use_portal_cache = (
        _can_use_cache(
            cf.infra.cache_stages,
            "V",
            [
                portal_manifest_path,
                portal_cells_path,
            ],
        )
        and dropped_duplicate_repr_cells == 0
    )
    if use_portal_cache:
        logger.info(
            "[%s] Using cached portal metadata from %s",
            viz_candidate_name,
            portal_manifest_path,
        )
    else:
        if dropped_duplicate_repr_cells > 0:
            logger.info(
                "[%s] Regenerating portal metadata because detection "
                "deduplication changed downstream predictions",
                viz_candidate_name,
            )
        export_slide_portal_assets(
            slide_id=viz_candidate_name,
            mosaic_dicom_path=mosaic_dicom_path,
            detected_cells=results,
            gmm_predictions=gmm_predictions,
            portal_dir=portal_dir,
            strip_padding=50,
        )
    logger.info("[%s] End step 6: export portal metadata", viz_candidate_name)
    logger.info("Finished single-cell silica eval pipeline for %s", viz_candidate_name)
    return {
        "mouse_id": mouse_id,
        "mosaic_id": mosaic_id,
        "viz_candidate_name": viz_candidate_name,
        "output_dir": output_dir,
        "static_output_dir": static_output_dir,
        "portal_dir": portal_dir,
        "dzi_dir": dzi_dir,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to the cell inference YAML config.",
    )
    return parser.parse_args()


def main(config_path: str) -> None:
    logging_format_str = (
        "[%(levelname)-s|%(asctime)s|%(name)s|"
        + "%(filename)s:%(lineno)d|%(funcName)s] %(message)s"
    )
    logging.basicConfig(
        level=logging.INFO,
        format=logging_format_str,
    )
    logging.getLogger("fontTools.subset").setLevel(logging.WARNING)
    assert config_path, "Expected a non-empty config path."
    assert os.path.exists(config_path), f"Config file not found: {config_path}"
    cf = OmegaConf.load(config_path)
    assert cf.infra.out_root, "Expected infra.out_root in config."
    assert (
        cf.infra.static_infra_out_root
    ), "Expected infra.static_infra_out_root in config."

    logger.info("Loading cell detection model from %s", cf.cell_detection.ckpt_path)
    detection_model = get_model(
        checkpoint_path=cf.cell_detection.ckpt_path,
        num_classes=len(DEFAULT_CLASSES),
        device=cf.infra.device,
    )

    logger.info("Loading cell representation model from pretrained weights")
    representation_model = Dinov2EvalSystem(**cf.silica).to(cf.infra.device)
    representation_model.eval()

    logger.info("Loading pretrained GMM metadata")
    gmm_metadata = instantiate_gmm(_resolve_standard_gmm_paths(cf.gmm))

    selected_mosaic_runs = select_mosaic_runs_for_task(
        cf.data.mosaic_runs,
        cf.infra.get("num_mosaic_per_task"),
    )

    pipeline_results = []
    failed_mosaic_runs = []
    for mosaic_run_entry in tqdm(
        selected_mosaic_runs,
        desc="Single-mosaic inference",
    ):
        mosaic_label = str(mosaic_run_entry)
        try:
            mosaic_run = parse_mosaic_run(mosaic_run_entry)
            mosaic_label = f"{mosaic_run['mouse_id']}/{mosaic_run['mosaic_id']}"
            pipeline_results.append(
                run_single_mosaic_pipeline(
                    mosaic_run=mosaic_run,
                    detection_model=detection_model,
                    representation_model=representation_model,
                    gmm_metadata=gmm_metadata,
                    cf=cf,
                )
            )
        except Exception as exc:
            logger.exception("Mosaic run failed for %s", mosaic_label)
            failed_mosaic_runs.append(
                {
                    "mosaic_run": mosaic_label,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    logger.info(
        "Completed %d mosaic runs: %s",
        len(pipeline_results),
        ", ".join(result["viz_candidate_name"] for result in pipeline_results),
    )
    if failed_mosaic_runs:
        failed_summary = "; ".join(
            f"{failure['mosaic_run']} ({failure['error_type']}: {failure['error']})"
            for failure in failed_mosaic_runs
        )
        raise RuntimeError(
            f"{len(failed_mosaic_runs)} of {len(selected_mosaic_runs)} mosaic runs "
            f"failed: {failed_summary}"
        )


if __name__ == "__main__":
    args = parse_args()
    main(config_path=args.config)
