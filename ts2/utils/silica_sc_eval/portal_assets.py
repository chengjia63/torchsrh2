from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import matplotlib.cm as cm
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

from ts2.utils.silica_sc_eval.generate_gmm_visualization import (
    load_mosaic_image,
    parse_assignment_vector,
    parse_cell_path,
)

logger = logging.getLogger(__name__)


def _ensure_dir(path: str | os.PathLike[str]) -> Path:
    out_path = Path(path)
    out_path.mkdir(parents=True, exist_ok=True)
    assert out_path.is_dir(), f"Failed to create directory: {out_path}"
    return out_path


def _representation_path_from_detection_row(row: pd.Series) -> str:
    centroid_r = int(round(float(row["centroid_r"])))
    centroid_c = int(round(float(row["centroid_c"])))
    return f"{row['patch']}#{centroid_r}_{centroid_c}"


def _build_representation_cell_table(detected_cells: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "patch",
        "celltype",
        "score",
        "centroid_r",
        "centroid_c",
        "global_centroid_r",
        "global_centroid_c",
    }
    missing_columns = required_columns.difference(detected_cells.columns)
    assert not missing_columns, (
        "Detected cells table is missing required columns for portal export: "
        f"{sorted(missing_columns)}"
    )

    eligible = detected_cells.loc[
        detected_cells["celltype"].isin({"nuclei", "mp"})
        & (detected_cells["score"].astype(float) > 0.5)
    ].copy()
    eligible["path"] = eligible.apply(_representation_path_from_detection_row, axis=1)
    duplicate_paths = eligible["path"].duplicated(keep=False)
    assert not duplicate_paths.any(), (
        "Duplicate representation paths reached portal export. Detection-time "
        "deduplication should have removed these rows first: "
        + ", ".join(eligible.loc[duplicate_paths, "path"].head(10).tolist())
    )
    return eligible


def _score_to_hex(normal_scores: np.ndarray) -> list[str]:
    cmap = cm.get_cmap("RdYlGn")
    rgba = cmap(np.asarray(normal_scores, dtype=np.float64))
    rgb = np.clip(np.round(rgba[:, :3] * 255.0), 0, 255).astype(np.uint8)
    return [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in rgb]


def build_portal_cell_payload(
    detected_cells: pd.DataFrame,
    gmm_predictions: pd.DataFrame,
) -> dict:
    assert not detected_cells.empty, "Detected cells table is empty."
    assert not gmm_predictions.empty, "GMM predictions table is empty."
    assert "path" in gmm_predictions.columns, "Expected `path` in GMM predictions."
    assert (
        "prediction" in gmm_predictions.columns
    ), "Expected `prediction` in GMM predictions."
    assert (
        "assignment" in gmm_predictions.columns
    ), "Expected `assignment` in GMM predictions."

    detected_repr_cells = _build_representation_cell_table(detected_cells)
    duplicate_prediction_paths = gmm_predictions["path"].duplicated(keep=False)
    assert not duplicate_prediction_paths.any(), (
        "Duplicate GMM prediction paths reached portal export. Detection-time "
        "deduplication should have removed these rows first: "
        + ", ".join(
            gmm_predictions.loc[duplicate_prediction_paths, "path"].head(10).tolist()
        )
    )

    merged = gmm_predictions.merge(
        detected_repr_cells[
            [
                "path",
                "patch",
                "celltype",
                "score",
                "global_centroid_r",
                "global_centroid_c",
            ]
        ],
        on="path",
        how="left",
        validate="one_to_one",
    )
    missing_matches = int(merged["global_centroid_r"].isna().sum())
    assert (
        missing_matches == 0
    ), f"Could not match {missing_matches} GMM predictions back to detected cells."

    assignment_vectors = [
        parse_assignment_vector(value) for value in merged["assignment"]
    ]
    assignment_matrix = np.stack(assignment_vectors, axis=0)
    dominant_cluster = np.argmax(assignment_matrix, axis=1).astype(np.int32)
    dominant_cluster_confidence = assignment_matrix[
        np.arange(len(assignment_matrix)),
        dominant_cluster,
    ]

    normal_score = merged["prediction"].astype(float).to_numpy(dtype=np.float64)
    tumor_score = 1.0 - normal_score
    tumor_score_display = 100 - np.clip(np.floor(normal_score * 100.0), 0, 99).astype(
        np.int32
    )
    dominant_cluster_display = np.clip(
        np.floor(dominant_cluster_confidence * 100.0),
        0,
        99,
    ).astype(np.int32)

    payload = {
        "cell_count": int(len(merged)),
        "num_clusters": int(assignment_matrix.shape[1]),
        "x": np.round(
            merged["global_centroid_c"].astype(float).to_numpy(dtype=np.float64)
        )
        .astype(np.int32)
        .tolist(),
        "y": np.round(
            merged["global_centroid_r"].astype(float).to_numpy(dtype=np.float64)
        )
        .astype(np.int32)
        .tolist(),
        "normal_score": [round(float(value), 6) for value in normal_score],
        "tumor_score": [round(float(value), 6) for value in tumor_score],
        "tumor_score_display": tumor_score_display.tolist(),
        "dominant_cluster": dominant_cluster.tolist(),
        "dominant_cluster_display": dominant_cluster_display.tolist(),
        "detection_score": [
            round(float(value), 6)
            for value in merged["score"].astype(float).to_numpy(dtype=np.float64)
        ],
        "cell_type": merged["celltype"].astype(str).tolist(),
        "dot_color": _score_to_hex(normal_score),
    }
    return payload


def _nearest_cell_area_mean_tumor_probability(
    *,
    image_shape: tuple[int, ...],
    cell_y: np.ndarray,
    cell_x: np.ndarray,
    tumor_probability: np.ndarray,
) -> float:
    assert (
        len(image_shape) >= 2
    ), f"Expected image shape with height and width, got {image_shape}"
    height, width = int(image_shape[0]), int(image_shape[1])
    assert (
        height > 0 and width > 0
    ), f"Expected positive image size, got {(height, width)}"

    cell_y = np.asarray(cell_y, dtype=np.float64)
    cell_x = np.asarray(cell_x, dtype=np.float64)
    assert (
        cell_y.shape == cell_x.shape
    ), f"Cell coordinate shape mismatch: y={cell_y.shape}, x={cell_x.shape}"
    assert (
        np.isfinite(cell_y).all() and np.isfinite(cell_x).all()
    ), "Cell coordinates contain non-finite values."
    in_bounds = (cell_y >= 0.0) & (cell_y < height) & (cell_x >= 0.0) & (cell_x < width)
    assert bool(in_bounds.all()), (
        "Cannot compute area-based slide tumor probability because some cell "
        f"coordinates fall outside image bounds {(height, width)}."
    )

    seed_coordinates = np.stack([cell_y, cell_x], axis=1)
    unique_seed_coordinates = np.unique(seed_coordinates, axis=0)
    assert len(unique_seed_coordinates) == len(seed_coordinates), (
        "Cannot compute area-based slide tumor probability because multiple cells "
        "have identical coordinates."
    )

    tree = KDTree(seed_coordinates)
    area_weights = np.zeros(len(seed_coordinates), dtype=np.int64)
    chunk_num_rows = max(1, 1_000_000 // width)
    x_coords = np.arange(width, dtype=np.float64)
    for row_start in range(0, height, chunk_num_rows):
        row_end = min(row_start + chunk_num_rows, height)
        y_coords = np.arange(row_start, row_end, dtype=np.float64)
        yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")
        pixel_coordinates = np.column_stack([yy.reshape(-1), xx.reshape(-1)])
        nearest_cell_indices = tree.query(
            pixel_coordinates,
            k=1,
            return_distance=False,
        ).reshape(-1)
        area_weights += np.bincount(
            nearest_cell_indices,
            minlength=len(seed_coordinates),
        )

    assert (
        int(area_weights.sum()) == height * width
    ), "Nearest-cell area weights do not cover the full image."

    return float(np.average(tumor_probability, weights=area_weights))


def build_slide_statistics(
    *,
    gmm_predictions: pd.DataFrame,
    image_shape: tuple[int, ...],
    tumor_probability_threshold: float = 0.5,
) -> dict:
    assert not gmm_predictions.empty, "GMM predictions table is empty."
    assert (
        "prediction" in gmm_predictions.columns
    ), "Expected `prediction` in GMM predictions."
    assert "path" in gmm_predictions.columns, "Expected `path` in GMM predictions."
    assert 0.0 <= tumor_probability_threshold <= 1.0, (
        "Expected tumor_probability_threshold in [0, 1], got "
        f"{tumor_probability_threshold}"
    )

    duplicate_prediction_paths = gmm_predictions["path"].duplicated(keep=False)
    assert not duplicate_prediction_paths.any(), (
        "Duplicate GMM prediction paths reached slide statistics. Detection-time "
        "deduplication should have removed these rows first: "
        + ", ".join(
            gmm_predictions.loc[duplicate_prediction_paths, "path"].head(10).tolist()
        )
    )

    normal_probability = (
        gmm_predictions["prediction"].astype(float).to_numpy(dtype=np.float64)
    )
    assert np.isfinite(
        normal_probability
    ).all(), "GMM predictions contain non-finite probability values."
    assert (
        (normal_probability >= 0.0) & (normal_probability <= 1.0)
    ).all(), "GMM predictions contain probability values outside [0, 1]."
    tumor_probability = 1.0 - normal_probability
    cell_coordinates = np.asarray(
        [parse_cell_path(path) for path in gmm_predictions["path"]],
        dtype=np.float64,
    )
    assert cell_coordinates.shape == (
        len(gmm_predictions),
        2,
    ), "Cell coordinate shape does not match prediction count."

    hard_slide_tumor_probability = float(
        np.mean(tumor_probability > tumor_probability_threshold)
    )
    soft_slide_tumor_probability = float(np.mean(tumor_probability))
    area_soft_slide_tumor_probability = _nearest_cell_area_mean_tumor_probability(
        image_shape=image_shape,
        cell_y=cell_coordinates[:, 0],
        cell_x=cell_coordinates[:, 1],
        tumor_probability=tumor_probability,
    )

    return {
        "cell_count": int(len(tumor_probability)),
        "image_width": int(image_shape[1]),
        "image_height": int(image_shape[0]),
        "tumor_probability_threshold": float(tumor_probability_threshold),
        "hard_slide_tumor_probability": round(hard_slide_tumor_probability, 6),
        "percent_cells_tumor_probability_gt_0_5": round(
            hard_slide_tumor_probability * 100.0,
            6,
        ),
        "soft_slide_tumor_probability": round(soft_slide_tumor_probability, 6),
        "area_soft_slide_tumor_probability": round(
            area_soft_slide_tumor_probability,
            6,
        ),
        "area_pixel_count": int(image_shape[0] * image_shape[1]),
    }


def export_slide_portal_assets(
    *,
    slide_id: str,
    mosaic_dicom_path: str,
    detected_cells: pd.DataFrame,
    gmm_predictions: pd.DataFrame,
    portal_dir: str,
    slide_statistics: dict | None = None,
    strip_padding: int = 50,
    tile_size: int = 256,
    overlap: int = 0,
) -> dict[str, str]:
    logger.info("Exporting portal metadata for %s", slide_id)
    portal_path = _ensure_dir(portal_dir)
    mosaic_rgb = load_mosaic_image(
        mosaic_dicom_path=mosaic_dicom_path,
        strip_padding=strip_padding,
    )

    cells_payload = build_portal_cell_payload(
        detected_cells=detected_cells,
        gmm_predictions=gmm_predictions,
    )
    if slide_statistics is None:
        slide_statistics = build_slide_statistics(
            gmm_predictions=gmm_predictions,
            image_shape=mosaic_rgb.shape,
        )
    required_slide_statistics = {
        "hard_slide_tumor_probability",
        "soft_slide_tumor_probability",
        "area_soft_slide_tumor_probability",
    }
    missing_slide_statistics = required_slide_statistics.difference(slide_statistics)
    assert not missing_slide_statistics, (
        "Slide statistics are missing required portal keys: "
        f"{sorted(missing_slide_statistics)}"
    )
    cells_path = portal_path / "cells.json"
    with open(cells_path, "w", encoding="utf-8") as fd:
        json.dump(cells_payload, fd, separators=(",", ":"))
    assert cells_path.exists(), f"Failed to write portal cell payload: {cells_path}"

    manifest = {
        "slide_id": slide_id,
        "image_width": int(mosaic_rgb.shape[1]),
        "image_height": int(mosaic_rgb.shape[0]),
        "cell_count": int(cells_payload["cell_count"]),
        "num_clusters": int(cells_payload["num_clusters"]),
        "tile_size": int(tile_size),
        "overlap": int(overlap),
        "cells": {
            "path": cells_path.name,
            "score_label": "Tumor likelihood",
            "dot_color_label": "Normal-green to tumor-red",
        },
        "slide_statistics": slide_statistics,
    }
    manifest_path = portal_path / "slide_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fd:
        json.dump(manifest, fd, indent=2)
    assert manifest_path.exists(), f"Failed to write portal manifest: {manifest_path}"

    logger.info("Portal metadata for %s written under %s", slide_id, portal_path)
    return {
        "portal_dir": str(portal_path),
        "manifest_path": str(manifest_path),
        "cells_path": str(cells_path),
    }
