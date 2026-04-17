from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path

import matplotlib.cm as cm
import numpy as np
import pandas as pd
from PIL import Image

from ts2.utils.silica_sc_eval.generate_gmm_visualization import (
    load_mosaic_image,
    parse_assignment_vector,
)

logger = logging.getLogger(__name__)

try:
    import pyvips
except Exception:  # pragma: no cover - optional runtime dependency
    pyvips = None


_LANCZOS = (
    Image.Resampling.LANCZOS
    if hasattr(Image, "Resampling")
    else Image.LANCZOS
)


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

    assignment_vectors = [parse_assignment_vector(value) for value in merged["assignment"]]
    assignment_matrix = np.stack(assignment_vectors, axis=0)
    dominant_cluster = np.argmax(assignment_matrix, axis=1).astype(np.int32)
    dominant_cluster_confidence = assignment_matrix[
        np.arange(len(assignment_matrix)),
        dominant_cluster,
    ]

    normal_score = merged["prediction"].astype(float).to_numpy(dtype=np.float64)
    tumor_score = 1.0 - normal_score
    tumor_score_display = (
        100 - np.clip(np.floor(normal_score * 100.0), 0, 99).astype(np.int32)
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
        ).astype(np.int32).tolist(),
        "y": np.round(
            merged["global_centroid_r"].astype(float).to_numpy(dtype=np.float64)
        ).astype(np.int32).tolist(),
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

def _write_dzi_descriptor(
    descriptor_path: Path,
    width: int,
    height: int,
    tile_size: int,
    overlap: int,
    tile_format: str,
) -> None:
    descriptor = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<Image TileSize="{tile_size}" Overlap="{overlap}" Format="{tile_format}" '
        'xmlns="http://schemas.microsoft.com/deepzoom/2008">\n'
        f'  <Size Width="{width}" Height="{height}"/>\n'
        "</Image>\n"
    )
    descriptor_path.write_text(descriptor, encoding="utf-8")
    assert descriptor_path.exists(), f"Failed to write DZI descriptor: {descriptor_path}"


def _save_tile_pyramid_with_pil(
    image: Image.Image,
    output_prefix: Path,
    tile_size: int,
    overlap: int,
    jpeg_quality: int,
    tile_format: str = "jpg",
) -> tuple[str, str]:
    width, height = image.size
    max_level = int(math.ceil(math.log2(max(width, height)))) if max(width, height) > 1 else 0
    descriptor_path = output_prefix.with_suffix(".dzi")
    files_dir = output_prefix.parent / f"{output_prefix.name}_files"
    _ensure_dir(files_dir)
    _write_dzi_descriptor(
        descriptor_path=descriptor_path,
        width=width,
        height=height,
        tile_size=tile_size,
        overlap=overlap,
        tile_format=tile_format,
    )

    current_image = image.copy()
    current_level = max_level
    while True:
        level_dir = files_dir / str(current_level)
        _ensure_dir(level_dir)
        level_width, level_height = current_image.size
        num_cols = int(math.ceil(level_width / tile_size))
        num_rows = int(math.ceil(level_height / tile_size))

        for row in range(num_rows):
            for col in range(num_cols):
                left = col * tile_size
                top = row * tile_size
                right = min(left + tile_size, level_width)
                bottom = min(top + tile_size, level_height)
                crop_left = max(0, left - (overlap if col > 0 else 0))
                crop_top = max(0, top - (overlap if row > 0 else 0))
                crop_right = min(
                    level_width,
                    right + (overlap if col < num_cols - 1 else 0),
                )
                crop_bottom = min(
                    level_height,
                    bottom + (overlap if row < num_rows - 1 else 0),
                )
                tile = current_image.crop((crop_left, crop_top, crop_right, crop_bottom))
                tile_path = level_dir / f"{col}_{row}.{tile_format}"
                tile.save(tile_path, format="JPEG", quality=jpeg_quality)

        if current_level == 0:
            break
        next_size = (
            max(1, int(math.ceil(level_width / 2.0))),
            max(1, int(math.ceil(level_height / 2.0))),
        )
        current_image = current_image.resize(next_size, _LANCZOS)
        current_level -= 1

    return descriptor_path.name, files_dir.name


def _save_tile_pyramid_with_pyvips(
    image: Image.Image,
    output_prefix: Path,
    tile_size: int,
    overlap: int,
    jpeg_quality: int,
) -> tuple[str, str]:
    assert pyvips is not None, "pyvips is not available."
    rgb = np.asarray(image, dtype=np.uint8)
    assert rgb.ndim == 3 and rgb.shape[2] == 3, (
        f"Expected RGB image array, got shape {rgb.shape}"
    )
    vips_image = pyvips.Image.new_from_memory(
        rgb.tobytes(),
        rgb.shape[1],
        rgb.shape[0],
        rgb.shape[2],
        format="uchar",
    )
    vips_image.dzsave(
        str(output_prefix),
        tile_size=tile_size,
        overlap=overlap,
        layout="dz",
        suffix=f".jpg[Q={jpeg_quality}]",
    )
    descriptor_path = output_prefix.with_suffix(".dzi")
    files_dir = output_prefix.parent / f"{output_prefix.name}_files"
    assert descriptor_path.exists(), f"Missing DZI descriptor after pyvips export: {descriptor_path}"
    assert files_dir.is_dir(), f"Missing DZI tile directory after pyvips export: {files_dir}"
    return descriptor_path.name, files_dir.name


def save_image_as_dzi(
    image: Image.Image,
    output_prefix: str | os.PathLike[str],
    tile_size: int = 256,
    overlap: int = 0,
    jpeg_quality: int = 88,
) -> tuple[str, str]:
    output_prefix = Path(output_prefix)
    _ensure_dir(output_prefix.parent)
    image = image.convert("RGB")
    if pyvips is not None:
        logger.info("Saving DZI with pyvips: %s", output_prefix)
        return _save_tile_pyramid_with_pyvips(
            image=image,
            output_prefix=output_prefix,
            tile_size=tile_size,
            overlap=overlap,
            jpeg_quality=jpeg_quality,
        )

    logger.warning(
        "pyvips is unavailable; falling back to PIL DZI export for %s. "
        "This is correct but slower for large slides.",
        output_prefix,
    )
    return _save_tile_pyramid_with_pil(
        image=image,
        output_prefix=output_prefix,
        tile_size=tile_size,
        overlap=overlap,
        jpeg_quality=jpeg_quality,
    )


def export_slide_dzi_assets(
    *,
    slide_id: str,
    mosaic_dicom_path: str,
    dzi_dir: str,
    strip_padding: int = 50,
    tile_size: int = 256,
    overlap: int = 0,
    jpeg_quality: int = 88,
) -> dict[str, str]:
    logger.info("Exporting DZI assets for %s", slide_id)
    dzi_path = _ensure_dir(dzi_dir)
    mosaic_rgb = load_mosaic_image(
        mosaic_dicom_path=mosaic_dicom_path,
        strip_padding=strip_padding,
    )
    color_image = Image.fromarray(mosaic_rgb)

    color_dzi_name, color_tiles_dir = save_image_as_dzi(
        image=color_image,
        output_prefix=dzi_path / "color",
        tile_size=tile_size,
        overlap=overlap,
        jpeg_quality=jpeg_quality,
    )

    logger.info("DZI assets for %s written under %s", slide_id, dzi_path)
    return {
        "dzi_dir": str(dzi_path),
        "color_dzi_path": str(dzi_path / color_dzi_name),
        "color_tiles_dir": str(dzi_path / color_tiles_dir),
    }


def export_slide_portal_assets(
    *,
    slide_id: str,
    mosaic_dicom_path: str,
    detected_cells: pd.DataFrame,
    gmm_predictions: pd.DataFrame,
    portal_dir: str,
    strip_padding: int = 50,
    tile_size: int = 256,
    overlap: int = 0,
    jpeg_quality: int = 88,
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
        "base_layers": {
            "color": {
                "label": "Full Color",
                "dzi": "color.dzi",
                "tiles_dir": "color_files",
            },
        },
        "cells": {
            "path": cells_path.name,
            "score_label": "Tumor likelihood",
            "dot_color_label": "Normal-green to tumor-red",
        },
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
