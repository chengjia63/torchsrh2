import logging
import os

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def parse_cell_path(cell_path: str) -> tuple[int, int]:
    patch_name, cell_coord = cell_path.split("#", 1)
    patch_coord = patch_name.rsplit("-", 1)[1]
    patch_top_str, patch_left_str = patch_coord.split("_", 1)
    cell_r_str, cell_c_str = cell_coord.split("_", 1)

    patch_top = int(round(float(patch_top_str)))
    patch_left = int(round(float(patch_left_str)))
    cell_r = int(round(float(cell_r_str)))
    cell_c = int(round(float(cell_c_str)))
    return patch_top + cell_r, patch_left + cell_c


def parse_assignment_vector(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        vector = value.astype(np.float64, copy=False)
    elif isinstance(value, (list, tuple)):
        vector = np.asarray(value, dtype=np.float64)
    elif isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            stripped = stripped[1:-1].strip()
        if "," in stripped:
            parts = [part.strip() for part in stripped.split(",") if part.strip()]
        else:
            parts = [part for part in stripped.split() if part]
        vector = np.asarray([float(part) for part in parts], dtype=np.float64)
    else:
        raise TypeError(f"Unsupported assignment value type: {type(value)}")

    assert vector.ndim == 1, f"Expected 1D assignment vector, got shape {vector.shape}"
    assert vector.size > 0, "Assignment vector is empty."
    return vector


def load_mosaic_image(
    mosaic_dicom_path: str,
    strip_padding: int = 50,
) -> np.ndarray:
    logger.info("Loading mosaic image from %s", mosaic_dicom_path)
    assert os.path.exists(
        mosaic_dicom_path
    ), f"Mosaic DICOM not found: {mosaic_dicom_path}"
    image = pydicom.dcmread(mosaic_dicom_path).pixel_array
    assert image.ndim == 3, f"Expected 3D mosaic image, got shape {image.shape}"
    image = np.pad(
        image,
        ((strip_padding, strip_padding), (strip_padding, 0), (0, 0)),
    )[:, :-strip_padding, ...]

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 1)
        image = (image * 255).astype(np.uint8)
    if image.shape[-1] != 3:
        raise ValueError("Mosaic image must be RGB.")
    return image


def _get_score_text_style(score_text: str, circle_radius: int) -> tuple[float, int]:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = 1
    max_text_width = max(1, int(circle_radius * 3.2))
    max_text_height = max(1, int(circle_radius * 1.1))
    font_scale = 0.31

    for candidate_scale in (0.31, 0.29, 0.27, 0.25, 0.23, 0.21, 0.19, 0.17):
        (text_width, text_height), _ = cv2.getTextSize(
            score_text,
            font,
            candidate_scale,
            font_thickness,
        )
        if text_width <= max_text_width and text_height <= max_text_height:
            font_scale = candidate_scale
            break

    return font_scale, font_thickness


def _softmax_sharpen_score(p: float, temperature: float = 0.35) -> float:
    assert temperature > 0, f"Temperature must be positive, got {temperature}"
    p = float(np.clip(p, 1.0e-6, 1.0 - 1.0e-6))
    logit = np.log(p / (1.0 - p))
    sharpened_logit = logit / temperature
    return float(1.0 / (1.0 + np.exp(-sharpened_logit)))


def generate_visualization(
    predictions: pd.DataFrame,
    mosaic_dicom_path: str,
    out_dir: str,
    candidate_name: str,
    strip_padding: int = 50,
    circle_radius: int = 10,
    alpha: float = 0.5,
) -> None:
    logger.info("Generating visualization for %s", candidate_name)
    os.makedirs(out_dir, exist_ok=True)

    assert len(predictions) > 0, "No predictions provided for visualization."
    assert (
        "prediction" in predictions.columns
    ), "Predictions must contain a `prediction` column."
    assert "path" in predictions.columns, "Predictions must contain a `path` column."
    assert (
        "assignment" in predictions.columns
    ), "Predictions must contain an `assignment` column."

    norm_pred = np.asarray(predictions["prediction"], dtype=float)
    cell_coords = np.asarray([parse_cell_path(path) for path in predictions["path"]])
    assert len(norm_pred) == len(
        cell_coords
    ), "Coordinate count does not match prediction count."
    logger.info("Preparing histogram and overlay for %d cells", len(norm_pred))

    plt.rcParams["font.size"] = 14
    matplotlib.rcParams["pdf.fonttype"] = 42

    fig, ax = plt.subplots(1, 1)
    ax.hist(1 - norm_pred, color="#475569", zorder=2)
    ax.set_xlim(0, 1)
    ax.xaxis.set_ticks_position("none")
    ax.yaxis.set_ticks_position("none")
    ax.set_ylabel("Cells")
    ax.set_xlabel("Tumor likelihood")
    ax.grid(axis="both", color="#cbd5e1", zorder=0)
    plt.tight_layout()
    hist_path = os.path.join(out_dir, f"{candidate_name}-hist.pdf")
    fig.savefig(hist_path)
    plt.close(fig)
    assert os.path.exists(hist_path), f"Failed to save histogram: {hist_path}"

    image = load_mosaic_image(mosaic_dicom_path, strip_padding=strip_padding)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    overlay = image_bgr.copy()
    cmap = plt.get_cmap("RdYlGn")

    for (y, x), p in tqdm(
        zip(cell_coords, norm_pred),
        total=len(norm_pred),
        desc="Drawing GMM overlay",
    ):
        color = np.asarray(cmap(p)[:3]) * 255
        color_bgr = tuple(int(v) for v in color[::-1])
        cv2.circle(
            overlay,
            (int(x), int(y)),
            radius=circle_radius,
            color=color_bgr,
            thickness=-1,
        )

    blended = cv2.addWeighted(overlay, alpha, image_bgr, 1 - alpha, 0)
    image_path = os.path.join(out_dir, f"{candidate_name}.png")
    success = cv2.imwrite(image_path, blended)
    assert success, f"Failed to save overlay image: {image_path}"
    assert os.path.exists(image_path), f"Overlay image was not written: {image_path}"

    sharpened_overlay = image_bgr.copy()
    sharpened_pred = np.asarray(
        [_softmax_sharpen_score(p) for p in norm_pred],
        dtype=np.float64,
    )

    for (y, x), p in tqdm(
        zip(cell_coords, sharpened_pred),
        total=len(sharpened_pred),
        desc="Drawing sharpened GMM overlay",
    ):
        color = np.asarray(cmap(p)[:3]) * 255
        color_bgr = tuple(int(v) for v in color[::-1])
        cv2.circle(
            sharpened_overlay,
            (int(x), int(y)),
            radius=circle_radius,
            color=color_bgr,
            thickness=-1,
        )

    sharpened_blended = cv2.addWeighted(
        sharpened_overlay, alpha, image_bgr, 1 - alpha, 0
    )
    sharpened_image_path = os.path.join(out_dir, f"{candidate_name}-sharpened.png")
    success = cv2.imwrite(sharpened_image_path, sharpened_blended)
    assert success, f"Failed to save sharpened overlay image: {sharpened_image_path}"
    assert os.path.exists(
        sharpened_image_path
    ), f"Sharpened overlay image was not written: {sharpened_image_path}"

    grayscale_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    grayscale_image = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)
    white_mask = np.full_like(grayscale_image, 255)
    score_overlay = cv2.addWeighted(white_mask, 0.5, grayscale_image, 0.5, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for (y, x), p in tqdm(
        zip(cell_coords, norm_pred),
        total=len(norm_pred),
        desc="Drawing score text",
    ):
        score_value = 100 - int(np.clip(np.floor(p * 100), 0, 99))
        score_text = f"{score_value:02d}"
        font_scale, font_thickness = _get_score_text_style(score_text, circle_radius)
        (text_width, text_height), baseline = cv2.getTextSize(
            score_text,
            font,
            font_scale,
            font_thickness,
        )
        text_x = int(round(x - text_width / 2))
        text_y = int(round(y + text_height / 2))
        color = np.asarray(cmap(p)[:3]) * 255
        font_color = tuple(int(v) for v in color[::-1])
        cv2.putText(
            score_overlay,
            score_text,
            (text_x, text_y),
            font,
            font_scale,
            font_color,
            font_thickness,
            lineType=cv2.LINE_AA,
        )

    score_image_path = os.path.join(out_dir, f"{candidate_name}-score.png")
    success = cv2.imwrite(score_image_path, score_overlay)
    assert success, f"Failed to save score overlay image: {score_image_path}"
    assert os.path.exists(
        score_image_path
    ), f"Score overlay image was not written: {score_image_path}"

    assignment_vectors = [parse_assignment_vector(value) for value in predictions["assignment"]]
    assignment_matrix = np.stack(assignment_vectors, axis=0)
    dominant_clusters = np.argmax(assignment_matrix, axis=1)
    num_clusters = assignment_matrix.shape[1]
    cluster_cell_pct = (
        np.bincount(dominant_clusters, minlength=num_clusters).astype(np.float64)
        / len(dominant_clusters)
        * 100.0
    )

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.bar(
        np.arange(num_clusters),
        cluster_cell_pct,
        color="#64748b",
        edgecolor="#334155",
        linewidth=0.5,
        zorder=2,
    )
    ax.set_xlabel("Cluster")
    ax.set_ylabel("% of cells")
    ax.set_title("Cell fraction per cluster")
    ax.grid(axis="y", color="#cbd5e1", zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    cluster_pct_pdf_path = os.path.join(out_dir, f"{candidate_name}-cluster-pct.pdf")
    cluster_pct_png_path = os.path.join(out_dir, f"{candidate_name}-cluster-pct.png")
    fig.savefig(cluster_pct_pdf_path)
    fig.savefig(cluster_pct_png_path, dpi=200)
    plt.close(fig)
    assert os.path.exists(
        cluster_pct_pdf_path
    ), f"Failed to save cluster percentage PDF: {cluster_pct_pdf_path}"
    assert os.path.exists(
        cluster_pct_png_path
    ), f"Failed to save cluster percentage PNG: {cluster_pct_png_path}"

    mixture_overlay = cv2.addWeighted(white_mask, 0.5, grayscale_image, 0.5, 0)

    for (y, x), p, assignment in tqdm(
        zip(cell_coords, norm_pred, assignment_vectors),
        total=len(norm_pred),
        desc="Drawing mixture text",
    ):
        dominant_idx = int(np.argmax(assignment))
        dominant_pct = int(np.clip(np.floor(float(assignment[dominant_idx]) * 100), 0, 99))
        mixture_text = f"{dominant_idx:02d},{dominant_pct:02d}"
        font_scale, font_thickness = _get_score_text_style(
            mixture_text, circle_radius
        )
        (text_width, text_height), _ = cv2.getTextSize(
            mixture_text,
            font,
            font_scale,
            font_thickness,
        )
        text_x = int(round(x - text_width / 2))
        text_y = int(round(y + text_height / 2))
        color = np.asarray(cmap(p)[:3]) * 255
        font_color = tuple(int(v) for v in color[::-1])
        cv2.putText(
            mixture_overlay,
            mixture_text,
            (text_x, text_y),
            font,
            font_scale,
            font_color,
            font_thickness,
            lineType=cv2.LINE_AA,
        )

    mixture_image_path = os.path.join(out_dir, f"{candidate_name}-mixture.png")
    success = cv2.imwrite(mixture_image_path, mixture_overlay)
    assert success, f"Failed to save mixture overlay image: {mixture_image_path}"
    assert os.path.exists(
        mixture_image_path
    ), f"Mixture overlay image was not written: {mixture_image_path}"
    logger.info("Saved visualization outputs to %s", out_dir)
