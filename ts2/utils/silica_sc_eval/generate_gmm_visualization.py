import os

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom


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


def load_mosaic_image(
    mosaic_dicom_path: str,
    strip_padding: int = 50,
) -> np.ndarray:
    image = pydicom.dcmread(mosaic_dicom_path).pixel_array
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


def generate_visualization(
    predictions: pd.DataFrame,
    mosaic_dicom_path: str,
    out_dir: str,
    candidate_name: str,
    strip_padding: int = 50,
    circle_radius: int = 10,
    alpha: float = 0.5,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    if "prediction" not in predictions.columns or "path" not in predictions.columns:
        raise ValueError("Predictions CSV must contain `path` and `prediction` columns.")

    norm_pred = np.asarray(predictions["prediction"], dtype=float)
    cell_coords = np.asarray([parse_cell_path(path) for path in predictions["path"]])

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
    fig.savefig(os.path.join(out_dir, f"{candidate_name}-hist.pdf"))
    plt.close(fig)

    image = load_mosaic_image(mosaic_dicom_path, strip_padding=strip_padding)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    overlay = image_bgr.copy()
    cmap = plt.get_cmap("RdYlGn")

    for (y, x), p in zip(cell_coords, norm_pred):
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
    cv2.imwrite(os.path.join(out_dir, f"{candidate_name}.png"), blended)
