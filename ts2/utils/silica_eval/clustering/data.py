import base64
import io
import logging
import os
import re
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm

from ts2.data.transforms import HistologyTransform
from ts2.train.main_cell_inference import SingleCellListInferenceDataset
from ts2.utils.srh_viz import prepare_three_channel_viz_image

logger = logging.getLogger(__name__)


def _parse_proposal(value: Any) -> tuple[int, int]:
    if isinstance(value, tuple):
        return int(value[0]), int(value[1])
    if isinstance(value, list):
        return int(value[0]), int(value[1])
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("(") and stripped.endswith(")"):
            stripped = stripped[1:-1]
        parts = [part.strip() for part in stripped.split(",")]
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
    raise ValueError(f"Could not parse proposal coordinates from value: {value!r}")


def _encode_cell_path(patch: str, proposal: Any) -> str:
    row, col = _parse_proposal(proposal)
    return f"{patch}#{row}_{col}"


def _extract_patient_from_patch_name(patch: str) -> str:
    patch = str(patch)
    match = re.search(r"(NIO_UM_[0-9]+)", patch)
    assert match is not None, f"Could not infer patient id from patch name: {patch}"
    return match.group(1)


def load_embedding_table(
    pred_path: str,
    cell_instances_path: str,
    label_column: str = "patch_type",
) -> pd.DataFrame:
    logger.info("Loading embedding table from %s", pred_path)

    data = torch.load(pred_path)
    table = pd.DataFrame(data)
    assert (
        "embeddings" in table.columns
    ), "Expected `embeddings` column in embedding table."
    assert "path" in table.columns, "Expected `path` column in embedding table."

    label_source = pd.read_csv(cell_instances_path, dtype=str)
    label_source = label_source.copy()
    label_source["path"] = label_source.apply(
        lambda row: _encode_cell_path(row["patch"], row["proposal"]), axis=1
    )
    if "patient" not in label_source.columns:
        label_source["patient"] = label_source["patch"].map(
            _extract_patient_from_patch_name
        )
    label_source = label_source.rename(columns={label_column: "label"})[
        ["path", "label", "patient"]
    ]
    labels_per_path = label_source.groupby("path")["label"].nunique()
    ambiguous_paths = labels_per_path[labels_per_path > 1]
    assert (
        ambiguous_paths.empty
    ), "Cell instances CSV produced paths with conflicting labels: " + ", ".join(
        ambiguous_paths.index[:10]
    )
    label_source = label_source.drop_duplicates(subset=["path", "label"])

    table = table.drop(columns=["label"], errors="ignore").merge(
        label_source,
        on="path",
        how="left",
        validate="many_to_one",
    )
    missing_label_count = int(table["label"].isna().sum())
    assert (
        missing_label_count == 0
    ), f"Could not match {missing_label_count} prediction rows to labels in source CSV."
    total_patient_count = int(table["patient"].astype(str).nunique())
    logger.info("Total patients in embedding table: %d", total_patient_count)
    logger.info("Loaded %d rows with columns: %s", len(table), table.columns.tolist())
    return table


def sample_idx(group: pd.DataFrame, n: int = 8192) -> pd.Index:
    return group.sample(n=min(n, len(group)), random_state=1000).index


def sample_cells(db_data: pd.DataFrame, n_per_class: int = 8192) -> list[int]:
    logger.info("Sampling up to %d cells per label", n_per_class)
    sampled_idx = (
        db_data.groupby("label").apply(sample_idx, n=n_per_class).explode().values
    )
    cell_samples = sorted(sampled_idx)
    assert len(cell_samples) > 0, "No sampled cells were selected."
    logger.info("Sampled %d total cells", len(cell_samples))
    return cell_samples


def sample_global_cells(db_data: pd.DataFrame, n: int | None = None) -> list[int]:
    assert n > 0, f"global sample n must be positive, got {n}"
    sampled_idx = db_data.sample(n=min(n, len(db_data)), random_state=1000).index
    cell_samples = sorted(sampled_idx)
    assert len(cell_samples) > 0, "No sampled cells were selected for global sample."
    logger.info("Sampled %d total cells for global sample", len(cell_samples))
    return cell_samples


def im_to_bytestr(image: np.ndarray) -> str:
    output = io.BytesIO()
    Image.fromarray(image).save(output, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode()


def build_dataset(
    config_path: str, cell_instances: str
) -> SingleCellListInferenceDataset:
    logger.info(
        "Building dataset from config=%s cell_instances=%s", config_path, cell_instances
    )
    assert os.path.exists(config_path), f"Dataset config not found: {config_path}"
    assert os.path.exists(
        cell_instances
    ), f"Cell instances file not found: {cell_instances}"
    with open(config_path) as fd:
        cf = OmegaConf.create(yaml.safe_load(fd))
    cf.data.test_dataset.params.cell_instances = cell_instances
    dataset = SingleCellListInferenceDataset(
        transform=HistologyTransform(**cf.data.xform_params),
        **cf.data.test_dataset.params,
    )
    assert len(dataset) > 0, "Dataset instantiated with zero cells."
    logger.info("Dataset size: %d", len(dataset))
    return dataset


def get_sample_images(
    dataset: SingleCellListInferenceDataset, cell_samples: list[int]
) -> tuple[list, list[str], np.ndarray]:
    logger.info("Loading sampled cell images for TSNE tooltips")
    images = [
        dataset[i]["image"]
        for i in tqdm(cell_samples, desc="Loading sampled cell images")
    ]
    assert len(images) == len(
        cell_samples
    ), "Mismatch between sampled cells and loaded images."
    image_arrays = [
        prepare_three_channel_viz_image(image.squeeze()).numpy() for image in images
    ]
    im_str = [im_to_bytestr(image) for image in image_arrays]
    stacked = np.stack(image_arrays)
    assert (
        stacked.ndim == 4
    ), f"Expected stacked image array to be 4D, got shape {stacked.shape}"
    logger.info(
        "Loaded %d sampled images; stacked image array shape: %s",
        len(images),
        stacked.shape,
    )
    return images, im_str, stacked

