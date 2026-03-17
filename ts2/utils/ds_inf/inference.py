import logging
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torchvision
from tqdm.auto import tqdm

from ts2.utils.ds_inf.model import get_model

DEFAULT_CLASSES = ["na", "nuclei", "cyto", "rbc", "mp"]
logger = logging.getLogger(__name__)


def preprocess_srh_batch(
    images: torch.Tensor | np.ndarray,
    subtracted_base: float = 5000.0,
) -> torch.Tensor:
    """Convert a batch of 2-channel SRH images into normalized 3-channel tensors."""
    logger.info("Preprocessing SRH batch")
    images = torch.as_tensor(images)
    assert (
        images.ndim == 4
    ), f"Expected 4D input `(N, 2, H, W)`, got shape {tuple(images.shape)}."
    assert images.shape[1] == 2, f"Expected 2 channels, got {images.shape[1]}."

    images = images.to(torch.float32)
    ch2 = images[:, 0, :, :]
    ch3 = images[:, 1, :, :]
    ch1 = ch3 - ch2 + subtracted_base
    stacked = torch.stack((ch1, ch2, ch3), dim=1)
    logger.info("Prepared normalized SRH batch with shape %s", tuple(stacked.shape))
    return (stacked / 65536.0).clamp_(0.0, 1.0)


def get_xform(subtracted_base: float = 5000.0):
    """Return a preprocessing callable in the style of the original pipeline."""

    def _transform(image, target=None):
        image_3ch = preprocess_srh_batch(
            torch.as_tensor(image).unsqueeze(0),
            subtracted_base=subtracted_base,
        ).squeeze(0)
        return image_3ch, target

    return _transform


def matrix_nms(
    seg_masks: torch.Tensor,
    cate_labels: torch.Tensor,
    cate_scores: torch.Tensor,
    kernel: str = "gaussian",
    sigma: float = 2.0,
    sum_masks: torch.Tensor | None = None,
) -> torch.Tensor:
    """Matrix NMS applied to instance masks."""
    del cate_labels

    n_samples = len(cate_scores)
    if n_samples == 0:
        return cate_scores

    if sum_masks is None:
        sum_masks = seg_masks.sum((1, 2)).float()

    seg_masks = seg_masks.reshape(n_samples, -1).float()
    inter_matrix = torch.mm(seg_masks, seg_masks.transpose(1, 0))
    sum_masks_x = sum_masks.expand(n_samples, n_samples)
    iou_matrix = (
        inter_matrix / (sum_masks_x + sum_masks_x.transpose(1, 0) - inter_matrix)
    ).triu(diagonal=1)

    compensate_iou, _ = iou_matrix.max(0)
    compensate_iou = compensate_iou.expand(n_samples, n_samples).transpose(1, 0)
    decay_iou = iou_matrix

    if kernel == "gaussian":
        decay_matrix = torch.exp(-1 * sigma * (decay_iou**2))
        compensate_matrix = torch.exp(-1 * sigma * (compensate_iou**2))
        decay_coefficient, _ = (decay_matrix / compensate_matrix).min(0)
    elif kernel == "linear":
        decay_matrix = (1 - decay_iou) / (1 - compensate_iou)
        decay_coefficient, _ = decay_matrix.min(0)
    else:
        raise NotImplementedError(f"Unsupported kernel: {kernel}")

    return cate_scores * decay_coefficient


def score_threshold_with_matrix_nms(
    output: dict[str, torch.Tensor],
    confidence_threshold: float = 0.2,
    mask_confidence: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Apply matrix NMS, confidence filtering, and mask binarization."""
    output = {key: value.detach().cpu() for key, value in output.items()}
    required_keys = {"scores", "masks", "labels", "boxes"}
    missing_keys = required_keys.difference(output)
    assert not missing_keys, f"Missing required output keys: {sorted(missing_keys)}"

    if len(output["scores"]) == 0:
        output["masks"] = output["masks"].to(torch.uint8).squeeze(1)
        return output

    output["scores"] = matrix_nms(
        output["masks"].squeeze(1),
        output["labels"],
        output["scores"],
    )

    keep_by_score = torch.argwhere(output["scores"] > confidence_threshold).squeeze(-1)
    keep_by_label = torch.where(output["labels"] > 0)[0]
    keep = torch.from_numpy(
        np.intersect1d(keep_by_score.numpy(), keep_by_label.numpy())
    )

    for key in output:
        output[key] = torch.index_select(output[key], 0, keep)

    output["masks"] = (output["masks"] > mask_confidence).to(torch.uint8).squeeze(1)
    return output


def compute_centroids(batch_mask: torch.Tensor) -> torch.Tensor:
    """Compute `(row, col)` centroids from a mask tensor of shape `(N, H, W)`."""
    assert (
        batch_mask.ndim == 3
    ), f"Expected mask shape `(N, H, W)`, got {tuple(batch_mask.shape)}."

    batch_size, height, width = batch_mask.shape
    device = batch_mask.device

    y_coords = (
        torch.arange(height, device=device)
        .view(1, height, 1)
        .expand(batch_size, height, width)
    )
    x_coords = (
        torch.arange(width, device=device)
        .view(1, 1, width)
        .expand(batch_size, height, width)
    )

    mask_sum = batch_mask.sum(dim=(1, 2)).clamp(min=1e-6)
    x_centroids = (batch_mask * x_coords).sum(dim=(1, 2)) / mask_sum
    y_centroids = (batch_mask * y_coords).sum(dim=(1, 2)) / mask_sum
    return torch.stack((y_centroids, x_centroids), dim=1)


def package_cells_one_patch(
    result: dict[str, torch.Tensor],
    patch_name: str,
    celltypes: Sequence[str],
) -> list[dict[str, float | str]]:
    """Convert one patch prediction into lightweight metadata rows."""
    required_keys = {"boxes", "labels", "scores", "masks"}
    missing_keys = required_keys.difference(result)
    assert not missing_keys, f"Missing required result keys: {sorted(missing_keys)}"
    if len(result["boxes"]) == 0:
        return []

    bbox_cxcywh = torchvision.ops.box_convert(
        boxes=result["boxes"],
        in_fmt="xyxy",
        out_fmt="cxcywh",
    )
    centroids = compute_centroids(result["masks"])

    return [
        {
            "patch": patch_name,
            "celltype": celltypes[int(label.item())],
            "score": score.item(),
            "bbox_cx": bbox[0].item(),
            "bbox_cy": bbox[1].item(),
            "bbox_w": bbox[2].item(),
            "bbox_h": bbox[3].item(),
            "centroid_r": centroid[0].item(),
            "centroid_c": centroid[1].item(),
        }
        for label, score, bbox, centroid in zip(
            result["labels"],
            result["scores"],
            bbox_cxcywh,
            centroids,
        )
    ]


def run_inference(
    images: torch.Tensor | np.ndarray,
    model: torch.nn.Module,
    classes: Sequence[str],
    patch_names: Iterable[str] | None = None,
    confidence_threshold: float = 0.2,
    mask_confidence: float = 0.5,
    subtracted_base: float = 5000.0,
    batch_size: int = 8,
) -> list[dict[str, float | str]]:
    """Run batched inference using an already-loaded model."""
    logger.info("Starting cell inference")
    processed = preprocess_srh_batch(images, subtracted_base=subtracted_base)
    device = next(model.parameters()).device
    image_batches = torch.split(processed, batch_size)
    assert len(image_batches) > 0, "No image batches were created for inference."
    logger.info(
        "Running inference for %d images in %d batches on %s",
        processed.shape[0],
        len(image_batches),
        device,
    )

    outputs: list[dict[str, torch.Tensor]] = []
    with torch.inference_mode():
        for image_batch in tqdm(
            image_batches,
            total=len(image_batches),
            desc="Cell detection",
        ):
            batch_outputs = model(image_batch.to(device))
            outputs.extend(
                score_threshold_with_matrix_nms(
                    output,
                    confidence_threshold=confidence_threshold,
                    mask_confidence=mask_confidence,
                )
                for output in batch_outputs
            )
    assert (
        len(outputs) == processed.shape[0]
    ), f"Expected {processed.shape[0]} outputs, got {len(outputs)}."

    if patch_names is None:
        patch_names_list = [f"patch_{idx:05d}" for idx in range(len(outputs))]
    else:
        patch_names_list = list(patch_names)
        assert len(patch_names_list) == len(
            outputs
        ), f"Expected {len(outputs)} patch names, got {len(patch_names_list)}."

    rows: list[dict[str, float | str]] = []
    for patch_name, output in tqdm(
        zip(patch_names_list, outputs),
        total=len(outputs),
        desc="Packaging detections",
    ):
        rows.extend(package_cells_one_patch(output, patch_name, classes))
    logger.info("Finished cell inference with %d packaged detections", len(rows))
    return rows
