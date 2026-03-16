from __future__ import annotations

from typing import Any

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_model_instance_segmentation(
    num_classes: int,
) -> torchvision.models.detection.MaskRCNN:
    """Create the Mask R-CNN architecture used by the original code."""
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        256,
        num_classes,
    )
    return model


def _extract_state_dict(ckpt: dict[str, Any]) -> dict[str, torch.Tensor]:
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    return {key.removeprefix("model."): value for key, value in state_dict.items()}


def load_model_from_checkpoint(
    checkpoint_path: str,
    num_classes: int,
    device: str | torch.device | None = None,
) -> torchvision.models.detection.MaskRCNN:
    """Load a Lightning-style checkpoint or raw state_dict into the inference model."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = get_model_instance_segmentation(num_classes=num_classes)
    model.load_state_dict(_extract_state_dict(checkpoint))
    model.to(device)
    model.eval()
    return model


def get_model(
    checkpoint_path: str,
    num_classes: int,
    device: str | torch.device | None = None,
) -> torchvision.models.detection.MaskRCNN:
    """Mirror the original inference helper."""
    return load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        num_classes=num_classes,
        device=device,
    )
