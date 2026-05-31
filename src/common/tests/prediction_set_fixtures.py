"""Test helpers for building YOLO instance prediction sets (small fixtures only)."""

from __future__ import annotations

import numpy as np

from common.mask_ops import masks_hw_to_binary, resize_mask_nearest
from common.prediction_set import GRAIN_CLASS_ID, PredictionSet, binary_mask_to_segmentation


def yolo_prediction_set_from_masks(
    masks_hw: np.ndarray,
    scores: np.ndarray,
    *,
    height: int,
    width: int,
) -> PredictionSet:
    """Build a YOLO prediction set from per-mask planes (tests and small fixtures)."""
    if masks_hw.ndim != 3:
        raise ValueError(f"masks_hw must be (n, H, W), got {masks_hw.shape}")
    detections: list[dict] = []
    for index in range(masks_hw.shape[0]):
        binary = masks_hw_to_binary(masks_hw[index : index + 1])[0]
        if binary.shape != (height, width):
            binary = resize_mask_nearest(binary.astype(np.uint8), height, width).astype(bool)
        if not binary.any():
            continue
        detections.append(
            {
                "segmentation": binary_mask_to_segmentation(
                    binary, height=height, width=width
                ),
                "score": float(scores[index]),
                "category_id": GRAIN_CLASS_ID,
            }
        )
    return PredictionSet(
        schema_version=1,
        height=height,
        width=width,
        producer="yolo",
        detections=tuple(detections),
    )
