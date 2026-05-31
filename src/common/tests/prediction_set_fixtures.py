"""Test helpers for building YOLO instance prediction sets (small fixtures only)."""

from __future__ import annotations

import numpy as np
import pytest

from common.mask_ops import masks_hw_to_binary, resize_mask_nearest
from common.prediction_set import (
    GRAIN_CLASS_ID,
    PredictionSet,
    binary_mask_to_segmentation,
    segmentation_to_binary_mask,
)


def assert_instance_map_partitions_equal(
    actual: np.ndarray, expected: np.ndarray
) -> None:
    """Same foreground partition (label ids may differ)."""
    np.testing.assert_array_equal(actual.shape, expected.shape)
    np.testing.assert_array_equal(actual > 0, expected > 0)


def _normalize_rle(segmentation: dict) -> dict:
    counts = segmentation["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("utf-8")
    return {"size": list(segmentation["size"]), "counts": counts}


def assert_yolo_canonical_sets_equal(
    actual: PredictionSet, expected: PredictionSet
) -> None:
    """Match canonical YOLO grains by score and RLE (order-independent)."""
    assert actual.producer == expected.producer == "yolo"
    assert (actual.height, actual.width) == (expected.height, expected.width)
    assert len(actual.detections) == len(expected.detections)

    def sort_key(det: dict) -> tuple:
        mask = segmentation_to_binary_mask(det["segmentation"])
        return (-float(det["score"]), -int(mask.sum()), mask.tobytes())

    for actual_det, expected_det in zip(
        sorted(actual.detections, key=sort_key),
        sorted(expected.detections, key=sort_key),
    ):
        assert float(actual_det["score"]) == pytest.approx(float(expected_det["score"]))
        assert _normalize_rle(actual_det["segmentation"]) == _normalize_rle(
            expected_det["segmentation"]
        )


def yolo_prediction_set_from_masks(
    masks_hw: np.ndarray,
    scores: np.ndarray,
    *,
    height: int,
    width: int,
) -> PredictionSet:
    """Build overlapping YOLO detector proposals from per-mask planes (tests only)."""
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
