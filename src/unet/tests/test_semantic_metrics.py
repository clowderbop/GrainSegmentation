"""Tests for U-Net semantic metrics."""

from __future__ import annotations

import numpy as np

from unet.semantic_metrics import compute_semantic_metrics_dict, per_class_iou


def test_per_class_iou_known_confusion() -> None:
    """INTENT: per-class IoU and semantic metrics dict match hand-computed values on a known confusion matrix."""
    gt = np.array(
        [
            [0, 0, 1, 1],
            [0, 2, 1, 1],
            [2, 2, 1, 0],
        ],
        dtype=np.int32,
    )
    pred = np.array(
        [
            [0, 1, 1, 1],
            [0, 2, 2, 1],
            [2, 2, 0, 0],
        ],
        dtype=np.int32,
    )
    ious = per_class_iou(pred, gt)
    assert np.isclose(ious[0], 3.0 / 5.0)
    assert np.isclose(ious[1], 0.5)
    assert np.isclose(ious[2], 3.0 / 4.0)

    metrics = compute_semantic_metrics_dict(pred, gt)
    assert np.isclose(metrics["pixel_accuracy"], 0.75)
    assert np.isclose(metrics["iou_class_0"], ious[0])
    assert np.isclose(metrics["mean_iou"], np.mean(list(ious.values())))
