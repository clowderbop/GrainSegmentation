"""Semantic segmentation metrics for U-Net 3-class label maps."""

from __future__ import annotations

from typing import Any

import numpy as np

from common.image_io import validate_semantic_labels
from common.reporting import aggregate_mean_metrics, json_safe_for_dump

SEMANTIC_CLASS_LABELS = (0, 1, 2)
SEMANTIC_METRIC_KEYS = (
    "pixel_accuracy",
    "mean_iou",
    "iou_class_0",
    "iou_class_1",
    "iou_class_2",
)


def per_class_iou(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    class_ids: tuple[int, ...] = SEMANTIC_CLASS_LABELS,
) -> dict[int, float]:
    pred_v = validate_semantic_labels(pred, "prediction")
    gt_v = validate_semantic_labels(gt, "ground_truth")
    if pred_v.shape != gt_v.shape:
        raise ValueError(
            f"pred shape {pred_v.shape} does not match gt shape {gt_v.shape}"
        )
    out: dict[int, float] = {}
    for class_id in class_ids:
        pred_mask = pred_v == class_id
        gt_mask = gt_v == class_id
        inter = float(np.logical_and(pred_mask, gt_mask).sum())
        union = float(np.logical_or(pred_mask, gt_mask).sum())
        out[class_id] = inter / union if union > 0 else float("nan")
    return out


def pixel_accuracy(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_v = validate_semantic_labels(pred, "prediction")
    gt_v = validate_semantic_labels(gt, "ground_truth")
    if pred_v.shape != gt_v.shape:
        raise ValueError(
            f"pred shape {pred_v.shape} does not match gt shape {gt_v.shape}"
        )
    total = pred_v.size
    if total == 0:
        return 1.0
    return float((pred_v == gt_v).sum()) / float(total)


def compute_semantic_metrics_dict(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    class_ious = per_class_iou(pred, gt)
    values = [v for v in class_ious.values() if np.isfinite(v)]
    mean_iou = float(np.mean(values)) if values else float("nan")
    return {
        "pixel_accuracy": pixel_accuracy(pred, gt),
        "mean_iou": mean_iou,
        "iou_class_0": float(class_ious[0]),
        "iou_class_1": float(class_ious[1]),
        "iou_class_2": float(class_ious[2]),
    }


def build_semantic_sample_row(
    sample_id: str, metrics: dict[str, float]
) -> dict[str, Any]:
    row: dict[str, Any] = {"sample_id": sample_id}
    for key in SEMANTIC_METRIC_KEYS:
        row[key] = float(metrics[key])
    return row


def build_semantic_eval_report(
    *,
    variant: str | None,
    unit: str,
    samples: list[dict[str, Any]],
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": 1,
        "model_type": "unet",
        "metric_kind": "semantic",
        "variant": variant,
        "unit": unit,
        "samples": samples,
    }
    if len(samples) > 1:
        report["mean"] = aggregate_mean_metrics(samples, keys=SEMANTIC_METRIC_KEYS)
    if extras:
        report["extras"] = json_safe_for_dump(extras)
    return report
