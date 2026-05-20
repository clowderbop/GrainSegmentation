
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np


INSTANCE_METRIC_KEYS: tuple[str, ...] = (
    "aji",
    "precision_iou50",
    "recall_iou50",
    "f1_iou50",
    "precision_iou75",
    "recall_iou75",
    "f1_iou75",
    "mP_iou50_95",
    "mR_iou50_95",
    "mF1_iou50_95",
)

SCHEMA_VERSION = 1


def json_safe_for_dump(value: Any) -> Any:
    if isinstance(value, np.generic):
        return json_safe_for_dump(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return json_safe_for_dump(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe_for_dump(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_for_dump(v) for v in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def count_instances(instance_map: np.ndarray) -> int:
    return int(np.sum(np.unique(instance_map) != 0))


def build_sample_row(
    sample_id: str,
    *,
    metrics: dict[str, float],
    gt_instances: int,
    pred_instances: int,
    empty_gt: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "sample_id": sample_id,
        "gt_instances": int(gt_instances),
        "pred_instances": int(pred_instances),
        "empty_gt": bool(empty_gt),
    }
    for key in INSTANCE_METRIC_KEYS:
        row[key] = float(metrics[key])
    if extra:
        for k, v in extra.items():
            if k in row:
                raise ValueError(f"extra key {k!r} clashes with built-in row field")
            row[k] = v
    return row


def aggregate_mean_metrics(
    rows: list[dict[str, Any]],
    keys: tuple[str, ...] = INSTANCE_METRIC_KEYS,
) -> dict[str, float]:
    mean: dict[str, float] = {}
    for key in keys:
        values: list[float] = []
        for row in rows:
            if key not in row:
                continue
            v = row[key]
            if isinstance(v, bool) or isinstance(v, (dict, list)):
                continue
            if isinstance(v, (int, float, np.floating, np.integer)):
                fv = float(v)
                if np.isfinite(fv):
                    values.append(fv)
        mean[key] = float(np.mean(values)) if values else float("nan")
    return mean


def build_legacy_flat_dict(
    rows: list[dict[str, Any]],
    *,
    mean: dict[str, float] | None,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for row in rows:
        sid = str(row["sample_id"])
        out[sid] = {k: float(row[k]) for k in INSTANCE_METRIC_KEYS}
    if mean is not None:
        out["mean"] = {k: float(v) for k, v in mean.items()}
    return out


def build_instance_eval_report(
    *,
    model_type: str,
    variant: str | None,
    unit: str,
    samples: list[dict[str, Any]],
    extras: dict[str, Any] | None = None,
    include_legacy_flat: bool = True,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model_type": model_type,
        "variant": variant,
        "unit": unit,
        "samples": samples,
    }
    mean_block: dict[str, float] | None = None
    if len(samples) > 1:
        mean_block = aggregate_mean_metrics(samples)
        report["mean"] = mean_block
    merged_extras: dict[str, Any] = dict(extras) if extras else {}
    if include_legacy_flat:
        leg = dict(merged_extras.get("legacy") or {})
        leg["per_sample_flat"] = build_legacy_flat_dict(samples, mean=mean_block)
        merged_extras["legacy"] = leg
    if merged_extras:
        report["extras"] = merged_extras
    return report
