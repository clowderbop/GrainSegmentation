
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from common.instance_metric_bundle import INSTANCE_METRIC_BUNDLE_KEYS

INSTANCE_METRIC_KEYS: tuple[str, ...] = INSTANCE_METRIC_BUNDLE_KEYS

SCHEMA_VERSION = 2


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
    metrics: dict[str, float | int],
    empty_gt: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "sample_id": sample_id,
        "empty_gt": bool(empty_gt),
    }
    for key in INSTANCE_METRIC_KEYS:
        value = metrics[key]
        row[key] = int(value) if key.endswith("_count") else float(value)
    if extra:
        for k, v in extra.items():
            if k in row:
                raise ValueError(f"extra key {k!r} clashes with built-in row field")
            row[k] = v
    return row


def patch_aggregate_grainy_key(metric_key: str) -> str:
    return f"mean_{metric_key}_grainy"


def patch_aggregate_weighted_key(metric_key: str) -> str:
    return f"mean_{metric_key}_weighted"


def patch_aggregate_extra_keys(
    metric_keys: tuple[str, ...] = INSTANCE_METRIC_KEYS,
) -> tuple[str, ...]:
    """Extra report keys written for patch unit instance evaluations."""
    keys: list[str] = ["n_patches", "n_empty_gt"]
    for metric_key in metric_keys:
        keys.append(patch_aggregate_grainy_key(metric_key))
        keys.append(patch_aggregate_weighted_key(metric_key))
    return tuple(keys)


def compute_patch_metric_aggregates(
    rows: list[dict[str, Any]],
    *,
    metric_keys: tuple[str, ...] = INSTANCE_METRIC_KEYS,
) -> dict[str, float | int]:
    """Unweighted and grain-weighted means over grain-bearing patches for the bundle."""
    n_patches = len(rows)
    grainy = [row for row in rows if not row.get("empty_gt")]
    n_empty_gt = n_patches - len(grainy)

    def _finite_metric_values(
        subset: list[dict[str, Any]], key: str
    ) -> list[tuple[dict[str, Any], float]]:
        pairs: list[tuple[dict[str, Any], float]] = []
        for row in subset:
            if key not in row:
                continue
            value = float(row[key])
            if np.isfinite(value):
                pairs.append((row, value))
        return pairs

    def _mean_for_key(
        subset: list[dict[str, Any]], key: str, *, weight_by_gt: bool
    ) -> float:
        pairs = _finite_metric_values(subset, key)
        if not pairs:
            return float("nan")
        if weight_by_gt:
            total_weight = sum(float(row.get("gt_instance_count", 0)) for row, _ in pairs)
            if total_weight <= 0:
                return float("nan")
            return float(
                sum(value * float(row.get("gt_instance_count", 0)) for row, value in pairs)
                / total_weight
            )
        return float(np.mean([value for _, value in pairs]))

    agg: dict[str, float | int] = {
        "n_patches": n_patches,
        "n_empty_gt": n_empty_gt,
    }
    for key in metric_keys:
        agg[patch_aggregate_grainy_key(key)] = _mean_for_key(
            grainy, key, weight_by_gt=False
        )
        agg[patch_aggregate_weighted_key(key)] = _mean_for_key(
            grainy, key, weight_by_gt=True
        )
    return agg


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


def build_instance_eval_report(
    *,
    model_type: str,
    variant: str | None,
    unit: str,
    samples: list[dict[str, Any]],
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model_type": model_type,
        "metric_kind": "instance",
        "variant": variant,
        "unit": unit,
        "samples": samples,
    }
    if len(samples) > 1:
        report["mean"] = aggregate_mean_metrics(samples)
    if unit == "patch" and samples:
        patch_agg = compute_patch_metric_aggregates(samples)
        report.setdefault("extras", {})
        report["extras"].update(patch_agg)
    if extras:
        report.setdefault("extras", {})
        report["extras"].update(extras)
    return report
