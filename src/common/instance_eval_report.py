"""Read instance metric bundles from evaluate_instances JSON reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from common.instance_metric_bundle import INSTANCE_METRIC_BUNDLE_KEYS

TRAIN_WHOLE_SECTION_SAMPLE_IDS: tuple[str, ...] = ("train",)


def _metric_values_from_report_samples(
    report: dict[str, Any], metric_key: str
) -> list[float]:
    samples = report.get("samples")
    if not isinstance(samples, list):
        return []
    return [
        float(row[metric_key])
        for row in samples
        if isinstance(row, dict) and metric_key in row
    ]


def extract_metric_from_report(report: dict[str, Any], metric_key: str) -> float:
    mean = report.get("mean")
    if isinstance(mean, dict) and metric_key in mean:
        return float(mean[metric_key])
    values = _metric_values_from_report_samples(report, metric_key)
    if values:
        return float(sum(values) / len(values))
    raise ValueError(
        f"instance metrics report has no {metric_key!r} in mean or samples"
    )


def extract_instance_metric_bundle_from_report(
    report: dict[str, Any],
) -> dict[str, float]:
    return {
        key: extract_metric_from_report(report, key) for key in INSTANCE_METRIC_BUNDLE_KEYS
    }


def validate_train_whole_section_report(
    report: dict[str, Any],
    *,
    expected_sample_ids: Sequence[str] = TRAIN_WHOLE_SECTION_SAMPLE_IDS,
    model_type: str | None = "unet",
) -> None:
    """Require a train whole-section instance evaluation report (ADR 0003)."""
    if report.get("metric_kind") != "instance":
        raise ValueError(
            f"report metric_kind must be 'instance', got {report.get('metric_kind')!r}"
        )
    if report.get("unit") != "whole":
        raise ValueError(
            f"train whole-section selection requires unit='whole', got {report.get('unit')!r}"
        )
    if model_type is not None and report.get("model_type") != model_type:
        raise ValueError(
            f"report model_type must be {model_type!r}, got {report.get('model_type')!r}"
        )
    samples = report.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("report must include a non-empty samples list")
    sample_ids = tuple(
        sorted(
            str(row["sample_id"])
            for row in samples
            if isinstance(row, dict) and "sample_id" in row
        )
    )
    expected = tuple(sorted(expected_sample_ids))
    if sample_ids != expected:
        raise ValueError(
            f"train whole-section report sample_ids {sample_ids!r} != expected {expected!r}"
        )


def load_instance_eval_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_train_whole_section_bundle(path: Path, *, model_type: str = "unet") -> dict[str, float]:
    report = load_instance_eval_report(path)
    validate_train_whole_section_report(report, model_type=model_type)
    return extract_instance_metric_bundle_from_report(report)


def mean_bundle_across_variants(
    per_variant_bundles: dict[str, dict[str, float]],
) -> dict[str, float]:
    if not per_variant_bundles:
        raise ValueError("per_variant_bundles must not be empty")
    bundles = list(per_variant_bundles.values())
    out: dict[str, float] = {}
    for key in INSTANCE_METRIC_BUNDLE_KEYS:
        if key.endswith("_count"):
            out[key] = int(round(float(np.mean([b[key] for b in bundles]))))
        else:
            out[key] = float(np.mean([float(b[key]) for b in bundles]))
    return out


def instance_metrics_report_path_for_variant(
    eval_output_dir: Path, variant: str
) -> Path:
    run_dir = eval_output_dir / f"run_unet_finetuned_{variant}.keras"
    return run_dir / "instance_metrics.json"


def load_train_whole_section_bundles_from_eval_dir(
    eval_output_dir: Path,
    variant_names: Sequence[str],
    *,
    model_type: str = "unet",
) -> dict[str, dict[str, float]]:
    per_variant: dict[str, dict[str, float]] = {}
    for variant in variant_names:
        path = instance_metrics_report_path_for_variant(eval_output_dir, variant)
        if not path.is_file():
            raise FileNotFoundError(
                f"missing train whole-section instance metrics for {variant}: {path}"
            )
        per_variant[variant] = load_train_whole_section_bundle(path, model_type=model_type)
    return per_variant
