"""Normalize eval JSON into metric rows for reporting tables."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.load_metrics import (
    instance_metric_row,
    load_instance_metrics_json,
    metrics_table_from_runs,
)
from analysis.discover import EvalRunRef

WHOLE_SINGLE_SAMPLE = {
    "schema_version": 1,
    "model_type": "yolo",
    "variant": "PPL",
    "unit": "whole",
    "samples": [
        {
            "sample_id": "test",
            "aji": 0.13,
            "f1_iou50": 0.43,
            "precision_iou50": 0.5,
            "recall_iou50": 0.4,
            "precision_iou75": 0.3,
            "recall_iou75": 0.2,
            "f1_iou75": 0.25,
            "mP_iou50_95": 0.3,
            "mR_iou50_95": 0.2,
            "mF1_iou50_95": 0.27,
            "gt_instances": 10,
            "predicted_grain_count": 8,
            "empty_gt": False,
        }
    ],
}

PATCH_WITH_MEAN = {
    **WHOLE_SINGLE_SAMPLE,
    "unit": "patch",
    "mean": {"aji": 0.16, "f1_iou50": 0.40},
    "extras": {
        "mean_aji_grainy": 0.10,
        "mean_f1_iou50_grainy": 0.36,
    },
}


def test_load_instance_metrics_uses_sample_when_no_mean() -> None:
    metrics = load_instance_metrics_json(WHOLE_SINGLE_SAMPLE)
    assert metrics["aji"] == pytest.approx(0.13)
    assert metrics["f1_iou50"] == pytest.approx(0.43)


def test_load_instance_metrics_prefers_mean_block() -> None:
    metrics = load_instance_metrics_json(PATCH_WITH_MEAN)
    assert metrics["aji"] == pytest.approx(0.16)


def test_instance_metric_row_includes_display_name() -> None:
    row = instance_metric_row(
        producer="yolo",
        variant="PPL+AllPPX",
        unit="whole",
        metrics={"aji": 0.5, "f1_iou50": 0.7},
    )
    assert row["display_name"] == "FullStack"
    assert row["aji"] == pytest.approx(0.5)


def test_metrics_table_from_runs(tmp_path: Path) -> None:
    metrics_path = tmp_path / "instance_metrics.json"
    metrics_path.write_text(json.dumps(WHOLE_SINGLE_SAMPLE), encoding="utf-8")
    runs = [
        EvalRunRef(
            producer="yolo",
            variant="PPL",
            unit="whole",
            instance_metrics_path=metrics_path,
        )
    ]
    table = metrics_table_from_runs(runs)
    assert len(table) == 1
    assert table.iloc[0]["producer"] == "yolo"
    assert table.iloc[0]["display_name"] == "PPL"
