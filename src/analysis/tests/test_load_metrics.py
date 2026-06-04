"""Normalize eval JSON into metric rows for reporting tables."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.load_metrics import (
    IncompleteInstanceMetricBundleError,
    instance_metric_row,
    load_instance_metrics_json,
    metrics_table_from_runs,
)
from analysis.discover import EvalRunRef
from common.instance_metric_bundle import INSTANCE_METRIC_BUNDLE_KEYS

PQ_SAMPLE_ROW = {
    "sample_id": "test",
    "pq": 0.42,
    "dq": 0.5,
    "sq": 0.84,
    "precision_iou50": 0.5,
    "recall_iou50": 0.4,
    "f1_iou50": 0.43,
    "precision_iou75": 0.3,
    "recall_iou75": 0.2,
    "f1_iou75": 0.25,
    "mP_iou50_95": 0.3,
    "mR_iou50_95": 0.2,
    "mF1_iou50_95": 0.27,
    "gt_instance_count": 10,
    "pred_instance_count": 8,
    "pred_gt_instance_ratio": 0.8,
    "aji_plus": 0.13,
    "empty_gt": False,
}

WHOLE_SINGLE_SAMPLE = {
    "schema_version": 2,
    "model_type": "yolo",
    "variant": "PPL",
    "unit": "whole",
    "samples": [PQ_SAMPLE_ROW],
}

PATCH_WITH_MEAN = {
    **WHOLE_SINGLE_SAMPLE,
    "unit": "patch",
    "mean": {key: PQ_SAMPLE_ROW[key] for key in INSTANCE_METRIC_BUNDLE_KEYS},
    "extras": {
        "n_patches": 2,
        "n_empty_gt": 0,
        "mean_pq_grainy": 0.40,
        "mean_pq_weighted": 0.41,
        "mean_aji_plus_grainy": 0.12,
    },
}


def test_load_instance_metrics_reads_pq_bundle_from_single_sample() -> None:
    metrics = load_instance_metrics_json(WHOLE_SINGLE_SAMPLE)
    assert metrics["pq"] == pytest.approx(0.42)
    assert metrics["dq"] == pytest.approx(0.5)
    assert metrics["sq"] == pytest.approx(0.84)
    assert metrics["pred_gt_instance_ratio"] == pytest.approx(0.8)
    assert metrics["aji_plus"] == pytest.approx(0.13)
    assert "aji" not in metrics


def test_load_instance_metrics_prefers_mean_block() -> None:
    metrics = load_instance_metrics_json(PATCH_WITH_MEAN)
    assert metrics["pq"] == pytest.approx(0.42)


EMPTY_GT_FALSE_POSITIVE_ROW = {
    "sample_id": "patch_fp",
    "pq": 0.0,
    "dq": 0.0,
    "sq": 0.0,
    "precision_iou50": 0.0,
    "recall_iou50": 0.0,
    "f1_iou50": 0.0,
    "precision_iou75": 0.0,
    "recall_iou75": 0.0,
    "f1_iou75": 0.0,
    "mP_iou50_95": 0.0,
    "mR_iou50_95": 0.0,
    "mF1_iou50_95": 0.0,
    "gt_instance_count": 0,
    "pred_instance_count": 2,
    "pred_gt_instance_ratio": None,
    "aji_plus": 0.0,
    "empty_gt": True,
}

EMPTY_GT_FALSE_POSITIVE_REPORT = {
    "schema_version": 2,
    "model_type": "yolo",
    "variant": "PPL",
    "unit": "patch",
    "samples": [EMPTY_GT_FALSE_POSITIVE_ROW],
}


def test_load_accepts_null_pred_gt_ratio_for_empty_gt_false_positive() -> None:
    metrics = load_instance_metrics_json(EMPTY_GT_FALSE_POSITIVE_REPORT)
    assert metrics["gt_instance_count"] == 0
    assert metrics["pred_instance_count"] == 2
    assert metrics["pred_gt_instance_ratio"] == float("inf")


def test_metrics_table_from_runs_accepts_serialized_empty_gt_false_positive(
    tmp_path: Path,
) -> None:
    metrics_path = tmp_path / "instance_metrics.json"
    metrics_path.write_text(json.dumps(EMPTY_GT_FALSE_POSITIVE_REPORT), encoding="utf-8")
    runs = [
        EvalRunRef(
            producer="yolo",
            variant="PPL",
            unit="patch",
            instance_metrics_path=metrics_path,
        )
    ]
    table = metrics_table_from_runs(runs)
    assert table.iloc[0]["pred_gt_instance_ratio"] == float("inf")


def test_load_rejects_finite_pred_gt_ratio_for_empty_gt_false_positive() -> None:
    malformed = {
        **EMPTY_GT_FALSE_POSITIVE_REPORT,
        "samples": [
            {
                **EMPTY_GT_FALSE_POSITIVE_ROW,
                "pred_gt_instance_ratio": 0.5,
            }
        ],
    }
    with pytest.raises(
        IncompleteInstanceMetricBundleError,
        match="must be \\+inf when gt_instance_count is 0",
    ):
        load_instance_metrics_json(malformed)


def test_load_rejects_pre_policy_aji_only_report() -> None:
    stale = {
        "samples": [{"sample_id": "test", "aji": 0.2, "f1_iou50": 0.3, "empty_gt": False}],
    }
    with pytest.raises(IncompleteInstanceMetricBundleError, match="pre-policy"):
        load_instance_metrics_json(stale)


def test_load_rejects_incomplete_bundle() -> None:
    partial = {
        "samples": [
            {
                **PQ_SAMPLE_ROW,
                "pq": 0.42,
                "dq": 0.5,
            }
        ],
    }
    del partial["samples"][0]["sq"]
    with pytest.raises(IncompleteInstanceMetricBundleError, match="missing fields"):
        load_instance_metrics_json(partial)


def test_metrics_table_from_runs_rejects_stale_report(tmp_path: Path) -> None:
    metrics_path = tmp_path / "instance_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "samples": [
                    {"sample_id": "test", "aji": 0.1, "f1_iou50": 0.2, "empty_gt": False}
                ]
            }
        ),
        encoding="utf-8",
    )
    runs = [
        EvalRunRef(
            producer="yolo",
            variant="PPL",
            unit="whole",
            instance_metrics_path=metrics_path,
        )
    ]
    with pytest.raises(IncompleteInstanceMetricBundleError, match="pre-policy"):
        metrics_table_from_runs(runs)


def test_instance_metric_row_includes_display_name() -> None:
    row = instance_metric_row(
        producer="yolo",
        variant="PPL+AllPPX",
        unit="whole",
        metrics={"pq": 0.5, "dq": 0.6, "sq": 0.7},
    )
    assert row["display_name"] == "FullStack"
    assert row["pq"] == pytest.approx(0.5)


def test_metrics_table_from_runs_exposes_pq_bundle_columns(tmp_path: Path) -> None:
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
    row = table.iloc[0]
    assert row["producer"] == "yolo"
    assert row["display_name"] == "PPL"
    assert row["pq"] == pytest.approx(0.42)
    assert row["dq"] == pytest.approx(0.5)
    assert row["pred_gt_instance_ratio"] == pytest.approx(0.8)


def test_metrics_table_omits_mask_ap_from_instance_rows(tmp_path: Path) -> None:
    metrics_path = tmp_path / "instance_metrics.json"
    metrics_path.write_text(json.dumps(WHOLE_SINGLE_SAMPLE), encoding="utf-8")
    mask_ap_path = tmp_path / "mask_ap_metrics.json"
    mask_ap_path.write_text(
        json.dumps({"mean_coco_mask_ap": {"mean_AP50": 0.9, "mean_AP": 0.8}}),
        encoding="utf-8",
    )
    runs = [
        EvalRunRef(
            producer="yolo",
            variant="PPL",
            unit="whole",
            instance_metrics_path=metrics_path,
            mask_ap_metrics_path=mask_ap_path,
        )
    ]
    table = metrics_table_from_runs(runs)
    assert "mask_ap50" not in table.columns
    assert "mask_ap" not in table.columns


def test_patch_rows_include_pq_patch_aggregates(tmp_path: Path) -> None:
    metrics_path = tmp_path / "instance_metrics.json"
    metrics_path.write_text(json.dumps(PATCH_WITH_MEAN), encoding="utf-8")
    runs = [
        EvalRunRef(
            producer="yolo",
            variant="PPL",
            unit="patch",
            instance_metrics_path=metrics_path,
        )
    ]
    table = metrics_table_from_runs(runs)
    assert table.iloc[0]["mean_pq_grainy"] == pytest.approx(0.40)
