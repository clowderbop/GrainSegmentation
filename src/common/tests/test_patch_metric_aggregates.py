"""Patch metric aggregates for instance eval reports (ADR 0003)."""

from __future__ import annotations

import pytest

from common.reporting import compute_patch_metric_aggregates


def test_patch_metric_aggregates_grainy_and_weighted_means() -> None:
    rows = [
        {"aji": 0.2, "f1_iou50": 0.3, "empty_gt": False, "gt_instances": 10},
        {"aji": 0.8, "f1_iou50": 0.9, "empty_gt": False, "gt_instances": 30},
        {"aji": 0.0, "f1_iou50": 0.0, "empty_gt": True, "gt_instances": 0},
    ]
    agg = compute_patch_metric_aggregates(rows)
    assert agg["n_patches"] == 3
    assert agg["n_empty_gt"] == 1
    assert agg["mean_aji_grainy"] == pytest.approx(0.5)
    assert agg["mean_f1_iou50_grainy"] == pytest.approx(0.6)
    assert agg["mean_aji_weighted"] == pytest.approx(0.65)
    assert agg["mean_f1_iou50_weighted"] == pytest.approx(0.75)
