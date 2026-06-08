"""Patch metric aggregates for instance eval reports (ADR 0003)."""

from __future__ import annotations

import pytest

from common.instance_metric_bundle import INSTANCE_METRIC_BUNDLE_KEYS
from common.reporting import (
    compute_patch_metric_aggregates,
    patch_aggregate_extra_keys,
    patch_aggregate_grainy_key,
    patch_aggregate_weighted_key,
)


def _bundle_row(
    *,
    pq: float,
    aji_plus: float,
    gt_instance_count: int,
    empty_gt: bool,
    dq: float | None = None,
    f1_iou50: float | None = None,
    pred_gt_instance_ratio: float = 1.0,
) -> dict:
    row = {
        key: pq
        for key in INSTANCE_METRIC_BUNDLE_KEYS
        if key
        not in (
            "tp",
            "fp",
            "fn",
            "gt_instance_count",
            "pred_instance_count",
            "pred_gt_instance_ratio",
            "aji_plus",
        )
    }
    row["dq"] = dq if dq is not None else pq
    row["sq"] = pq
    row["f1_iou50"] = f1_iou50 if f1_iou50 is not None else pq
    row["aji_plus"] = aji_plus
    pred_instance_count = gt_instance_count
    row["gt_instance_count"] = gt_instance_count
    row["pred_instance_count"] = pred_instance_count
    row["pred_gt_instance_ratio"] = pred_gt_instance_ratio
    if empty_gt or gt_instance_count == 0:
        row["tp"] = 0
        row["fp"] = pred_instance_count
        row["fn"] = gt_instance_count
    else:
        row["tp"] = gt_instance_count
        row["fp"] = pred_instance_count - gt_instance_count
        row["fn"] = 0
    row["empty_gt"] = empty_gt
    return row


def test_patch_metric_aggregates_cover_full_bundle() -> None:
    """INTENT: compute_patch_metric_aggregates emits grainy and GT-weighted means for all bundle metrics."""
    rows = [
        _bundle_row(pq=0.2, aji_plus=0.3, gt_instance_count=10, empty_gt=False),
        _bundle_row(
            pq=0.8,
            aji_plus=0.9,
            gt_instance_count=30,
            empty_gt=False,
            dq=0.7,
            f1_iou50=0.85,
        ),
        _bundle_row(pq=0.0, aji_plus=0.0, gt_instance_count=0, empty_gt=True),
    ]
    agg = compute_patch_metric_aggregates(rows)
    assert set(agg.keys()) == set(patch_aggregate_extra_keys())
    assert agg["n_patches"] == 3
    assert agg["n_empty_gt"] == 1
    assert agg[patch_aggregate_grainy_key("pq")] == pytest.approx(0.5)
    assert agg[patch_aggregate_grainy_key("aji_plus")] == pytest.approx(0.6)
    assert agg[patch_aggregate_grainy_key("dq")] == pytest.approx(0.45)
    assert agg[patch_aggregate_grainy_key("f1_iou50")] == pytest.approx(0.525)
    assert agg[patch_aggregate_weighted_key("pq")] == pytest.approx(0.65)
    assert agg[patch_aggregate_weighted_key("aji_plus")] == pytest.approx(0.75)
    assert agg[patch_aggregate_weighted_key("pred_gt_instance_ratio")] == pytest.approx(
        1.0
    )


def test_patch_metric_aggregates_skip_non_finite_values() -> None:
    """INTENT: compute_patch_metric_aggregates ignores non-finite pred_gt_instance_ratio values when averaging."""
    rows = [
        _bundle_row(
            pq=0.5,
            aji_plus=0.5,
            gt_instance_count=5,
            empty_gt=False,
            pred_gt_instance_ratio=float("inf"),
        ),
        _bundle_row(
            pq=1.0,
            aji_plus=1.0,
            gt_instance_count=5,
            empty_gt=False,
            pred_gt_instance_ratio=1.0,
        ),
    ]
    agg = compute_patch_metric_aggregates(rows)
    assert agg[patch_aggregate_grainy_key("pred_gt_instance_ratio")] == pytest.approx(
        1.0
    )
