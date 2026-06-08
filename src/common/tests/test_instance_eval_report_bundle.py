"""Instance evaluation reports expose the shared PQ metric bundle (issue 02)."""

from __future__ import annotations

import pytest

from common.evaluate_instances import evaluate_instance_samples
from common.instance_metric_bundle import (
    INSTANCE_METRIC_BUNDLE_INT_KEYS,
    INSTANCE_METRIC_BUNDLE_KEYS,
)
from common.reporting import (
    INSTANCE_METRIC_KEYS,
    patch_aggregate_extra_keys,
    patch_aggregate_grainy_key,
    patch_aggregate_weighted_key,
)
from common.tests.evaluate_instances_fixtures import perfect_match_eval_sample


def test_whole_section_sample_row_includes_full_metric_bundle(tmp_path) -> None:
    """INTENT: evaluate_instance_samples whole-section rows expose every instance metric bundle field."""
    report = evaluate_instance_samples(
        [perfect_match_eval_sample(tmp_path)],
        model_type="yolo",
        variant="PPL",
        unit="whole",
    )
    row = report["samples"][0]
    assert INSTANCE_METRIC_KEYS[0] == "pq"
    for key in INSTANCE_METRIC_BUNDLE_KEYS:
        assert key in row
    assert row["pq"] == pytest.approx(1.0)
    assert row["dq"] == pytest.approx(1.0)
    assert row["sq"] == pytest.approx(1.0)
    assert row["f1_iou50"] == pytest.approx(1.0)
    assert row["mF1_iou50_95"] == pytest.approx(1.0)
    assert row["pred_gt_instance_ratio"] == pytest.approx(1.0)
    assert row["aji_plus"] == pytest.approx(1.0)
    assert row["tp"] == 1
    assert row["fp"] == 0
    assert row["fn"] == 0
    assert row["gt_instance_count"] == 1
    assert row["pred_instance_count"] == 1
    assert row["empty_gt"] is False
    assert "aji" not in row


def test_patch_report_includes_full_bundle_aggregates(tmp_path) -> None:
    """INTENT: patch evaluate_instance_samples reports per-sample bundles and grainy/weighted extras."""
    samples = [
        perfect_match_eval_sample(tmp_path, sample_id="patch1"),
        perfect_match_eval_sample(tmp_path, sample_id="patch2"),
    ]
    report = evaluate_instance_samples(
        samples,
        model_type="yolo",
        variant="PPL",
        unit="patch",
    )
    for row in report["samples"]:
        for key in INSTANCE_METRIC_BUNDLE_KEYS:
            assert key in row

    extras = report["extras"]
    assert set(extras.keys()) >= set(patch_aggregate_extra_keys())
    assert extras["n_patches"] == 2
    assert extras["n_empty_gt"] == 0
    for key in INSTANCE_METRIC_BUNDLE_KEYS:
        if key in ("fp", "fn"):
            expected = 0.0
        elif key in INSTANCE_METRIC_BUNDLE_INT_KEYS:
            expected = 1.0
        else:
            expected = 1.0
        assert extras[patch_aggregate_grainy_key(key)] == pytest.approx(expected)
        assert extras[patch_aggregate_weighted_key(key)] == pytest.approx(expected)
    assert extras[patch_aggregate_grainy_key("dq")] == pytest.approx(1.0)
    assert extras[patch_aggregate_grainy_key("mF1_iou50_95")] == pytest.approx(1.0)
    assert extras[
        patch_aggregate_weighted_key("pred_gt_instance_ratio")
    ] == pytest.approx(1.0)
