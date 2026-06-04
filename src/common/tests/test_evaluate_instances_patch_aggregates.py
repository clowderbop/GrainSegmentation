"""Patch instance eval reports include patch metric aggregates in extras."""

from __future__ import annotations

from pathlib import Path

from common.evaluate_instances import evaluate_instance_samples
from common.reporting import patch_aggregate_grainy_key
from common.tests.evaluate_instances_fixtures import perfect_match_eval_sample


def test_patch_eval_report_includes_grainy_aggregates(tmp_path: Path) -> None:
    samples = [
        perfect_match_eval_sample(tmp_path, sample_id="patch1", width=32, height=32),
        perfect_match_eval_sample(tmp_path, sample_id="patch2", width=32, height=32),
    ]
    report = evaluate_instance_samples(
        samples,
        model_type="yolo",
        variant="PPL",
        unit="patch",
    )
    extras = report["extras"]
    assert extras["n_patches"] == 2
    assert extras[patch_aggregate_grainy_key("pq")] == 1.0
    assert extras[patch_aggregate_grainy_key("dq")] == 1.0
    assert extras[patch_aggregate_grainy_key("f1_iou50")] == 1.0
