"""Tests for instance evaluation report bundle extraction."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from common.evaluate_instances import evaluate_instance_samples
from common.instance_eval_report import (
    TRAIN_WHOLE_SECTION_SAMPLE_IDS,
    extract_instance_metric_bundle_from_report,
    load_train_whole_section_bundle,
    mean_bundle_across_variants,
    validate_train_whole_section_report,
)
from common.instance_metric_bundle import INSTANCE_METRIC_BUNDLE_KEYS
from common.tests.evaluate_instances_fixtures import perfect_match_eval_sample


def test_extract_instance_metric_bundle_from_whole_report(tmp_path: Path) -> None:
    report = evaluate_instance_samples(
        [perfect_match_eval_sample(tmp_path, sample_id="train")],
        model_type="unet",
        variant="PPL",
        unit="whole",
    )
    bundle = extract_instance_metric_bundle_from_report(report)
    for key in INSTANCE_METRIC_BUNDLE_KEYS:
        assert key in bundle
    assert bundle["pq"] == pytest.approx(1.0)


def test_validate_train_whole_section_report_accepts_train_sample() -> None:
    report = {
        "metric_kind": "instance",
        "unit": "whole",
        "model_type": "unet",
        "samples": [{"sample_id": "train"}],
    }
    validate_train_whole_section_report(report)


def test_validate_train_whole_section_report_rejects_patch_unit() -> None:
    report = {
        "metric_kind": "instance",
        "unit": "patch",
        "model_type": "unet",
        "samples": [{"sample_id": "train"}],
    }
    with pytest.raises(ValueError, match="unit='whole'"):
        validate_train_whole_section_report(report)


def test_validate_train_whole_section_report_rejects_wrong_sample_ids() -> None:
    report = {
        "metric_kind": "instance",
        "unit": "whole",
        "model_type": "unet",
        "samples": [{"sample_id": "test"}],
    }
    with pytest.raises(ValueError, match="sample_ids"):
        validate_train_whole_section_report(
            report, expected_sample_ids=TRAIN_WHOLE_SECTION_SAMPLE_IDS
        )


def test_load_train_whole_section_bundle_validates_before_extract(
    tmp_path: Path,
) -> None:
    report = evaluate_instance_samples(
        [perfect_match_eval_sample(tmp_path, sample_id="train")],
        model_type="unet",
        variant="PPL",
        unit="patch",
    )
    path = tmp_path / "patch.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="unit='whole'"):
        load_train_whole_section_bundle(path)


def test_mean_bundle_across_variants_averages_numeric_fields() -> None:
    bundles = {
        variant: {key: 0.0 for key in INSTANCE_METRIC_BUNDLE_KEYS}
        for variant in ("PPL", "PPLPPXblend")
    }
    bundles["PPL"]["pq"] = 1.0
    bundles["PPLPPXblend"]["pq"] = 0.5

    mean = mean_bundle_across_variants(bundles)
    assert mean["pq"] == pytest.approx(0.75)
