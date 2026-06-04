"""Tests for common.evaluate_instances."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from analysis.load_metrics import load_instance_metrics_json
from common.evaluate_instances import collect_manifest_samples, evaluate_instance_samples
from common.prediction_set import prediction_set_path, save_prediction_set
from common.tests.evaluate_instances_fixtures import (
    empty_gt_false_positive_eval_sample,
    perfect_match_eval_sample,
    write_blank_image,
)
from common.tests.prediction_set_fixtures import yolo_prediction_set_from_masks


def test_collect_manifest_samples_patch(tmp_path: Path) -> None:
    image_path = tmp_path / "patch001_PPL.tif"
    write_blank_image(image_path, 64, 64)
    ps = yolo_prediction_set_from_masks(
        masks_hw=np.ones((1, 64, 64), dtype=np.float32),
        scores=np.array([0.9], dtype=np.float32),
        height=64,
        width=64,
    )
    pred_path = prediction_set_path(tmp_path, "patch001")
    save_prediction_set(pred_path, ps)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "variant": "PPL",
                "unit": "patch",
                "grainseg_root": str(tmp_path),
                "path_base": "work_root",
                "samples": [
                    {
                        "sample_id": "patch001",
                        "image": str(image_path),
                        "instance_prediction_set": str(pred_path),
                        "gt_txt": str(tmp_path / "patch001.txt"),
                        "gt_origin": "patch_stem",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "patch001.txt").write_text("0 0 0 0 0 0 0 0\n", encoding="utf-8")

    samples = collect_manifest_samples(manifest_path)
    assert len(samples) == 1
    assert samples[0].sample_id == "patch001"


def test_evaluate_two_synthetic_instance_maps(tmp_path: Path) -> None:
    report = evaluate_instance_samples(
        [perfect_match_eval_sample(tmp_path, sample_id="sample")],
        model_type="yolo",
        variant="PPL",
        unit="patch",
    )
    assert report["samples"][0]["pq"] == 1.0


def test_empty_ground_truth_false_positive_writes_valid_instance_metrics_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty GT with predictions must serialize pred_gt_instance_ratio as JSON null."""
    from common.evaluate_instances import main

    sample = empty_gt_false_positive_eval_sample(tmp_path)
    output_json = tmp_path / "instance_metrics.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_instances",
            "--image",
            str(sample.image_path),
            "--instance-prediction-set",
            str(sample.instance_prediction_set),
            "--gt-txt",
            str(sample.gt_txt),
            "--output-json",
            str(output_json),
            "--model-type",
            "yolo",
            "--unit",
            "patch",
        ],
    )

    main()

    report = json.loads(output_json.read_text(encoding="utf-8"))
    row = report["samples"][0]
    assert row["empty_gt"] is True
    assert row["gt_instance_count"] == 0
    assert row["pred_instance_count"] == 1
    assert row["pred_gt_instance_ratio"] is None

    metrics = load_instance_metrics_json(report)
    assert metrics["pred_gt_instance_ratio"] == float("inf")
