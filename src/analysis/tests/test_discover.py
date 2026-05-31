"""Eval artifact discovery on a grainseg root (path conventions v1)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from analysis.discover import (
    DiscoveryError,
    UltralyticsValRef,
    discover_eval_runs,
    discover_ultralytics_val,
    latest_patch_job_dir,
)

MINIMAL_INSTANCE_METRICS = {
    "schema_version": 1,
    "model_type": "yolo",
    "unit": "whole",
    "samples": [],
}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _touch_file(path: Path, *, mtime: float) -> None:
    os.utime(path, (mtime, mtime))


@pytest.fixture
def grainseg_tree(tmp_path: Path) -> Path:
    root = tmp_path / "GrainSeg"
    for variant in ("PPL", "PPL+AllPPX"):
        _write_json(
            root / f"eval/yolo_{variant}/instance_metrics.json",
            {**MINIMAL_INSTANCE_METRICS, "variant": variant},
        )
        _write_json(
            root / f"eval/unet_test/run_unet_finetuned_{variant}/instance_metrics.json",
            {**MINIMAL_INSTANCE_METRICS, "model_type": "unet", "variant": variant},
        )
    older_metrics = root / "eval/yolo_patches/PPL/100/instance_metrics.json"
    newer_metrics = root / "eval/yolo_patches/PPL/200/instance_metrics.json"
    _write_json(older_metrics, MINIMAL_INSTANCE_METRICS)
    _write_json(newer_metrics, MINIMAL_INSTANCE_METRICS)
    now = time.time()
    _touch_file(older_metrics, mtime=now - 100)
    _touch_file(newer_metrics, mtime=now)
    _write_json(
        root / "runs/yolo26-seg-val/PPL/test/metrics.json",
        {"seg": {"map50": 0.5}},
    )
    return root


def test_latest_patch_job_dir_picks_newest(grainseg_tree: Path) -> None:
    variant_dir = grainseg_tree / "eval/yolo_patches/PPL"
    assert latest_patch_job_dir(variant_dir).name == "200"


def test_discover_eval_runs_whole_and_patch(grainseg_tree: Path) -> None:
    runs = discover_eval_runs(grainseg_tree)
    yolo_whole = [r for r in runs if r.producer == "yolo" and r.unit == "whole"]
    unet_whole = [r for r in runs if r.producer == "unet" and r.unit == "whole"]
    assert {r.variant for r in yolo_whole} == {"PPL", "PPL+AllPPX"}
    assert {r.variant for r in unet_whole} == {"PPL", "PPL+AllPPX"}
    patch = next(r for r in runs if r.producer == "yolo" and r.unit == "patch")
    assert patch.variant == "PPL"
    assert patch.patch_job_dir is not None
    assert patch.patch_job_dir.name == "200"


def test_discover_ultralytics_val(grainseg_tree: Path) -> None:
    refs = discover_ultralytics_val(grainseg_tree)
    assert refs == [
        UltralyticsValRef(
            variant="PPL",
            metrics_path=grainseg_tree / "runs/yolo26-seg-val/PPL/test/metrics.json",
        )
    ]


def test_discover_raises_when_whole_run_missing(tmp_path: Path) -> None:
    root = tmp_path / "GrainSeg"
    with pytest.raises(DiscoveryError, match="yolo whole"):
        discover_eval_runs(root, strict=True)
