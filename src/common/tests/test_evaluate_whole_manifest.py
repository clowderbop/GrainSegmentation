"""Tests for whole-unit evaluate_instances manifest requirements."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from common.evaluate_instances import _resolve_eval_samples, collect_manifest_samples
from common.prediction_set import prediction_set_path, save_prediction_set


def _write_blank(path: Path, w: int = 16, h: int = 16) -> None:
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    tifffile.imwrite(path, rgb, photometric="rgb")


def test_whole_requires_manifest(tmp_path: Path) -> None:
    """INTENT: _resolve_eval_samples requires --manifest when unit is whole."""

    class Args:
        unit = "whole"
        manifest = None
        image = None
        instance_prediction_set = None
        prediction_set_dir = None
        gt_gpkg = None
        gt_txt = None
        gt_origin = None
        sample_id = None

    import argparse
    from typing import cast

    with pytest.raises(ValueError, match="Provide --manifest"):
        _resolve_eval_samples(cast(argparse.Namespace, Args()))


def test_whole_accepts_manifest(tmp_path: Path) -> None:
    """INTENT: _resolve_eval_samples loads whole-section samples from a valid eval manifest."""
    image = tmp_path / "train_PPL.tif"
    _write_blank(image)
    pred = prediction_set_path(tmp_path, "train")
    save_prediction_set(
        pred,
        {
            "schema_version": 1,
            "height": 16,
            "width": 16,
            "producer": "unet",
            "detections": [],
        },
    )
    manifest_path = tmp_path / "eval.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "variant": "PPL",
                "unit": "whole",
                "grainseg_root": str(tmp_path),
                "path_base": "work_root",
                "samples": [
                    {
                        "sample_id": "train",
                        "image": str(image),
                        "instance_prediction_set": str(pred),
                        "gt_gpkg": str(tmp_path / "labels.gpkg"),
                        "gt_origin": "whole_image",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "labels.gpkg").write_text("", encoding="utf-8")

    class Args:
        unit = "whole"
        manifest = manifest_path
        image = None
        instance_prediction_set = None
        prediction_set_dir = None
        gt_gpkg = None
        gt_txt = None
        gt_origin = None
        sample_id = None

    import argparse
    from typing import cast

    samples = _resolve_eval_samples(cast(argparse.Namespace, Args()))
    assert len(samples) == 1
    assert samples[0].sample_id == "train"


def test_collect_manifest_samples_rejects_image_outside_work_root(
    tmp_path: Path,
) -> None:
    """INTENT: collect_manifest_samples rejects manifest image paths outside the staged work root."""
    work_root = tmp_path / "run"
    work_root.mkdir()
    external_image = tmp_path / "external.tif"
    _write_blank(external_image)
    gt = work_root / "labels.gpkg"
    gt.write_text("", encoding="utf-8")
    pred = prediction_set_path(work_root, "test")
    save_prediction_set(
        pred,
        {
            "schema_version": 1,
            "height": 16,
            "width": 16,
            "producer": "yolo",
            "detections": [],
        },
    )
    manifest_path = work_root / "eval.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "variant": "PPL",
                "unit": "whole",
                "grainseg_root": str(work_root),
                "path_base": "work_root",
                "samples": [
                    {
                        "sample_id": "test",
                        "image": "../external.tif",
                        "instance_prediction_set": "prediction_sets/test.json",
                        "gt_gpkg": "labels.gpkg",
                        "gt_origin": "whole_image",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="staged work"):
        collect_manifest_samples(manifest_path)
