"""Tests for whole-unit evaluate_instances manifest requirements."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from common.evaluate_instances import _resolve_eval_samples
from common.instance_predictions import instance_map_path, write_instance_map_tiff


def _write_blank(path: Path, w: int = 16, h: int = 16) -> None:
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    tifffile.imwrite(path, rgb, photometric="rgb")


def test_whole_requires_manifest(tmp_path: Path) -> None:
    class Args:
        unit = "whole"
        manifest = None
        image = None
        pred_instances = None
        pred_instances_dir = None
        gt_gpkg = None
        gt_txt = None
        gt_origin = None
        sample_id = None

    with pytest.raises(ValueError, match="Provide --manifest"):
        _resolve_eval_samples(Args())  # type: ignore[arg-type]


def test_whole_accepts_manifest(tmp_path: Path) -> None:
    image = tmp_path / "train_PPL.tif"
    _write_blank(image)
    pred = instance_map_path(tmp_path, "train")
    write_instance_map_tiff(pred, np.zeros((16, 16), dtype=np.int32))
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
                        "pred_instances": str(pred),
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
        pred_instances = None
        pred_instances_dir = None
        gt_gpkg = None
        gt_txt = None
        gt_origin = None
        sample_id = None

    samples = _resolve_eval_samples(Args())  # type: ignore[arg-type]
    assert len(samples) == 1
    assert samples[0].sample_id == "train"
