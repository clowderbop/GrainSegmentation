"""Watershed hyperparameter tuning loads GPKG GT via the OpenCV painter (ADR 0005)."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import tifffile

from unet.tune_watershed import _collect_samples

_REPO_SRC = Path(__file__).resolve().parents[2]
_MICRO_GPKG = (
    _REPO_SRC
    / "common"
    / "tests"
    / "fixtures"
    / "gpkg_merged_instance_map"
    / "micro_labels.gpkg"
)
_GOLDEN_NPZ = _MICRO_GPKG.parent / "instance_map.npz"
_HEIGHT = 48
_WIDTH = 64


def _golden_map() -> np.ndarray:
    with np.load(_GOLDEN_NPZ) as data:
        return np.asarray(data["instance_map"], dtype=np.int32)


def test_collect_samples_gpkg_gt_matches_golden(tmp_path: Path) -> None:
    """``tune_watershed._collect_samples`` paints GPKG scene polygons at image resolution."""
    image_path = tmp_path / "train_PPL.tif"
    rgb = np.zeros((_HEIGHT, _WIDTH, 3), dtype=np.uint8)
    tifffile.imwrite(image_path, rgb, photometric="rgb")

    pred_path = tmp_path / "preds" / "train_pred.tif"
    pred_path.parent.mkdir(parents=True)
    semantic = np.zeros((_HEIGHT, _WIDTH), dtype=np.uint8)
    semantic[8:20, 8:24] = 1
    tifffile.imwrite(pred_path, semantic)

    manifest_path = tmp_path / "manifest.json"
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
                        "image": str(image_path),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    args = Namespace(
        manifest=manifest_path,
        gt_gpkg=_MICRO_GPKG,
        preds_dir=tmp_path / "preds",
        model_path=None,
        max_samples=None,
        num_inputs=None,
    )
    sample_ids, true_instances, _pred_semantic = _collect_samples(args)

    assert sample_ids == ["train"]
    assert len(true_instances) == 1
    assert np.array_equal(true_instances[0], _golden_map())
