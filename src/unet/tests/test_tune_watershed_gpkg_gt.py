"""Watershed hyperparameter tuning loads GPKG GT via the OpenCV painter (ADR 0005)."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
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


def _make_tune_collect_args(
    tmp_path: Path,
    *,
    sample_id: str = "train",
    pred_shape: tuple[int, int] = (_HEIGHT, _WIDTH),
    image_shape: tuple[int, int] = (_HEIGHT, _WIDTH),
    paint_semantic_region: bool = False,
) -> Namespace:
    """Minimal manifest + cached pred (+ manifest RGB stub) for ``_collect_samples``."""
    pred_height, pred_width = pred_shape
    image_height, image_width = image_shape

    image_path = tmp_path / f"{sample_id}_PPL.tif"
    rgb = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    tifffile.imwrite(image_path, rgb, photometric="rgb")

    pred_path = tmp_path / "preds" / f"{sample_id}_pred.tif"
    pred_path.parent.mkdir(parents=True)
    semantic = np.zeros((pred_height, pred_width), dtype=np.uint8)
    if paint_semantic_region:
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
                        "sample_id": sample_id,
                        "image": str(image_path),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    return Namespace(
        manifest=manifest_path,
        gt_gpkg=_MICRO_GPKG,
        preds_dir=tmp_path / "preds",
        max_samples=None,
        num_inputs=None,
    )


def test_collect_samples_uses_pred_geometry_without_loading_rgb(
    tmp_path: Path,
) -> None:
    """When manifest RGB and cached pred disagree in size, pred shape drives GT painting."""
    args = _make_tune_collect_args(
        tmp_path,
        pred_shape=(_HEIGHT, _WIDTH),
        image_shape=(32, 40),
        paint_semantic_region=True,
    )

    with patch("common.samples.load_rgb_image") as load_rgb:
        load_rgb.side_effect = AssertionError("must not load microscopy RGB for geometry")
        sample_ids, true_instances, pred_semantic = _collect_samples(args)

    load_rgb.assert_not_called()
    assert sample_ids == ["train"]
    assert true_instances[0].shape == (_HEIGHT, _WIDTH)
    assert pred_semantic[0].shape == (_HEIGHT, _WIDTH)


def test_collect_samples_raises_on_gt_pred_shape_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Shape mismatch between painted GT and cached pred names the affected sample id."""
    args = _make_tune_collect_args(tmp_path)

    def _wrong_shape_gt(*_args: object, **_kwargs: object) -> np.ndarray:
        return np.zeros((10, 10), dtype=np.int32)

    monkeypatch.setattr("unet.tune_watershed.polygons_to_instance_map", _wrong_shape_gt)

    with pytest.raises(ValueError, match=r"sample 'train'"):
        _collect_samples(args)


def test_collect_samples_gpkg_gt_matches_golden(tmp_path: Path) -> None:
    """``tune_watershed._collect_samples`` paints GPKG scene polygons at pred resolution."""
    args = _make_tune_collect_args(tmp_path, paint_semantic_region=True)
    sample_ids, true_instances, _pred_semantic = _collect_samples(args)

    assert sample_ids == ["train"]
    assert len(true_instances) == 1
    assert np.array_equal(true_instances[0], _golden_map())
