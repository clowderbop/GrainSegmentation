"""Watershed hyperparameter tuning loads GPKG GT via the OpenCV painter (ADR 0005)."""

from __future__ import annotations

import csv
import json
import sys
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import tifffile

from unet import tune_watershed
from unet.tune_watershed import _collect_samples
from unet.tests.tune_watershed_cli_fixtures import (
    MICRO_GPKG,
    TUNE_CLI_HEIGHT,
    TUNE_CLI_WIDTH,
    make_tune_collect_args,
    write_mini_tune_grid,
)
from unet.watershed_tune_extraction_cache import (
    build_watershed_tune_sample_caches,
    mean_train_pq_for_watershed_params_cached,
)

_GOLDEN_NPZ = MICRO_GPKG.parent / "instance_map.npz"


def _golden_map() -> np.ndarray:
    with np.load(_GOLDEN_NPZ) as data:
        return np.asarray(data["instance_map"], dtype=np.int32)


def test_collect_samples_works_with_metadata_only_manifest_without_rgb_files(
    tmp_path: Path,
) -> None:
    """INTENT: _collect_samples works from metadata-only manifests when cached preds supply geometry."""
    sample_id = "train"
    pred_path = tmp_path / "preds" / f"{sample_id}_pred.tif"
    pred_path.parent.mkdir(parents=True)
    semantic = np.zeros((TUNE_CLI_HEIGHT, TUNE_CLI_WIDTH), dtype=np.uint8)
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
                        "image": "missing_PPL.tif",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    args = Namespace(
        manifest=manifest_path,
        gt_gpkg=MICRO_GPKG,
        preds_dir=tmp_path / "preds",
        max_samples=None,
        num_inputs=None,
    )
    sample_ids, true_instances, pred_semantic = _collect_samples(args)
    assert sample_ids == [sample_id]
    assert true_instances[0].shape == (TUNE_CLI_HEIGHT, TUNE_CLI_WIDTH)
    assert pred_semantic[0].shape == (TUNE_CLI_HEIGHT, TUNE_CLI_WIDTH)


def test_collect_samples_uses_pred_geometry_without_loading_rgb(
    tmp_path: Path,
) -> None:
    """INTENT: _collect_samples uses cached pred shape for GT painting without loading RGB."""
    args = make_tune_collect_args(
        tmp_path,
        pred_shape=(TUNE_CLI_HEIGHT, TUNE_CLI_WIDTH),
        image_shape=(32, 40),
        paint_semantic_region=True,
    )

    with patch("common.samples.load_rgb_image") as load_rgb:
        load_rgb.side_effect = AssertionError("must not load microscopy RGB for geometry")
        sample_ids, true_instances, pred_semantic = _collect_samples(args)

    load_rgb.assert_not_called()
    assert sample_ids == ["train"]
    assert true_instances[0].shape == (TUNE_CLI_HEIGHT, TUNE_CLI_WIDTH)
    assert pred_semantic[0].shape == (TUNE_CLI_HEIGHT, TUNE_CLI_WIDTH)


def test_collect_samples_raises_on_gt_pred_shape_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """INTENT: _collect_samples raises a sample-named error when painted GT and pred shapes differ."""
    args = make_tune_collect_args(tmp_path)

    def _wrong_shape_gt(*_args: object, **_kwargs: object) -> np.ndarray:
        return np.zeros((10, 10), dtype=np.int32)

    monkeypatch.setattr("unet.tune_watershed.polygons_to_instance_map", _wrong_shape_gt)

    with pytest.raises(ValueError, match=r"sample 'train'"):
        _collect_samples(args)


def test_collect_samples_gpkg_gt_matches_golden(tmp_path: Path) -> None:
    """INTENT: _collect_samples paints GPKG scene polygons to match the golden instance map."""
    args = make_tune_collect_args(tmp_path, paint_semantic_region=True)
    sample_ids, true_instances, _pred_semantic = _collect_samples(args)

    assert sample_ids == ["train"]
    assert len(true_instances) == 1
    assert np.array_equal(true_instances[0], _golden_map())


def test_tune_watershed_main_uses_extraction_cache_for_cached_preds_workflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """INTENT: tune_watershed main builds tune caches once and scores every grid combo via cached path."""
    grid_path = tmp_path / "mini_grid.yaml"
    write_mini_tune_grid(grid_path)
    args = make_tune_collect_args(tmp_path, paint_semantic_region=True)
    out_csv = tmp_path / "grid.csv"
    out_json = tmp_path / "best.json"

    cache_builds = 0
    real_build = build_watershed_tune_sample_caches

    def spy_build(pred_semantic: list[np.ndarray]) -> list[object]:
        nonlocal cache_builds
        cache_builds += 1
        return real_build(pred_semantic)

    cached_scoring_calls = 0
    real_cached = mean_train_pq_for_watershed_params_cached

    def spy_cached(*a: object, **kw: object) -> tuple[dict[str, float | int], list[dict]]:
        nonlocal cached_scoring_calls
        cached_scoring_calls += 1
        return real_cached(*a, **kw)

    monkeypatch.setattr(
        tune_watershed, "build_watershed_tune_sample_caches", spy_build
    )
    monkeypatch.setattr(
        tune_watershed,
        "mean_train_pq_for_watershed_params_cached",
        spy_cached,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tune_watershed",
            "--preds-dir",
            str(args.preds_dir),
            "--manifest",
            str(args.manifest),
            "--gt-gpkg",
            str(args.gt_gpkg),
            "--output-csv",
            str(out_csv),
            "--output-json",
            str(out_json),
            "--grid-config",
            str(grid_path),
        ],
    )
    tune_watershed.main()

    assert cache_builds == 1
    assert cached_scoring_calls == 4
    assert out_csv.is_file()
    with out_csv.open(encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 4
