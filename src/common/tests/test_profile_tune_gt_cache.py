"""Tests for profile selection ground truth cache (ADR 0005)."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import tifffile

from common.profile_tune_gt_cache import (
    build_gt_fingerprint,
    gt_cache_dir,
    load_gt_instance_map_cache,
    write_gt_instance_map_cache,
    write_train_gt_cache,
)

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "gpkg_merged_instance_map"
_MICRO_GPKG = _FIXTURES / "micro_labels.gpkg"
_GOLDEN_NPZ = _FIXTURES / "instance_map.npz"
_FIXTURE_HEIGHT = 48
_FIXTURE_WIDTH = 64


def _write_anchor_tiff(path: Path, *, height: int, width: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, np.zeros((height, width, 3), dtype=np.uint8))


def test_build_gt_fingerprint_excludes_variant_includes_geometry(
    tmp_path: Path,
) -> None:
    """INTENT: build_gt_fingerprint keys on geometry and GPKG hash but not microscopy variant."""
    labels_gpkg = tmp_path / "train_labels.gpkg"
    labels_gpkg.write_bytes(b"labels")
    fingerprint = build_gt_fingerprint(
        sample_id="train",
        labels_gpkg=labels_gpkg,
        width=_FIXTURE_WIDTH,
        height=_FIXTURE_HEIGHT,
    )
    assert fingerprint["schema_version"] == 2
    assert fingerprint["sample_id"] == "train"
    assert fingerprint["width"] == _FIXTURE_WIDTH
    assert fingerprint["height"] == _FIXTURE_HEIGHT
    assert fingerprint["train_labels_gpkg_sha256"]
    assert "variant" not in fingerprint


def test_write_and_load_gt_cache_round_trip(tmp_path: Path) -> None:
    """INTENT: write_gt_instance_map_cache and load_gt_instance_map_cache round-trip an instance map."""
    gt_map = np.zeros((8, 8), dtype=np.int32)
    gt_map[1:4, 1:4] = 1
    labels_gpkg = tmp_path / "train_labels.gpkg"
    labels_gpkg.write_bytes(b"labels")
    fingerprint = build_gt_fingerprint(
        sample_id="train",
        labels_gpkg=labels_gpkg,
        width=8,
        height=8,
    )
    cache_dir = gt_cache_dir(tmp_path / ".cache")
    write_gt_instance_map_cache(cache_dir, gt_map, fingerprint=fingerprint)
    loaded, meta = load_gt_instance_map_cache(cache_dir, expected=fingerprint)
    np.testing.assert_array_equal(loaded, gt_map)
    assert meta["sample_id"] == "train"
    assert "variant" not in meta


def test_load_gt_cache_rejects_fingerprint_mismatch(tmp_path: Path) -> None:
    """INTENT: load_gt_instance_map_cache rejects caches whose fingerprint no longer matches the GPKG."""
    gt_map = np.ones((4, 4), dtype=np.int32)
    gpkg = tmp_path / "train_labels.gpkg"
    gpkg.write_bytes(b"gpkg-v1")
    fingerprint = build_gt_fingerprint(
        sample_id="train", labels_gpkg=gpkg, width=4, height=4
    )
    cache_dir = gt_cache_dir(tmp_path / ".cache")
    write_gt_instance_map_cache(cache_dir, gt_map, fingerprint=fingerprint)
    gpkg.write_bytes(b"gpkg-v2")
    mismatched = build_gt_fingerprint(
        sample_id="train", labels_gpkg=gpkg, width=4, height=4
    )
    with pytest.raises(ValueError, match="fingerprint"):
        load_gt_instance_map_cache(cache_dir, expected=mismatched)


def test_write_train_gt_cache_from_micro_gpkg_matches_golden(tmp_path: Path) -> None:
    """INTENT: write_train_gt_cache materializes the micro fixture GPKG to the golden instance map."""
    grainseg_root = tmp_path / "GrainSeg"
    labels_gpkg = grainseg_root / "dataset" / "train" / "train_labels.gpkg"
    labels_gpkg.parent.mkdir(parents=True)
    shutil.copy2(_MICRO_GPKG, labels_gpkg)
    anchor = grainseg_root / "dataset" / "train" / "train_PPL.tif"
    _write_anchor_tiff(
        anchor, height=_FIXTURE_HEIGHT, width=_FIXTURE_WIDTH
    )
    work_root = tmp_path / ".cache"

    cache_dir = write_train_gt_cache(
        work_root=work_root,
        grainseg_root=grainseg_root,
        tmp_dir=tmp_path / "tmpdir",
    )

    assert cache_dir == gt_cache_dir(work_root)
    with np.load(_GOLDEN_NPZ) as data:
        golden = np.asarray(data["instance_map"], dtype=np.int32)
    loaded, _meta = load_gt_instance_map_cache(
        cache_dir,
        expected=build_gt_fingerprint(
            sample_id="train",
            labels_gpkg=labels_gpkg,
            width=_FIXTURE_WIDTH,
            height=_FIXTURE_HEIGHT,
        ),
    )
    assert np.array_equal(loaded, golden)


def test_gt_cache_cli_module_writes_micro_fixture(tmp_path: Path) -> None:
    """INTENT: the profile_tune_gt_cache CLI writes a loadable cache matching the micro fixture golden map."""
    grainseg_root = tmp_path / "GrainSeg"
    labels_gpkg = grainseg_root / "dataset" / "train" / "train_labels.gpkg"
    labels_gpkg.parent.mkdir(parents=True)
    shutil.copy2(_MICRO_GPKG, labels_gpkg)
    anchor = grainseg_root / "dataset" / "train" / "train_PPL.tif"
    _write_anchor_tiff(
        anchor, height=_FIXTURE_HEIGHT, width=_FIXTURE_WIDTH
    )
    output_dir = tmp_path / "run"
    common_src = Path(__file__).resolve().parents[1]
    tmpdir = tmp_path / "tmpdir"
    tmpdir.mkdir(parents=True)
    env = os.environ.copy()
    env["TMPDIR"] = str(tmpdir)
    subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-u",
            "-m",
            "common.profile_tune_gt_cache",
            "--output-dir",
            str(output_dir),
            "--grainseg-root",
            str(grainseg_root),
        ],
        cwd=common_src,
        check=True,
        env=env,
    )
    cache_dir = gt_cache_dir(output_dir / ".cache")
    loaded, _meta = load_gt_instance_map_cache(
        cache_dir,
        expected=build_gt_fingerprint(
            sample_id="train",
            labels_gpkg=labels_gpkg,
            width=_FIXTURE_WIDTH,
            height=_FIXTURE_HEIGHT,
        ),
    )
    with np.load(_GOLDEN_NPZ) as data:
        golden = np.asarray(data["instance_map"], dtype=np.int32)
    assert np.array_equal(loaded, golden)
