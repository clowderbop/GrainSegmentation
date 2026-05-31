"""Tests for profile selection ground truth cache (ADR 0005)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from yolo.profile_tune_gt_cache import (
    build_gt_fingerprint,
    gt_cache_dir,
    load_gt_instance_map_cache,
    write_gt_instance_map_cache,
)


def test_gt_cache_dir_under_work_root(tmp_path: Path) -> None:
    assert gt_cache_dir(tmp_path / "_work", "PPL") == tmp_path / "_work" / "gt_cache" / "PPL"


def test_write_and_load_gt_cache_round_trip(tmp_path: Path) -> None:
    gt_map = np.zeros((8, 8), dtype=np.int32)
    gt_map[1:4, 1:4] = 1
    labels_gpkg = tmp_path / "train_labels.gpkg"
    labels_gpkg.write_bytes(b"labels")
    fingerprint = build_gt_fingerprint(
        variant="PPL",
        sample_id="train",
        labels_gpkg=labels_gpkg,
    )
    cache_dir = gt_cache_dir(tmp_path / "_work", "PPL")
    write_gt_instance_map_cache(cache_dir, gt_map, fingerprint=fingerprint)
    loaded, meta = load_gt_instance_map_cache(cache_dir, expected=fingerprint)
    np.testing.assert_array_equal(loaded, gt_map)
    assert meta["variant"] == "PPL"


def test_load_gt_cache_rejects_fingerprint_mismatch(tmp_path: Path) -> None:
    gt_map = np.ones((4, 4), dtype=np.int32)
    gpkg = tmp_path / "train_labels.gpkg"
    gpkg.write_bytes(b"gpkg-v1")
    fingerprint = build_gt_fingerprint(
        variant="PPL", sample_id="train", labels_gpkg=gpkg
    )
    cache_dir = gt_cache_dir(tmp_path / "_work", "PPL")
    write_gt_instance_map_cache(cache_dir, gt_map, fingerprint=fingerprint)
    gpkg.write_bytes(b"gpkg-v2")
    mismatched = build_gt_fingerprint(
        variant="PPL", sample_id="train", labels_gpkg=gpkg
    )
    with pytest.raises(ValueError, match="fingerprint"):
        load_gt_instance_map_cache(cache_dir, expected=mismatched)
