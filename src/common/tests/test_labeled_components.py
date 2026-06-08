"""Tests for connected-component label map helpers."""

from __future__ import annotations

import time

import numpy as np

from common.labeled_components import drop_small_components, relabel_sequential


def _sequential_single_pixel_labels(n: int, height: int, width: int) -> np.ndarray:
    if n > height * width:
        raise ValueError("n exceeds raster capacity")
    labeled = np.zeros((height, width), dtype=np.int32)
    rows, cols = np.divmod(np.arange(n), width)
    labeled[rows, cols] = np.arange(1, n + 1, dtype=np.int32)
    return labeled


def test_relabel_sequential_renumbers_nonzero_labels_contiguously() -> None:
    labeled = np.array([[0, 3, 0], [7, 0, 3]], dtype=np.int32)
    out = relabel_sequential(labeled)
    assert out.tolist() == [[0, 1, 0], [2, 0, 1]]


def test_relabel_sequential_compacts_sparse_high_label_ids() -> None:
    labeled = np.zeros((4, 4), dtype=np.int32)
    labeled[0, 0] = 1
    labeled[1, 1] = 1_000_000
    out = relabel_sequential(labeled)
    assert sorted(int(v) for v in np.unique(out) if v) == [1, 2]


def test_drop_small_components_removes_labels_below_area_threshold() -> None:
    labeled = np.zeros((4, 4), dtype=np.int32)
    labeled[0, 0] = 1
    labeled[0, 1] = 1
    labeled[1, 0] = 2
    labeled[2, 2] = 3
    labeled[2, 3] = 3
    labeled[3, 2] = 3
    labeled[3, 3] = 3

    out = drop_small_components(labeled, min_area_px=3)

    assert out[0, 0] == 0
    assert out[0, 1] == 0
    assert out[1, 0] == 0
    assert int(out.max()) == 1
    assert int((out > 0).sum()) == 4


def test_drop_small_components_compacts_when_nothing_dropped() -> None:
    labeled = np.zeros((4, 4), dtype=np.int32)
    labeled[0, 0] = 1
    labeled[1, 1] = 5
    labeled[2, 2] = 10
    labeled[3, 3] = 15

    out = drop_small_components(labeled, min_area_px=1)

    assert sorted(int(v) for v in np.unique(out) if v) == [1, 2, 3, 4]


def test_drop_small_components_noop_when_min_area_zero() -> None:
    labeled = np.array([[1, 2], [3, 4]], dtype=np.int32)
    out = drop_small_components(labeled, min_area_px=0)
    assert np.array_equal(out, labeled)


def test_drop_small_components_scales_with_pixels_not_max_label_id() -> None:
    """Regression: watershed combos with min_area_px>0 must not scan per label id."""
    n_labels = 5_000
    labeled = _sequential_single_pixel_labels(n_labels, height=1_000, width=1_000)
    assert len(np.unique(labeled[labeled != 0])) == n_labels

    t0 = time.perf_counter()
    out = drop_small_components(labeled, min_area_px=64)
    elapsed = time.perf_counter() - t0

    assert int(out.max()) == 0
    assert elapsed < 5.0, f"drop_small_components took {elapsed:.2f}s; expected O(pixels)"


def test_relabel_sequential_scales_with_pixels_not_max_label_id() -> None:
    labeled = np.zeros((64, 64), dtype=np.int32)
    labeled[0, 0] = 1
    labeled[1, 1] = 50_000

    t0 = time.perf_counter()
    out = relabel_sequential(labeled)
    elapsed = time.perf_counter() - t0

    assert sorted(int(v) for v in np.unique(out) if v) == [1, 2]
    assert elapsed < 5.0, f"relabel_sequential took {elapsed:.2f}s; expected O(pixels)"
