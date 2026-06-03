"""Golden tests for OpenCV GPKG → merged instance view (ADR 0005)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from common.gpkg_instance_map import gpkg_to_merged_instance_map

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "gpkg_merged_instance_map"
_MICRO_GPKG = _FIXTURES / "micro_labels.gpkg"
_GOLDEN_NPZ = _FIXTURES / "instance_map.npz"
_FIXTURE_HEIGHT = 48
_FIXTURE_WIDTH = 64


def test_micro_gpkg_matches_golden_instance_map() -> None:
    painted = gpkg_to_merged_instance_map(
        _MICRO_GPKG,
        height=_FIXTURE_HEIGHT,
        width=_FIXTURE_WIDTH,
    )
    with np.load(_GOLDEN_NPZ) as data:
        golden = np.asarray(data["instance_map"], dtype=np.int32)

    assert painted.shape == golden.shape == (_FIXTURE_HEIGHT, _FIXTURE_WIDTH)
    assert painted.dtype == np.int32
    assert np.array_equal(painted, golden)
