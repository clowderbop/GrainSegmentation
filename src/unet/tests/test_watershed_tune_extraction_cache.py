"""Tune-path watershed extraction cache: parity and extraction counts."""

from __future__ import annotations

import numpy as np
import pytest

from common.merged_view_pq import MERGED_VIEW_PQ_RESULT_KEYS
from unet.extraction_tune_scoring import (
    WatershedParamSet,
    instance_map_for_watershed_params,
    mean_train_pq_for_watershed_params,
    select_best_watershed_tune_row,
    watershed_tune_row,
    watershed_per_sample_columns,
)
from unet.watershed_tune_grid import (
    iter_watershed_tune_param_sets,
    load_watershed_tune_grid,
    watershed_tune_candidate_count,
)
from unet.watershed_tune_extraction_cache import (
    WatershedTuneSampleCache,
    build_watershed_tune_sample_caches,
    instance_map_from_tune_cache,
    iter_unique_watershed_base_extraction_keys,
    mean_train_pq_for_watershed_params_cached,
    watershed_base_extraction_key_count,
)


def _multi_grain_semantic_with_boundaries(height: int = 64, width: int = 64) -> np.ndarray:
    semantic = np.zeros((height, width), dtype=np.uint8)
    boxes = ((8, 8, 28, 28), (36, 36, 56, 56), (8, 36, 20, 56))
    for r0, c0, r1, c1 in boxes:
        semantic[r0:r1, c0:c1] = 1
        semantic[r0, c0:c1] = 2
        semantic[r1 - 1, c0:c1] = 2
        semantic[r0:r1, c0] = 2
        semantic[r0:r1, c1 - 1] = 2
    semantic[17:19, 8:28] = 0
    return semantic


def _two_grain_gt(height: int = 64, width: int = 64) -> np.ndarray:
    gt = np.zeros((height, width), dtype=np.int32)
    gt[8:28, 8:28] = 1
    gt[36:56, 36:56] = 2
    return gt


def test_default_grid_has_twenty_four_base_keys_and_seventy_two_combos() -> None:
    grid = load_watershed_tune_grid().grid
    assert watershed_base_extraction_key_count(grid) == 24
    assert watershed_tune_candidate_count(grid) == 72


def test_cached_instance_map_matches_brute_force_for_single_param() -> None:
    semantic = _multi_grain_semantic_with_boundaries()
    params = WatershedParamSet(5, 0, 1, 0, False, None)
    grid = load_watershed_tune_grid().grid
    cache = build_watershed_tune_sample_caches([semantic])[0]

    expected = instance_map_for_watershed_params(semantic, params)
    actual = instance_map_from_tune_cache(cache, params)

    np.testing.assert_array_equal(actual, expected)


def test_cached_instance_maps_match_brute_force_for_default_grid() -> None:
    semantic = _multi_grain_semantic_with_boundaries()
    grid = load_watershed_tune_grid().grid
    cache = build_watershed_tune_sample_caches([semantic])[0]

    for params in iter_watershed_tune_param_sets(grid):
        expected = instance_map_for_watershed_params(semantic, params)
        actual = instance_map_from_tune_cache(cache, params)
        np.testing.assert_array_equal(actual, expected)


def test_cached_scoring_matches_brute_force_for_default_grid() -> None:
    gt = _two_grain_gt()
    semantic = _multi_grain_semantic_with_boundaries()
    grid = load_watershed_tune_grid().grid
    caches = build_watershed_tune_sample_caches([semantic])

    reference_rows: list[dict[str, object]] = []
    cached_rows: list[dict[str, object]] = []

    for params in iter_watershed_tune_param_sets(grid):
        ref_mean, ref_per = mean_train_pq_for_watershed_params([gt], [semantic], params)
        cached_mean, cached_per = mean_train_pq_for_watershed_params_cached(
            [gt], caches, params
        )
        for key in MERGED_VIEW_PQ_RESULT_KEYS:
            assert cached_mean[key] == pytest.approx(ref_mean[key])
            assert cached_per[0][key] == pytest.approx(ref_per[0][key])
        reference_rows.append(
            watershed_tune_row(
                params,
                ref_mean,
                per_sample_pq=watershed_per_sample_columns(
                    ["train"], ref_per, sanitize_sample_id=lambda s: s
                ),
            )
        )
        cached_rows.append(
            watershed_tune_row(
                params,
                cached_mean,
                per_sample_pq=watershed_per_sample_columns(
                    ["train"], cached_per, sanitize_sample_id=lambda s: s
                ),
            )
        )

    assert select_best_watershed_tune_row(reference_rows) == select_best_watershed_tune_row(
        cached_rows
    )


def test_tune_cache_runs_one_base_extraction_per_unique_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import unet.watershed_tune_extraction_cache as cache_mod

    semantic = _multi_grain_semantic_with_boundaries()
    grid = load_watershed_tune_grid().grid
    base_calls = 0
    real_base = cache_mod.watershed_base_extraction

    def counting_base(*args, **kwargs):
        nonlocal base_calls
        base_calls += 1
        return real_base(*args, **kwargs)

    monkeypatch.setattr(cache_mod, "watershed_base_extraction", counting_base)

    cache = WatershedTuneSampleCache(semantic)
    for key in iter_unique_watershed_base_extraction_keys(grid):
        cache.base_label_map(key)

    assert base_calls == watershed_base_extraction_key_count(grid)


def test_mean_train_pq_cached_logs_phase_timings(capsys: pytest.CaptureFixture[str]) -> None:
    gt = _two_grain_gt()
    semantic = _multi_grain_semantic_with_boundaries()
    grid = load_watershed_tune_grid().grid
    caches = build_watershed_tune_sample_caches([semantic])
    params = WatershedParamSet(5, 0, 1, 0, False, None)

    mean_train_pq_for_watershed_params_cached(
        [gt],
        caches,
        params,
        sample_ids=["train"],
        log=True,
    )

    out = capsys.readouterr().out
    assert "running watershed" in out
    assert "running metrics" in out
    assert "PQ=" in out
