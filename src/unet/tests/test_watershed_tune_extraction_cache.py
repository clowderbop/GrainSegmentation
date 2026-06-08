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
    build_gt_overlap_preps,
    build_watershed_tune_sample_caches,
    instance_map_from_tune_cache,
    iter_unique_watershed_base_extraction_keys,
    mean_train_pq_for_watershed_params_cached,
    watershed_base_extraction_key_count,
)


def _multi_grain_semantic_with_boundaries(
    height: int = 64, width: int = 64
) -> np.ndarray:
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


def test_default_grid_extraction_cache_ratio_from_axis_structure() -> None:
    """INTENT: scored combos are base extractions times min_area_px axis length."""
    grid = load_watershed_tune_grid().grid
    base_count = watershed_base_extraction_key_count(grid)
    combo_count = watershed_tune_candidate_count(grid)
    assert base_count == len(list(iter_unique_watershed_base_extraction_keys(grid)))
    assert combo_count == len(list(iter_watershed_tune_param_sets(grid)))
    assert combo_count == base_count * len(grid.min_area_px)
    assert base_count < combo_count


def test_cached_instance_map_matches_brute_force_for_single_param() -> None:
    """INTENT: tune-cache instance maps match brute-force extraction for one param set."""
    semantic = _multi_grain_semantic_with_boundaries()
    params = WatershedParamSet(5, 0, 1, 0, False, None)
    cache = build_watershed_tune_sample_caches([semantic])[0]

    expected = instance_map_for_watershed_params(semantic, params)
    actual = instance_map_from_tune_cache(cache, params)

    np.testing.assert_array_equal(actual, expected)


def test_cached_instance_maps_match_brute_force_for_default_grid() -> None:
    """INTENT: tune-cache instance maps match brute-force extraction across the default grid."""
    semantic = _multi_grain_semantic_with_boundaries()
    grid = load_watershed_tune_grid().grid
    cache = build_watershed_tune_sample_caches([semantic])[0]

    for params in iter_watershed_tune_param_sets(grid):
        expected = instance_map_for_watershed_params(semantic, params)
        actual = instance_map_from_tune_cache(cache, params)
        np.testing.assert_array_equal(actual, expected)


def test_mean_train_pq_cached_does_not_require_pred_semantic_arrays() -> None:
    """INTENT: cached mean PQ scoring matches brute-force without passing raw semantic arrays."""
    gt = _two_grain_gt()
    semantic = _multi_grain_semantic_with_boundaries()
    params = WatershedParamSet(5, 0, 1, 0, False, None)
    caches = build_watershed_tune_sample_caches([semantic])
    gt_preps = build_gt_overlap_preps([gt])

    ref_mean, ref_per = mean_train_pq_for_watershed_params([gt], [semantic], params)
    cached_mean, cached_per = mean_train_pq_for_watershed_params_cached(
        [gt], caches, params, gt_overlap_preps=gt_preps
    )

    for key in MERGED_VIEW_PQ_RESULT_KEYS:
        assert cached_mean[key] == pytest.approx(ref_mean[key])
        assert cached_per[0][key] == pytest.approx(ref_per[0][key])


def test_cached_scoring_matches_brute_force_for_default_grid() -> None:
    """INTENT: cached grid scoring and best-row selection match brute-force for the default grid."""
    gt = _two_grain_gt()
    semantic = _multi_grain_semantic_with_boundaries()
    grid = load_watershed_tune_grid().grid
    caches = build_watershed_tune_sample_caches([semantic])
    gt_preps = build_gt_overlap_preps([gt])

    reference_rows: list[dict[str, object]] = []
    cached_rows: list[dict[str, object]] = []

    for params in iter_watershed_tune_param_sets(grid):
        ref_mean, ref_per = mean_train_pq_for_watershed_params([gt], [semantic], params)
        cached_mean, cached_per = mean_train_pq_for_watershed_params_cached(
            [gt], caches, params, gt_overlap_preps=gt_preps
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

    assert select_best_watershed_tune_row(
        reference_rows
    ) == select_best_watershed_tune_row(cached_rows)


def test_tune_cache_runs_one_base_extraction_per_unique_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """INTENT: tune cache invokes base watershed extraction once per unique base key."""
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


def test_tune_scoring_reuses_prebuilt_gt_overlap_preps_across_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """INTENT: cached grid scoring reuses prebuilt GT overlap preps without recomputing them."""
    import common.instance_overlap as overlap_mod

    gt = _two_grain_gt()
    semantic = _multi_grain_semantic_with_boundaries()
    grid = load_watershed_tune_grid().grid
    caches = build_watershed_tune_sample_caches([semantic])
    gt_preps = build_gt_overlap_preps([gt])

    gt_prep_calls = 0
    real_gt_overlap_prep = overlap_mod.gt_overlap_prep

    def counting_gt_overlap_prep(true_instances: np.ndarray):
        nonlocal gt_prep_calls
        gt_prep_calls += 1
        return real_gt_overlap_prep(true_instances)

    monkeypatch.setattr(overlap_mod, "gt_overlap_prep", counting_gt_overlap_prep)

    for params in iter_watershed_tune_param_sets(grid):
        mean_train_pq_for_watershed_params_cached(
            [gt],
            caches,
            params,
            gt_overlap_preps=gt_preps,
        )

    assert gt_prep_calls == 0


def test_tune_scoring_runs_overlap_extraction_per_combo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """INTENT: cached grid scoring runs overlap extraction once per watershed param combo."""
    import common.merged_view_pq as pq_mod

    gt = _two_grain_gt()
    semantic = _multi_grain_semantic_with_boundaries()
    grid = load_watershed_tune_grid().grid
    caches = build_watershed_tune_sample_caches([semantic])
    gt_preps = build_gt_overlap_preps([gt])

    overlap_calls = 0
    real_overlap_stats = pq_mod.instance_overlap_stats

    def counting_overlap_stats(*args, **kwargs):
        nonlocal overlap_calls
        overlap_calls += 1
        return real_overlap_stats(*args, **kwargs)

    monkeypatch.setattr(pq_mod, "instance_overlap_stats", counting_overlap_stats)

    for params in iter_watershed_tune_param_sets(grid):
        mean_train_pq_for_watershed_params_cached(
            [gt],
            caches,
            params,
            gt_overlap_preps=gt_preps,
        )

    assert overlap_calls == watershed_tune_candidate_count(grid)


def test_cached_scoring_logs_extraction_cache_miss_then_hit(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """INTENT: extraction-cache logging reports one miss then one hit for distinct base keys."""
    gt = _two_grain_gt()
    semantic = _multi_grain_semantic_with_boundaries()
    caches = build_watershed_tune_sample_caches([semantic])
    gt_preps = build_gt_overlap_preps([gt])
    params_miss = WatershedParamSet(5, 0, 1, 0, False, None)
    params_hit = WatershedParamSet(5, 0, 1, 64, False, None)

    mean_train_pq_for_watershed_params_cached(
        [gt],
        caches,
        params_miss,
        gt_overlap_preps=gt_preps,
        sample_ids=["train"],
        log_extraction_cache=True,
    )
    mean_train_pq_for_watershed_params_cached(
        [gt],
        caches,
        params_hit,
        gt_overlap_preps=gt_preps,
        sample_ids=["train"],
        log_extraction_cache=True,
    )

    out = capsys.readouterr().out
    assert out.count("extraction cache: miss") == 1
    assert out.count("extraction cache: hit") == 1


def test_default_grid_extraction_cache_log_miss_and_hit_counts(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """INTENT: default-grid cache logging reports misses per base key and hits for reused keys."""
    gt = _two_grain_gt()
    semantic = _multi_grain_semantic_with_boundaries()
    grid = load_watershed_tune_grid().grid
    caches = build_watershed_tune_sample_caches([semantic])
    gt_preps = build_gt_overlap_preps([gt])
    base_key_count = watershed_base_extraction_key_count(grid)
    combo_count = watershed_tune_candidate_count(grid)

    for params in iter_watershed_tune_param_sets(grid):
        mean_train_pq_for_watershed_params_cached(
            [gt],
            caches,
            params,
            gt_overlap_preps=gt_preps,
            sample_ids=["train"],
            log_extraction_cache=True,
        )

    out = capsys.readouterr().out
    assert out.count("extraction cache: miss") == base_key_count
    assert out.count("extraction cache: hit") == combo_count - base_key_count
    assert base_key_count == 24
    assert combo_count == 72


def test_mean_train_pq_cached_logs_phase_timings(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """INTENT: cached mean PQ scoring logs watershed and metrics phase timings when enabled."""
    gt = _two_grain_gt()
    semantic = _multi_grain_semantic_with_boundaries()
    caches = build_watershed_tune_sample_caches([semantic])
    gt_preps = build_gt_overlap_preps([gt])
    params = WatershedParamSet(5, 0, 1, 0, False, None)

    mean_train_pq_for_watershed_params_cached(
        [gt],
        caches,
        params,
        gt_overlap_preps=gt_preps,
        sample_ids=["train"],
        log=True,
    )

    out = capsys.readouterr().out
    assert "running watershed" in out
    assert "running metrics" in out
    assert "PQ=" in out
