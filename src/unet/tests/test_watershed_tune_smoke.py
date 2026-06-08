"""Lightweight watershed tune timing harness (issue 04)."""

from __future__ import annotations

import pytest

from common.merged_view_pq import MERGED_VIEW_PQ_RESULT_KEYS
from unet.extraction_tune_scoring import WatershedParamSet
from unet.watershed_tune_smoke import (
    default_smoke_watershed_params,
    run_watershed_tune_smoke,
)


def test_run_watershed_tune_smoke_scores_one_combo_with_phase_timings(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """INTENT: smoke harness scores one combo and logs watershed and metrics phase timings."""
    mean_pq, per_sample = run_watershed_tune_smoke(
        default_smoke_watershed_params(),
        height=64,
        width=64,
        sample_id="train",
    )

    out = capsys.readouterr().out
    assert "watershed tune smoke" in out
    assert "running watershed" in out
    assert "running metrics" in out
    assert "watershed" in out and "metrics" in out
    assert "PQ=" in out
    assert "smoke complete" in out

    assert len(per_sample) == 1
    for key in MERGED_VIEW_PQ_RESULT_KEYS:
        assert key in mean_pq
        assert key in per_sample[0]
    assert mean_pq["pq"] == pytest.approx(per_sample[0]["pq"])
    assert int(mean_pq["gt_instance_count"]) == 2


def test_run_watershed_tune_smoke_uses_phased_extraction_not_tune_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """INTENT: smoke harness scores via phased extraction, not the tune extraction cache."""
    import unet.watershed_tune_extraction_cache as cache_mod

    cache_called = False
    real_cached = cache_mod.mean_train_pq_for_watershed_params_cached

    def spy_cached(*args, **kwargs):
        nonlocal cache_called
        cache_called = True
        return real_cached(*args, **kwargs)

    monkeypatch.setattr(
        cache_mod, "mean_train_pq_for_watershed_params_cached", spy_cached
    )

    run_watershed_tune_smoke(
        WatershedParamSet(5, 0, 1, 0, False, None),
        height=64,
        width=64,
    )

    assert not cache_called


def test_run_watershed_tune_smoke_uses_compute_merged_view_pq_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """INTENT: smoke harness routes scoring through merged-view PQ, not legacy bundle metrics."""
    import unet.extraction_tune_scoring as scoring

    pq_calls = 0
    real_pq = scoring.compute_merged_view_pq

    def spy_pq(gt, pred, **kwargs):
        nonlocal pq_calls
        pq_calls += 1
        return real_pq(gt, pred, **kwargs)

    monkeypatch.setattr(scoring, "compute_merged_view_pq", spy_pq)

    run_watershed_tune_smoke(
        WatershedParamSet(5, 0, 1, 0, False, None),
        height=64,
        width=64,
    )

    assert pq_calls == 1


def test_watershed_tune_smoke_cli_runs_one_combo(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """INTENT: watershed tune smoke CLI runs one param combo and prints completion diagnostics."""
    from unet.watershed_tune_smoke import main

    main(
        [
            "--height",
            "64",
            "--width",
            "64",
            "--min-distance",
            "5",
        ]
    )

    out = capsys.readouterr().out
    assert "watershed tune smoke" in out
    assert "shape=(64, 64)" in out
    assert "min_dist=5" in out
    assert "smoke complete" in out
