"""Consumer smoke: watershed tune MergedViewPqResult CSV/JSON artifact contract (issue 03)."""

from __future__ import annotations

import pytest

from common.merged_view_pq import MERGED_VIEW_PQ_RESULT_KEYS
from unet.extraction_tune_scoring import (
    WatershedParamSet,
    mean_train_pq_for_watershed_params,
    select_best_watershed_tune_row,
    watershed_best_json_summary,
    watershed_per_sample_columns,
    watershed_tune_fieldnames,
    watershed_tune_row,
)
from unet.watershed_tune_fixtures import (
    cached_semantic_pred_speckle_prone,
    cached_semantic_pred_two_grain,
    two_grain_merged_instance_view,
)

_SAMPLE_ID = "train"
_SANITIZE = lambda sample_id: sample_id  # noqa: E731


def _score_and_row(
    params: WatershedParamSet,
    *,
    gt=None,
    semantic=None,
) -> tuple[dict[str, float | int], list[dict[str, float | int]], dict[str, object]]:
    if gt is None:
        gt = two_grain_merged_instance_view()
    if semantic is None:
        semantic = cached_semantic_pred_two_grain()
    mean_pq, per_sample = mean_train_pq_for_watershed_params(
        [gt], [semantic], params, sample_ids=[_SAMPLE_ID]
    )
    row = watershed_tune_row(
        params,
        mean_pq,
        per_sample_pq=watershed_per_sample_columns(
            [_SAMPLE_ID], per_sample, sanitize_sample_id=_SANITIZE
        ),
    )
    return mean_pq, per_sample, row


def test_watershed_tune_scoring_calls_compute_merged_view_pq_not_eval_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """INTENT: watershed tune scoring routes through merged-view PQ, not instance metric bundle."""
    import unet.extraction_tune_scoring as scoring

    pq_calls = 0
    real_pq = scoring.compute_merged_view_pq

    def spy_pq(gt, pred, **kwargs):
        nonlocal pq_calls
        pq_calls += 1
        return real_pq(gt, pred, **kwargs)

    def forbid_bundle(*_args, **_kwargs):
        raise AssertionError(
            "watershed tune scoring must not call compute_instance_metric_bundle"
        )

    monkeypatch.setattr(scoring, "compute_merged_view_pq", spy_pq)
    monkeypatch.setattr(
        "common.instance_metric_bundle.compute_instance_metric_bundle",
        forbid_bundle,
    )

    params = WatershedParamSet(5, 0, 1, 0, False, None)
    mean_train_pq_for_watershed_params(
        [two_grain_merged_instance_view()],
        [cached_semantic_pred_two_grain()],
        params,
    )

    assert pq_calls == 1


def test_watershed_tune_best_selection_uses_mean_pq_only() -> None:
    """INTENT: best watershed param selection picks the highest mean PQ, not worse extractions."""
    good = WatershedParamSet(5, 0, 1, 0, False, None)
    bad = WatershedParamSet(1, 0, 1, 0, False, None)
    semantic = cached_semantic_pred_speckle_prone()
    gt = two_grain_merged_instance_view()

    _, _, row_good = _score_and_row(good, gt=gt, semantic=semantic)
    _, _, row_bad = _score_and_row(bad, gt=gt, semantic=semantic)

    best = select_best_watershed_tune_row([row_bad, row_good])

    assert float(best["mean_pq"]) == pytest.approx(float(row_good["mean_pq"]))
    assert float(best["mean_pq"]) > float(row_bad["mean_pq"])
    assert int(best["min_distance"]) == 5


def test_watershed_tune_grid_csv_includes_merged_view_pq_mean_and_per_sample_fields() -> (
    None
):
    """INTENT: watershed tune CSV rows expose merged-view PQ mean and per-sample fields only."""
    params = WatershedParamSet(5, 0, 1, 0, False, None)
    mean_pq, per_sample, row = _score_and_row(params)

    fieldnames = watershed_tune_fieldnames([_SAMPLE_ID], sanitize_sample_id=_SANITIZE)
    assert set(row.keys()) == set(fieldnames)

    for key in MERGED_VIEW_PQ_RESULT_KEYS:
        assert f"mean_{key}" in row
        assert f"{key}__{_SAMPLE_ID}" in row
        assert row[f"mean_{key}"] == row[f"{key}__{_SAMPLE_ID}"]

    assert int(row["mean_gt_instance_count"]) == int(mean_pq["gt_instance_count"])
    assert int(row["mean_pred_instance_count"]) == int(mean_pq["pred_instance_count"])
    assert float(row["mean_dq"]) == pytest.approx(float(mean_pq["dq"]))
    assert float(row["mean_sq"]) == pytest.approx(float(mean_pq["sq"]))
    assert int(row[f"gt_instance_count__{_SAMPLE_ID}"]) == int(
        per_sample[0]["gt_instance_count"]
    )
    assert int(row[f"pred_instance_count__{_SAMPLE_ID}"]) == int(
        per_sample[0]["pred_instance_count"]
    )


def test_watershed_tune_best_json_includes_merged_view_pq_mean_and_per_sample_fields() -> (
    None
):
    """INTENT: watershed best JSON summary carries merged-view PQ mean and per-sample fields."""
    params = WatershedParamSet(5, 0, 1, 0, False, None)
    _, _, row = _score_and_row(params)

    summary = watershed_best_json_summary(
        row, params, [_SAMPLE_ID], sanitize_sample_id=_SANITIZE
    )

    assert summary["selection_objective"] == "pq"
    for key in MERGED_VIEW_PQ_RESULT_KEYS:
        assert f"best_mean_{key}" in summary
        assert f"best_per_sample_{key}" in summary
        assert _SAMPLE_ID in summary[f"best_per_sample_{key}"]
        assert (
            summary[f"best_per_sample_{key}"][_SAMPLE_ID] == summary[f"best_mean_{key}"]
        )

    assert isinstance(summary["best_mean_pred_instance_count"], int)
    assert isinstance(summary["best_mean_gt_instance_count"], int)
    assert isinstance(summary["best_per_sample_dq"][_SAMPLE_ID], float)
    assert isinstance(summary["best_per_sample_sq"][_SAMPLE_ID], float)


def test_watershed_tune_diagnostics_surface_catastrophic_over_segmentation() -> None:
    """INTENT: catastrophic over-segmentation remains visible in tune CSV and best JSON artifacts."""
    gt = two_grain_merged_instance_view()
    semantic = cached_semantic_pred_speckle_prone()
    params = WatershedParamSet(1, 0, 1, 0, False, None)

    mean_pq, per_sample, row = _score_and_row(params, gt=gt, semantic=semantic)
    summary = watershed_best_json_summary(
        row, params, [_SAMPLE_ID], sanitize_sample_id=_SANITIZE
    )

    assert int(mean_pq["gt_instance_count"]) == 2
    assert int(mean_pq["pred_instance_count"]) > int(mean_pq["gt_instance_count"])
    assert float(mean_pq["pq"]) < 0.05
    assert int(row["mean_pred_instance_count"]) > int(row["mean_gt_instance_count"])
    assert row["mean_pq"] == "0.00000000"
    assert int(per_sample[0]["pred_instance_count"]) > 2
    assert summary["best_mean_pq"] == pytest.approx(0.0)
    assert (
        summary["best_mean_pred_instance_count"]
        > summary["best_mean_gt_instance_count"]
    )
