"""PQ bundle parity: profile selection scoring vs evaluate_instances (ADR 0005)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from common.instance_metric_bundle import INSTANCE_METRIC_BUNDLE_KEYS
from common.test_inference import YoloInferenceProfileCandidate
from common.variants import all_variant_names
from common.instance_eval_report import extract_instance_metric_bundle_from_report
from yolo.tiled_proposal_cache import TiledProposalRecord
from common.tests.profile_tune_fixtures import tiny_train_gt_map
from yolo.tests.profile_tune_fixtures import (
    candidate_for_variant,
    constant_merged_view_pq_result,
    tiled_proposal_records_disjoint_via_collector,
    tiled_proposal_records_from_overlapping_masks,
)


def _train_bundle_via_evaluate_instances(
    gt_instance_map: np.ndarray,
    records: list[TiledProposalRecord],
    *,
    candidate: YoloInferenceProfileCandidate,
    height: int,
    width: int,
    variant: str,
    image_path: Path,
    prediction_set_path: Path,
) -> dict[str, float]:
    """Canonical path: materialize prediction set, then ``evaluate_instance_samples``."""
    from common.evaluate_instances import InstanceEvalSample, evaluate_instance_samples
    from yolo.profile_tune_scoring import materialize_cross_tile_prediction_set_from_records

    materialize_cross_tile_prediction_set_from_records(
        records,
        height=height,
        width=width,
        path=prediction_set_path,
    )
    sample = InstanceEvalSample(
        sample_id="train",
        image_path=image_path,
        instance_prediction_set=prediction_set_path,
        gt_gpkg=image_path,
        gt_origin="whole_image",
    )
    gt = np.asarray(gt_instance_map)

    def _fixed_gt(_sample: InstanceEvalSample, *, image_width: int, image_height: int) -> np.ndarray:
        if image_width != width or image_height != height:
            raise ValueError(
                f"fixture GT size ({height}, {width}) != image ({image_height}, {image_width})"
            )
        return gt

    with (
        patch("common.evaluate_instances.image_dimensions", return_value=(height, width)),
        patch("common.evaluate_instances.load_gt_instance_map", side_effect=_fixed_gt),
    ):
        report = evaluate_instance_samples(
            [sample],
            model_type="yolo",
            variant=variant,
            unit="whole",
        )
    return extract_instance_metric_bundle_from_report(report)


def _assert_bundles_equal(
    fast: dict[str, float], canonical: dict[str, float]
) -> None:
    for key in INSTANCE_METRIC_BUNDLE_KEYS:
        assert fast[key] == pytest.approx(canonical[key], rel=0.0, abs=1e-9), key


_PROFILE_SELECTION_PQ_PARITY_KEYS = (
    "pq",
    "dq",
    "sq",
    "tp",
    "fp",
    "fn",
    "precision_iou50",
    "recall_iou50",
    "f1_iou50",
    "gt_instance_count",
    "pred_instance_count",
    "pred_gt_instance_ratio",
)


def _assert_pq_result_matches_bundle_subset(
    result: dict[str, float | int], bundle: dict[str, float | int]
) -> None:
    for key in _PROFILE_SELECTION_PQ_PARITY_KEYS:
        assert result[key] == pytest.approx(bundle[key], rel=0.0, abs=1e-9), key


def _write_ppl_train_proposal_cache(
    tmp_path: Path,
    *,
    records: list[TiledProposalRecord],
    candidate: YoloInferenceProfileCandidate,
    height: int,
    width: int,
) -> tuple[Path, Path, Path]:
    """Write v3 train tiled-proposal cache; return (grainseg_root, run_root, work_root)."""
    from common.test_inference import load_test_inference_recipe
    from yolo.profile_tune_work import weights_path
    from yolo.tiled_proposal_cache import (
        proposal_cache_dir,
        proposal_cache_record,
        recipe_whole_window_fingerprint,
        weights_sha256,
        write_tiled_proposals,
    )

    grainseg_root = tmp_path / "grainseg"
    run_root = grainseg_root / "runs" / "yolo26-seg"
    weights = weights_path(grainseg_root, "PPL", run_root)
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"weights")
    work_root = tmp_path / ".cache"
    recipe = load_test_inference_recipe()
    write_tiled_proposals(
        proposal_cache_dir(work_root / "PPL", conf=candidate.conf),
        records,
        proposal_cache_record(
            variant="PPL",
            weights_sha256=weights_sha256(weights),
            recipe_window_fingerprint=recipe_whole_window_fingerprint(recipe),
            conf=candidate.conf,
            mask_threshold=candidate.mask_threshold,
            sample_id="train",
            height=height,
            width=width,
        ),
    )
    return grainseg_root, run_root, work_root


@pytest.mark.parametrize("variant", all_variant_names())
def test_profile_selection_scoring_pq_matches_evaluate_instances(
    tmp_path: Path, variant: str
) -> None:
    """In-process cross-tile PQ scoring matches evaluate_instances PQ fields."""
    from yolo.profile_tune_scoring import compute_train_pq

    height, width = 16, 16
    gt_map = tiny_train_gt_map(height, width)
    records = tiled_proposal_records_disjoint_via_collector(
        height, width, mask_threshold=candidate_for_variant(variant).mask_threshold
    )
    candidate = candidate_for_variant(variant)
    variant_dir = tmp_path / variant
    image_path = variant_dir / "train.tif"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"\x00")
    pred_path = variant_dir / "prediction_sets" / "train.json"

    fast_result = compute_train_pq(
        gt_map,
        records,
        candidate=candidate,
        height=height,
        width=width,
    )
    canonical_bundle = _train_bundle_via_evaluate_instances(
        gt_map,
        records,
        candidate=candidate,
        height=height,
        width=width,
        variant=variant,
        image_path=image_path,
        prediction_set_path=pred_path,
    )
    _assert_pq_result_matches_bundle_subset(fast_result, canonical_bundle)
    assert pred_path.is_file()


def test_tiled_proposal_cache_scoring_uses_cross_tile_postprocess(
    tmp_path: Path,
) -> None:
    """On-disk v3 caches score through cross-tile association, not SAHI+score-merge."""
    from yolo.profile_tune_candidate import score_variant_train_metrics_from_cache

    height, width = 16, 16
    gt_map = tiny_train_gt_map(height, width)
    candidate = candidate_for_variant("PPL")
    records = tiled_proposal_records_disjoint_via_collector(
        height, width, mask_threshold=candidate.mask_threshold
    )
    grainseg_root, run_root, work_root = _write_ppl_train_proposal_cache(
        tmp_path,
        records=records,
        candidate=candidate,
        height=height,
        width=width,
    )
    from common.merged_view_pq import MERGED_VIEW_PQ_RESULT_KEYS
    from yolo.profile_tune_scoring import compute_train_pq

    expected = compute_train_pq(
        gt_map,
        records,
        candidate=candidate,
        height=height,
        width=width,
    )
    result = score_variant_train_metrics_from_cache(
        variant="PPL",
        candidate=candidate,
        grainseg_root=grainseg_root,
        run_root=run_root,
        work_root=work_root,
        gt_map=gt_map,
    )
    assert tuple(result.keys()) == MERGED_VIEW_PQ_RESULT_KEYS
    assert "aji_plus" not in result
    assert "f1_iou75" not in result
    for key in MERGED_VIEW_PQ_RESULT_KEYS:
        assert result[key] == pytest.approx(expected[key], rel=0.0, abs=1e-9), key


@pytest.mark.parametrize("variant", all_variant_names())
def test_overlapping_tiled_records_scoring_matches_evaluate_instances(
    tmp_path: Path, variant: str
) -> None:
    """Cross-tile association on overlapping single-tile records matches evaluate_instances PQ."""
    from yolo.profile_tune_scoring import compute_train_pq

    height, width = 16, 16
    gt_map = tiny_train_gt_map(height, width)
    records = tiled_proposal_records_from_overlapping_masks(height, width)
    candidate = candidate_for_variant(variant)
    variant_dir = tmp_path / variant
    image_path = variant_dir / "train.tif"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"\x00")
    pred_path = variant_dir / "prediction_sets" / "train.json"

    fast_result = compute_train_pq(
        gt_map,
        records,
        candidate=candidate,
        height=height,
        width=width,
    )
    canonical_bundle = _train_bundle_via_evaluate_instances(
        gt_map,
        records,
        candidate=candidate,
        height=height,
        width=width,
        variant=variant,
        image_path=image_path,
        prediction_set_path=pred_path,
    )
    _assert_pq_result_matches_bundle_subset(fast_result, canonical_bundle)


def _train_pq_via_evaluate_instances(
    gt_instance_map: np.ndarray,
    records: list[TiledProposalRecord],
    *,
    candidate: YoloInferenceProfileCandidate,
    height: int,
    width: int,
    variant: str,
    image_path: Path,
    prediction_set_path: Path,
) -> float:
    bundle = _train_bundle_via_evaluate_instances(
        gt_instance_map,
        records,
        candidate=candidate,
        height=height,
        width=width,
        variant=variant,
        image_path=image_path,
        prediction_set_path=prediction_set_path,
    )
    return float(bundle["pq"])


def test_compute_train_pq_logs_cross_tile_and_metrics_phases(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from yolo.profile_tune_scoring import compute_train_pq
    from yolo.tests.phase_logging_assertions import (
        assert_done_timing_lines,
        assert_substrings_in_order,
    )

    height, width = 16, 16
    gt_map = tiny_train_gt_map(height, width)
    records = tiled_proposal_records_disjoint_via_collector(
        height, width, mask_threshold=candidate_for_variant("PPL").mask_threshold
    )
    compute_train_pq(
        gt_map,
        records,
        candidate=candidate_for_variant("PPL"),
        height=height,
        width=width,
        log_timings=True,
    )
    captured = capsys.readouterr().out
    assert_substrings_in_order(
        captured,
        "Evaluating train PQ …",
        "running cross-tile association …",
        "Enriching proposals …",
        "Cross-tile association done",
        "Rasterizing merged instance view done",
        "    cross-tile association ",
        "running metrics …",
        "    metrics ",
        "Evaluating train PQ done",
    )
    assert_done_timing_lines(captured, min_count=6)


@pytest.mark.parametrize("variant", all_variant_names())
def test_compute_train_pq_matches_bundle_pq_fields_on_fixtures(
    variant: str,
) -> None:
    """Profile selection hot path returns MergedViewPqResult aligned with bundle PQ fields."""
    from common.instance_metric_bundle import compute_instance_metric_bundle
    from common.merged_view_pq import MERGED_VIEW_PQ_RESULT_KEYS
    from yolo.profile_tune_scoring import (
        compute_train_pq,
        merged_instance_view_from_tiled_records,
    )

    height, width = 16, 16
    gt_map = tiny_train_gt_map(height, width)
    records = tiled_proposal_records_disjoint_via_collector(
        height, width, mask_threshold=candidate_for_variant(variant).mask_threshold
    )
    candidate = candidate_for_variant(variant)
    pred_map = merged_instance_view_from_tiled_records(
        records, height=height, width=width
    )
    bundle = compute_instance_metric_bundle(gt_map, pred_map)

    result = compute_train_pq(
        gt_map,
        records,
        candidate=candidate,
        height=height,
        width=width,
    )

    assert tuple(result.keys()) == MERGED_VIEW_PQ_RESULT_KEYS
    _assert_pq_result_matches_bundle_subset(result, bundle)


def test_compute_train_pq_delegates_to_shared_merged_view_pq_scorer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Consumer smoke: profile selection hot path scores via compute_merged_view_pq."""
    from yolo.profile_tune_scoring import compute_train_pq

    captured: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def spy(gt: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
        captured.append((gt.shape, pred.shape))
        return constant_merged_view_pq_result(0.75)

    monkeypatch.setattr("yolo.profile_tune_scoring.compute_merged_view_pq", spy)

    height, width = 16, 16
    gt_map = tiny_train_gt_map(height, width)
    candidate = candidate_for_variant("PPL")
    records = tiled_proposal_records_disjoint_via_collector(
        height, width, mask_threshold=candidate.mask_threshold
    )

    result = compute_train_pq(
        gt_map,
        records,
        candidate=candidate,
        height=height,
        width=width,
    )

    assert len(captured) == 1
    assert captured[0] == (gt_map.shape, (height, width))
    assert result["pq"] == pytest.approx(0.75)


def test_compute_train_pq_hot_path_avoids_instance_metric_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tune hot path scores MergedViewPqResult only, not the eval instance metric bundle."""
    from yolo.profile_tune_scoring import compute_train_pq

    def fail_bundle(*_args: object, **_kwargs: object) -> dict[str, float]:
        raise AssertionError(
            "profile selection scoring must not call compute_instance_metric_bundle"
        )

    monkeypatch.setattr(
        "common.instance_metric_bundle.compute_instance_metric_bundle",
        fail_bundle,
    )

    height, width = 16, 16
    gt_map = tiny_train_gt_map(height, width)
    candidate = candidate_for_variant("PPL")
    records = tiled_proposal_records_disjoint_via_collector(
        height, width, mask_threshold=candidate.mask_threshold
    )

    result = compute_train_pq(
        gt_map,
        records,
        candidate=candidate,
        height=height,
        width=width,
    )

    from common.merged_view_pq import MERGED_VIEW_PQ_RESULT_KEYS

    assert tuple(result.keys()) == MERGED_VIEW_PQ_RESULT_KEYS
    assert "aji_plus" not in result
    assert "f1_iou75" not in result


def test_profile_selection_scoring_smoke_cached_records_merged_view_pq_schema(
    tmp_path: Path,
) -> None:
    """Cached v3 proposals + small GT merged view → MergedViewPqResult schema (issue 04)."""
    from common.merged_view_pq import MERGED_VIEW_PQ_RESULT_KEYS, mean_merged_view_pq_results
    from yolo.profile_tune_candidate import score_variant_train_metrics_from_cache

    height, width = 16, 16
    gt_map = tiny_train_gt_map(height, width)
    candidate = candidate_for_variant("PPL")
    records = tiled_proposal_records_disjoint_via_collector(
        height, width, mask_threshold=candidate.mask_threshold
    )
    grainseg_root, run_root, work_root = _write_ppl_train_proposal_cache(
        tmp_path,
        records=records,
        candidate=candidate,
        height=height,
        width=width,
    )

    result = score_variant_train_metrics_from_cache(
        variant="PPL",
        candidate=candidate,
        grainseg_root=grainseg_root,
        run_root=run_root,
        work_root=work_root,
        gt_map=gt_map,
    )

    assert tuple(result.keys()) == MERGED_VIEW_PQ_RESULT_KEYS
    mean_fields = mean_merged_view_pq_results([result])
    assert tuple(mean_fields.keys()) == MERGED_VIEW_PQ_RESULT_KEYS
    assert mean_fields["pq"] == pytest.approx(result["pq"])
