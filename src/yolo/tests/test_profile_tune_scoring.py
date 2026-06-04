"""PQ bundle parity: profile selection scoring vs evaluate_instances (ADR 0005)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from common.instance_metric_bundle import INSTANCE_METRIC_BUNDLE_KEYS
from common.test_inference import YoloInferenceProfileCandidate
from common.variants import all_variant_names
from yolo.inference_profile_tune import extract_instance_metric_bundle_from_report
from yolo.tests.profile_tune_fixtures import (
    candidate_for_variant,
    constant_metric_bundle,
    disjoint_sahi_proposals,
    overlapping_sahi_proposals,
    tiny_train_gt_map,
    v2_records_from_disjoint_via_collector,
    v2_records_from_overlapping_masks,
)


def _train_bundle_via_evaluate_instances(
    gt_instance_map: np.ndarray,
    proposals: list,
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
    from yolo.profile_tune_scoring import materialize_score_merged_prediction_set

    materialize_score_merged_prediction_set(
        proposals,
        candidate=candidate,
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


@pytest.mark.parametrize("variant", all_variant_names())
def test_profile_selection_scoring_bundle_matches_evaluate_instances(
    tmp_path: Path, variant: str
) -> None:
    """In-process scoring bundle matches evaluate_instances on the same materialized set."""
    from yolo.profile_tune_scoring import compute_train_instance_metric_bundle

    height, width = 16, 16
    gt_map = tiny_train_gt_map(height, width)
    proposals = disjoint_sahi_proposals(height, width)
    candidate = candidate_for_variant(variant)
    variant_dir = tmp_path / variant
    image_path = variant_dir / "train.tif"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"\x00")
    pred_path = variant_dir / "prediction_sets" / "train.json"

    fast_bundle = compute_train_instance_metric_bundle(
        gt_map,
        proposals,
        candidate=candidate,
        height=height,
        width=width,
    )
    canonical_bundle = _train_bundle_via_evaluate_instances(
        gt_map,
        proposals,
        candidate=candidate,
        height=height,
        width=width,
        variant=variant,
        image_path=image_path,
        prediction_set_path=pred_path,
    )
    _assert_bundles_equal(fast_bundle, canonical_bundle)
    assert pred_path.is_file()


@pytest.mark.parametrize("proposals_fn", [disjoint_sahi_proposals])
def test_v2_tiled_proposal_cache_scoring_bundle_matches_legacy(
    tmp_path: Path,
    proposals_fn,
) -> None:
    """ADR 0005 parity on on-disk v2 caches: direct paint bundle == legacy round-trip."""
    from common.instance_metric_bundle import compute_instance_metric_bundle
    from common.test_inference import load_test_inference_recipe
    from yolo.profile_tune_candidate import score_variant_train_metrics_from_cache
    from yolo.tests.profile_tune_legacy_scoring import legacy_merged_instance_view_from_proposals
    from yolo.tiled_proposal_cache import (
        proposal_cache_dir,
        proposal_cache_record,
        recipe_whole_window_fingerprint,
        sahi_predictions_from_tiled_proposal_records,
        weights_sha256,
        write_tiled_proposals,
        load_tiled_proposals,
        detector_cache_expected_record,
    )
    from yolo.profile_tune_work import weights_path

    height, width = 16, 16
    gt_map = tiny_train_gt_map(height, width)
    candidate = candidate_for_variant("PPL")
    grainseg_root = tmp_path / "grainseg"
    run_root = grainseg_root / "runs" / "yolo26-seg"
    weights = run_root / "PPL" / "weights" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"weights")
    work_root = tmp_path / ".cache"
    recipe = load_test_inference_recipe()
    v2_records = v2_records_from_disjoint_via_collector(
        height, width, mask_threshold=candidate.mask_threshold
    )
    write_tiled_proposals(
        proposal_cache_dir(
            work_root / "PPL", conf=candidate.conf, mask_threshold=candidate.mask_threshold
        ),
        v2_records,
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
    expected = detector_cache_expected_record(
        variant="PPL",
        weights_path=weights,
        conf=candidate.conf,
        mask_threshold=candidate.mask_threshold,
        sample_id="train",
        recipe=recipe,
    )
    cache_dir = proposal_cache_dir(
        work_root / "PPL", conf=candidate.conf, mask_threshold=candidate.mask_threshold
    )
    records, meta = load_tiled_proposals(cache_dir, expected=expected)
    proposals = sahi_predictions_from_tiled_proposal_records(
        records, height=int(meta["height"]), width=int(meta["width"])
    )

    fast_bundle = score_variant_train_metrics_from_cache(
        variant="PPL",
        candidate=candidate,
        grainseg_root=grainseg_root,
        run_root=run_root,
        work_root=work_root,
        gt_map=gt_map,
    )

    legacy_map = legacy_merged_instance_view_from_proposals(
        proposals,
        candidate=candidate,
        height=height,
        width=width,
    )
    legacy_bundle = compute_instance_metric_bundle(gt_map, legacy_map)
    _assert_bundles_equal(fast_bundle, legacy_bundle)


def test_v2_overlapping_tiled_proposal_cache_scoring_bundle_matches_legacy(
    tmp_path: Path,
) -> None:
    """ADR 0005: overlapping v2 records exercise slice-merge mask union (crop-local adapter)."""
    from common.instance_metric_bundle import compute_instance_metric_bundle
    from common.test_inference import load_test_inference_recipe
    from yolo.profile_tune_candidate import score_variant_train_metrics_from_cache
    from yolo.tests.profile_tune_legacy_scoring import legacy_merged_instance_view_from_proposals
    from yolo.tiled_proposal_cache import (
        detector_cache_expected_record,
        load_tiled_proposals,
        proposal_cache_dir,
        proposal_cache_record,
        recipe_whole_window_fingerprint,
        sahi_predictions_from_tiled_proposal_records,
        weights_sha256,
        write_tiled_proposals,
    )

    height, width = 16, 16
    gt_map = tiny_train_gt_map(height, width)
    candidate = candidate_for_variant("PPL")
    grainseg_root = tmp_path / "grainseg"
    run_root = grainseg_root / "runs" / "yolo26-seg"
    weights = run_root / "PPL" / "weights" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"weights")
    work_root = tmp_path / ".cache"
    recipe = load_test_inference_recipe()
    v2_records = v2_records_from_overlapping_masks(height, width)
    write_tiled_proposals(
        proposal_cache_dir(
            work_root / "PPL", conf=candidate.conf, mask_threshold=candidate.mask_threshold
        ),
        v2_records,
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

    fast_bundle = score_variant_train_metrics_from_cache(
        variant="PPL",
        candidate=candidate,
        grainseg_root=grainseg_root,
        run_root=run_root,
        work_root=work_root,
        gt_map=gt_map,
    )

    cache_dir = proposal_cache_dir(
        work_root / "PPL", conf=candidate.conf, mask_threshold=candidate.mask_threshold
    )
    expected = detector_cache_expected_record(
        variant="PPL",
        weights_path=weights,
        conf=candidate.conf,
        mask_threshold=candidate.mask_threshold,
        sample_id="train",
        recipe=recipe,
    )
    records, meta = load_tiled_proposals(cache_dir, expected=expected)
    proposals = sahi_predictions_from_tiled_proposal_records(
        records, height=int(meta["height"]), width=int(meta["width"])
    )
    legacy_map = legacy_merged_instance_view_from_proposals(
        proposals,
        candidate=candidate,
        height=height,
        width=width,
    )
    legacy_bundle = compute_instance_metric_bundle(gt_map, legacy_map)
    _assert_bundles_equal(fast_bundle, legacy_bundle)


@pytest.mark.parametrize("variant", all_variant_names())
def test_overlapping_score_merge_bundle_matches_evaluate_instances(
    tmp_path: Path, variant: str
) -> None:
    """Score-merge parity when slice-merge is a no-op (overlapping fixture masks)."""
    from yolo.profile_tune_scoring import (
        compute_train_instance_metric_bundle,
        merge_sliced_object_predictions,
    )

    height, width = 16, 16
    gt_map = tiny_train_gt_map(height, width)
    proposals = overlapping_sahi_proposals(height, width)
    candidate = candidate_for_variant(variant)
    variant_dir = tmp_path / variant
    image_path = variant_dir / "train.tif"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"\x00")
    pred_path = variant_dir / "prediction_sets" / "train.json"

    def _identity_merge(proposals_in: list, **_kwargs: object) -> list:
        return proposals_in

    with patch(
        "yolo.profile_tune_scoring.merge_sliced_object_predictions",
        side_effect=_identity_merge,
    ):
        fast_bundle = compute_train_instance_metric_bundle(
            gt_map,
            proposals,
            candidate=candidate,
            height=height,
            width=width,
        )
        canonical_bundle = _train_bundle_via_evaluate_instances(
            gt_map,
            proposals,
            candidate=candidate,
            height=height,
            width=width,
            variant=variant,
            image_path=image_path,
            prediction_set_path=pred_path,
        )
    _assert_bundles_equal(fast_bundle, canonical_bundle)


def _train_pq_via_evaluate_instances(
    gt_instance_map: np.ndarray,
    proposals: list,
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
        proposals,
        candidate=candidate,
        height=height,
        width=width,
        variant=variant,
        image_path=image_path,
        prediction_set_path=prediction_set_path,
    )
    return float(bundle["pq"])
