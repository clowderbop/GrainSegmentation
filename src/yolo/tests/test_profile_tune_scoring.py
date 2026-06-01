"""AJI parity: profile selection scoring vs evaluate_instances (ADR 0005)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from common.test_inference import YoloInferenceProfileCandidate
from common.variants import all_variant_names
from yolo.tests.profile_tune_fixtures import (
    candidate_for_variant,
    disjoint_sahi_proposals,
    overlapping_sahi_proposals,
    tiny_train_gt_map,
)


def _train_aji_via_evaluate_instances(
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
    """Canonical path: materialize prediction set, then ``evaluate_instance_samples``."""
    from common.evaluate_instances import InstanceEvalSample, evaluate_instance_samples
    from yolo.inference_profile_tune import extract_mean_aji_from_report
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
    return extract_mean_aji_from_report(report)


@pytest.mark.parametrize("variant", all_variant_names())
def test_profile_selection_scoring_aji_matches_evaluate_instances(
    tmp_path: Path, variant: str
) -> None:
    """In-process scoring AJI matches evaluate_instances on the same materialized set."""
    from yolo.profile_tune_scoring import compute_train_aji

    height, width = 16, 16
    gt_map = tiny_train_gt_map(height, width)
    proposals = disjoint_sahi_proposals(height, width)
    candidate = candidate_for_variant(variant)
    variant_dir = tmp_path / variant
    image_path = variant_dir / "train.tif"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"\x00")
    pred_path = variant_dir / "prediction_sets" / "train.json"

    fast_aji = compute_train_aji(
        gt_map,
        proposals,
        candidate=candidate,
        height=height,
        width=width,
    )
    canonical_aji = _train_aji_via_evaluate_instances(
        gt_map,
        proposals,
        candidate=candidate,
        height=height,
        width=width,
        variant=variant,
        image_path=image_path,
        prediction_set_path=pred_path,
    )
    assert fast_aji == pytest.approx(canonical_aji, rel=0.0, abs=1e-9)
    assert pred_path.is_file()


@pytest.mark.parametrize("variant", all_variant_names())
def test_overlapping_score_merge_aji_matches_evaluate_instances(
    tmp_path: Path, variant: str
) -> None:
    """Score-merge parity when slice-merge is a no-op (overlapping fixture masks)."""
    from yolo.profile_tune_scoring import compute_train_aji, merge_sliced_object_predictions

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
        fast_aji = compute_train_aji(
            gt_map,
            proposals,
            candidate=candidate,
            height=height,
            width=width,
        )
        canonical_aji = _train_aji_via_evaluate_instances(
            gt_map,
            proposals,
            candidate=candidate,
            height=height,
            width=width,
            variant=variant,
            image_path=image_path,
            prediction_set_path=pred_path,
        )
    assert fast_aji == pytest.approx(canonical_aji, rel=0.0, abs=1e-9)
