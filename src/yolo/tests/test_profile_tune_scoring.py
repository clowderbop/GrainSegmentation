"""AJI parity: profile selection scoring vs evaluate_instances (ADR 0005)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from common.test_inference import YoloInferenceProfileCandidate
from common.tests.test_prediction_set_sahi import (
    _FakeCategory,
    _FakeMask,
    _FakeScore,
)
from common.variants import all_variant_names


@dataclass
class _FakeBbox:
    minx: float = 4.0
    miny: float = 4.0
    maxx: float = 12.0
    maxy: float = 12.0

    def to_xyxy(self) -> list[float]:
        return [self.minx, self.miny, self.maxx, self.maxy]


@dataclass
class _FakeSahiPrediction:
    mask: _FakeMask
    score: _FakeScore
    category: _FakeCategory
    bbox: _FakeBbox = field(default_factory=_FakeBbox)

    def tolist(self) -> _FakeSahiPrediction:
        return self


def _tiny_gt_map(height: int = 16, width: int = 16) -> np.ndarray:
    gt = np.zeros((height, width), dtype=np.int32)
    gt[2:8, 2:8] = 1
    gt[10:14, 10:14] = 2
    return gt


def _overlapping_proposals(height: int, width: int) -> list[_FakeSahiPrediction]:
    """Two overlapping proposals (score-merge path) in separate tile offsets for SAHI merge."""
    masks = np.zeros((2, height, width), dtype=bool)
    masks[0, 4:12, 4:12] = True
    masks[1, 4:12, 4:12] = True
    return [
        _FakeSahiPrediction(
            mask=_FakeMask(bool_mask=masks[0]),
            score=_FakeScore(value=0.2),
            category=_FakeCategory(id=0),
            bbox=_FakeBbox(4.0, 4.0, 12.0, 12.0),
        ),
        _FakeSahiPrediction(
            mask=_FakeMask(bool_mask=masks[1]),
            score=_FakeScore(value=0.9),
            category=_FakeCategory(id=0),
            bbox=_FakeBbox(4.0, 4.0, 12.0, 12.0),
        ),
    ]


def _disjoint_proposals(height: int, width: int) -> list[_FakeSahiPrediction]:
    """Two non-overlapping proposals — exercises slice-merge without pair-wise mask fusion."""
    masks = np.zeros((2, height, width), dtype=bool)
    masks[0, 2:8, 2:8] = True
    masks[1, 10:14, 10:14] = True
    return [
        _FakeSahiPrediction(
            mask=_FakeMask(bool_mask=masks[0]),
            score=_FakeScore(value=0.7),
            category=_FakeCategory(id=0),
            bbox=_FakeBbox(2.0, 2.0, 8.0, 8.0),
        ),
        _FakeSahiPrediction(
            mask=_FakeMask(bool_mask=masks[1]),
            score=_FakeScore(value=0.8),
            category=_FakeCategory(id=0),
            bbox=_FakeBbox(10.0, 10.0, 14.0, 14.0),
        ),
    ]


def _candidate_for_variant(variant: str) -> YoloInferenceProfileCandidate:
    """One distinct grid point per registry variant (ADR 0005 parity fixtures)."""
    by_variant: dict[str, YoloInferenceProfileCandidate] = {
        "PPL": YoloInferenceProfileCandidate(
            postprocess_type="GREEDYNMM",
            match_metric="IOS",
            match_threshold=0.4,
            conf=0.15,
            mask_threshold=0.4,
        ),
        "PPLPPXblend": YoloInferenceProfileCandidate(
            postprocess_type="NMM",
            match_metric="IOU",
            match_threshold=0.5,
            conf=0.25,
            mask_threshold=0.5,
        ),
        "PPL+PPXblend": YoloInferenceProfileCandidate(
            postprocess_type="GREEDYNMM",
            match_metric="IOU",
            match_threshold=0.6,
            conf=0.35,
            mask_threshold=0.6,
        ),
        "PPL+AllPPX": YoloInferenceProfileCandidate(
            postprocess_type="NMM",
            match_metric="IOS",
            match_threshold=0.5,
            conf=0.25,
            mask_threshold=0.5,
        ),
    }
    return by_variant[variant]


def _train_aji_via_evaluate_instances(
    gt_instance_map: np.ndarray,
    proposals: list[_FakeSahiPrediction],
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
    gt_map = _tiny_gt_map(height, width)
    proposals = _disjoint_proposals(height, width)
    candidate = _candidate_for_variant(variant)
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
    gt_map = _tiny_gt_map(height, width)
    proposals = _overlapping_proposals(height, width)
    candidate = _candidate_for_variant(variant)
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
