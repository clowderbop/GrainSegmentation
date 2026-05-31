"""YOLO prediction sets built from SAHI-style object predictions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from common.prediction_set import (
    build_yolo_prediction_set_from_sahi_predictions,
    merge_yolo_proposals_by_score,
    prediction_set_to_merged_instance_view,
)
from common.tests.prediction_set_fixtures import yolo_prediction_set_from_masks


@dataclass
class _FakeMask:
    bool_mask: np.ndarray


@dataclass
class _FakeScore:
    value: float


@dataclass
class _FakeCategory:
    id: int


@dataclass
class _FakeSahiPrediction:
    mask: _FakeMask | None
    score: _FakeScore | None
    category: _FakeCategory


def test_build_yolo_prediction_set_from_sahi_rejects_missing_score() -> None:
    height, width = 8, 8
    mask = np.zeros((height, width), dtype=bool)
    mask[2:6, 2:6] = True
    pred = _FakeSahiPrediction(
        mask=_FakeMask(bool_mask=mask),
        score=None,
        category=_FakeCategory(id=0),
    )
    with pytest.raises(ValueError, match="missing score"):
        build_yolo_prediction_set_from_sahi_predictions(
            [pred], height=height, width=width
        )


def test_build_yolo_prediction_set_from_sahi_matches_mask_planes() -> None:
    height, width = 8, 8
    masks = np.zeros((2, height, width), dtype=np.float32)
    masks[0, 2:6, 2:6] = 1.0
    masks[1, 2:6, 2:6] = 1.0
    scores = np.array([0.2, 0.9], dtype=np.float32)

    predictions = [
        _FakeSahiPrediction(
            mask=_FakeMask(bool_mask=masks[0].astype(bool)),
            score=_FakeScore(value=float(scores[0])),
            category=_FakeCategory(id=0),
        ),
        _FakeSahiPrediction(
            mask=_FakeMask(bool_mask=masks[1].astype(bool)),
            score=_FakeScore(value=float(scores[1])),
            category=_FakeCategory(id=0),
        ),
    ]

    from_sahi = build_yolo_prediction_set_from_sahi_predictions(
        predictions, height=height, width=width
    )
    from_masks = yolo_prediction_set_from_masks(
        masks_hw=masks, scores=scores, height=height, width=width
    )
    merged = prediction_set_to_merged_instance_view(
        merge_yolo_proposals_by_score(from_sahi)
    )
    expected = prediction_set_to_merged_instance_view(
        merge_yolo_proposals_by_score(from_masks)
    )
    np.testing.assert_array_equal(merged > 0, expected > 0)
