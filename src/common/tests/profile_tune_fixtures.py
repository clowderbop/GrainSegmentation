"""Neutral SAHI-shaped fixtures for profile-tune scoring and proposal-cache tests."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from common.tests.test_prediction_set_sahi import (
    _FakeCategory,
    _FakeMask,
    _FakeScore,
)


@dataclass
class FakeBbox:
    minx: float = 4.0
    miny: float = 4.0
    maxx: float = 12.0
    maxy: float = 12.0
    shift_amount: list[int] = field(default_factory=lambda: [0, 0])

    def to_xyxy(self) -> list[float]:
        return [self.minx, self.miny, self.maxx, self.maxy]


@dataclass
class FakeSahiPrediction:
    mask: _FakeMask
    score: _FakeScore
    category: _FakeCategory
    bbox: FakeBbox = field(default_factory=FakeBbox)

    def tolist(self) -> FakeSahiPrediction:
        return self


@dataclass
class V1SahiPickleStub:
    """Stand-in for schema v1 on-disk SAHI ``ObjectPrediction`` (dense full-section mask)."""

    bool_mask: np.ndarray
    score_value: float = 0.5


def tiny_train_gt_map(height: int = 16, width: int = 16) -> np.ndarray:
    gt = np.zeros((height, width), dtype=np.int32)
    gt[2:8, 2:8] = 1
    gt[10:14, 10:14] = 2
    return gt


def overlapping_sahi_proposals(
    height: int, width: int
) -> list[FakeSahiPrediction]:
    """Two overlapping proposals (score-merge path)."""
    masks = np.zeros((2, height, width), dtype=bool)
    masks[0, 4:12, 4:12] = True
    masks[1, 4:12, 4:12] = True
    return [
        FakeSahiPrediction(
            mask=_FakeMask(bool_mask=masks[0]),
            score=_FakeScore(value=0.2),
            category=_FakeCategory(id=0),
            bbox=FakeBbox(4.0, 4.0, 12.0, 12.0),
        ),
        FakeSahiPrediction(
            mask=_FakeMask(bool_mask=masks[1]),
            score=_FakeScore(value=0.9),
            category=_FakeCategory(id=0),
            bbox=FakeBbox(4.0, 4.0, 12.0, 12.0),
        ),
    ]


def disjoint_sahi_proposals(height: int, width: int) -> list[FakeSahiPrediction]:
    """Two non-overlapping proposals — slice-merge without pair-wise mask fusion."""
    masks = np.zeros((2, height, width), dtype=bool)
    masks[0, 2:8, 2:8] = True
    masks[1, 10:14, 10:14] = True
    return [
        FakeSahiPrediction(
            mask=_FakeMask(bool_mask=masks[0]),
            score=_FakeScore(value=0.7),
            category=_FakeCategory(id=0),
            bbox=FakeBbox(2.0, 2.0, 8.0, 8.0),
        ),
        FakeSahiPrediction(
            mask=_FakeMask(bool_mask=masks[1]),
            score=_FakeScore(value=0.8),
            category=_FakeCategory(id=0),
            bbox=FakeBbox(10.0, 10.0, 14.0, 14.0),
        ),
    ]
