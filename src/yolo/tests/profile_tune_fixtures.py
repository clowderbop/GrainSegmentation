"""Shared SAHI-shaped fixtures for profile-tune and tiled-proposal tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from common.test_inference import YoloInferenceProfileCandidate
from common.tests.test_prediction_set_sahi import (
    _FakeCategory,
    _FakeMask,
    _FakeScore,
)
from common.variants import all_variant_names


@dataclass
class FakeBbox:
    minx: float = 4.0
    miny: float = 4.0
    maxx: float = 12.0
    maxy: float = 12.0

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


def candidate_for_variant(variant: str) -> YoloInferenceProfileCandidate:
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


def write_on_disk_v1_proposal_cache(
    cache_dir: Path, *, meta: dict[str, object]
) -> None:
    """Write a schema v1 cache layout (dense SAHI pickle + v1 meta sidecar)."""
    import json
    import pickle

    height = int(meta["height"])
    width = int(meta["width"])
    dense = np.zeros((height, width), dtype=bool)
    dense[0:4, 0:4] = True
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    v1_meta = dict(meta)
    v1_meta["schema_version"] = 1
    with (cache_dir / "proposals.pkl").open("wb") as handle:
        pickle.dump([V1SahiPickleStub(bool_mask=dense)], handle, protocol=pickle.HIGHEST_PROTOCOL)
    (cache_dir / "proposals.meta.json").write_text(
        json.dumps(v1_meta, indent=2), encoding="utf-8"
    )
