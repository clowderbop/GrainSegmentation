"""Pre-ADR-0007 prediction-set round-trip scoring (tests only, parity gate)."""

from __future__ import annotations

from typing import Any

import numpy as np

from common.prediction_set import (
    build_yolo_prediction_set_from_sahi_predictions,
    merge_yolo_proposals_by_score,
    prediction_set_to_merged_instance_view,
)
from common.test_inference import YoloInferenceProfileCandidate
from yolo.profile_tune_scoring import slice_merge_proposals


def legacy_merged_instance_view_from_proposals(
    proposals: list[Any],
    *,
    candidate: YoloInferenceProfileCandidate,
    height: int,
    width: int,
) -> np.ndarray:
    """Slice-merge → prediction-set encode → score merge → rasterize (parity reference)."""
    merged_predictions = slice_merge_proposals(proposals, candidate=candidate)
    pred_set = build_yolo_prediction_set_from_sahi_predictions(
        merged_predictions,
        height=height,
        width=width,
        mask_threshold=candidate.mask_threshold,
    )
    merged_set = merge_yolo_proposals_by_score(pred_set)
    return prediction_set_to_merged_instance_view(merged_set)
