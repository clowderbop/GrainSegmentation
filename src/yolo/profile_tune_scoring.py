"""In-process train AJI for YOLO profile selection scoring (ADR 0005)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from common.metrics import compute_aji
from common.prediction_set import (
    PredictionSet,
    build_yolo_prediction_set_from_sahi_predictions,
    merge_yolo_proposals_by_score,
    prediction_set_to_merged_instance_view,
    save_prediction_set,
)
from common.test_inference import YoloInferenceProfileCandidate
from yolo.sliced_detection import merge_sliced_object_predictions


def score_merged_prediction_set_from_proposals(
    proposals: list[Any],
    *,
    candidate: YoloInferenceProfileCandidate,
    height: int,
    width: int,
) -> PredictionSet:
    """SAHI proposals → slice-merge → score merge (canonical YOLO grains, no disk)."""
    merged_predictions = merge_sliced_object_predictions(
        proposals,
        postprocess_type=candidate.postprocess_type,
        match_metric=candidate.match_metric,
        match_threshold=candidate.match_threshold,
    )
    pred_set = build_yolo_prediction_set_from_sahi_predictions(
        merged_predictions,
        height=height,
        width=width,
        mask_threshold=candidate.mask_threshold,
    )
    return merge_yolo_proposals_by_score(pred_set)


def merged_instance_view_from_proposals(
    proposals: list[Any],
    *,
    candidate: YoloInferenceProfileCandidate,
    height: int,
    width: int,
) -> np.ndarray:
    """SAHI proposals → slice-merge → score merge → merged instance label map."""
    merged_set = score_merged_prediction_set_from_proposals(
        proposals,
        candidate=candidate,
        height=height,
        width=width,
    )
    return prediction_set_to_merged_instance_view(merged_set)


def compute_train_aji(
    gt_instance_map: np.ndarray,
    proposals: list[Any],
    *,
    candidate: YoloInferenceProfileCandidate,
    height: int,
    width: int,
) -> float:
    """Train AJI for one variant and grid point without persisting a prediction set."""
    pred_map = merged_instance_view_from_proposals(
        proposals,
        candidate=candidate,
        height=height,
        width=width,
    )
    gt = np.asarray(gt_instance_map)
    if pred_map.shape != gt.shape:
        raise ValueError(
            f"GT shape {gt.shape} does not match prediction shape {pred_map.shape}"
        )
    return float(compute_aji(gt, pred_map))


def materialize_score_merged_prediction_set(
    proposals: list[Any],
    *,
    candidate: YoloInferenceProfileCandidate,
    height: int,
    width: int,
    path: Path,
) -> Path:
    """Write score-merged canonical prediction set (coordinator / evaluate_instances path)."""
    merged_set = score_merged_prediction_set_from_proposals(
        proposals,
        candidate=candidate,
        height=height,
        width=width,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    save_prediction_set(path, merged_set)
    return path
