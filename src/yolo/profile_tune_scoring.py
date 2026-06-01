"""In-process train AJI for YOLO profile selection scoring (ADR 0005, 0007)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from common.instance_maps import score_merged_instance_map_from_sahi_predictions
from common.metrics import compute_aji
from common.prediction_set import (
    build_yolo_prediction_set_from_sahi_predictions,
    merge_yolo_proposals_by_score,
    save_prediction_set,
)
from common.test_inference import YoloInferenceProfileCandidate
from yolo.sliced_detection import merge_sliced_object_predictions


@dataclass
class ProfileSelectionScoringTimings:
    slice_merge_s: float = 0.0
    score_merge_s: float = 0.0
    aji_s: float = 0.0

    @property
    def total_s(self) -> float:
        return self.slice_merge_s + self.score_merge_s + self.aji_s


def _log_timing(phase: str, elapsed_s: float) -> None:
    print(f"    {phase} {elapsed_s:.1f}s", flush=True)


def slice_merge_proposals(
    proposals: list[Any],
    *,
    candidate: YoloInferenceProfileCandidate,
) -> list[Any]:
    """SAHI slice-merge for one grid point (shared by scoring and audit materialization)."""
    return merge_sliced_object_predictions(
        proposals,
        postprocess_type=candidate.postprocess_type,
        match_metric=candidate.match_metric,
        match_threshold=candidate.match_threshold,
    )


def merged_instance_view_from_proposals(
    proposals: list[Any],
    *,
    candidate: YoloInferenceProfileCandidate,
    height: int,
    width: int,
    timings: ProfileSelectionScoringTimings | None = None,
) -> np.ndarray:
    """SAHI proposals → slice-merge → direct score-merge paint (ADR 0007)."""
    t0 = time.perf_counter()
    merged_predictions = slice_merge_proposals(proposals, candidate=candidate)
    slice_merge_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    pred_map = score_merged_instance_map_from_sahi_predictions(
        merged_predictions,
        height=height,
        width=width,
        mask_threshold=candidate.mask_threshold,
    )
    score_merge_s = time.perf_counter() - t1
    if timings is not None:
        timings.slice_merge_s = slice_merge_s
        timings.score_merge_s = score_merge_s
    return pred_map


def compute_train_aji(
    gt_instance_map: np.ndarray,
    proposals: list[Any],
    *,
    candidate: YoloInferenceProfileCandidate,
    height: int,
    width: int,
    log_timings: bool = False,
) -> float:
    """Train AJI for one variant and grid point without persisting a prediction set."""
    timings = ProfileSelectionScoringTimings() if log_timings else None
    pred_map = merged_instance_view_from_proposals(
        proposals,
        candidate=candidate,
        height=height,
        width=width,
        timings=timings,
    )
    t0 = time.perf_counter()
    gt = np.asarray(gt_instance_map)
    if pred_map.shape != gt.shape:
        raise ValueError(
            f"GT shape {gt.shape} does not match prediction shape {pred_map.shape}"
        )
    aji = float(compute_aji(gt, pred_map))
    if timings is not None:
        timings.aji_s = time.perf_counter() - t0
        _log_timing("slice-merge", timings.slice_merge_s)
        _log_timing("score-merge", timings.score_merge_s)
        _log_timing("AJI", timings.aji_s)
    return aji


def materialize_score_merged_prediction_set(
    proposals: list[Any],
    *,
    candidate: YoloInferenceProfileCandidate,
    height: int,
    width: int,
    path: Path,
) -> Path:
    """Write score-merged canonical prediction set (coordinator / evaluate_instances path)."""
    merged_predictions = slice_merge_proposals(proposals, candidate=candidate)
    pred_set = build_yolo_prediction_set_from_sahi_predictions(
        merged_predictions,
        height=height,
        width=width,
        mask_threshold=candidate.mask_threshold,
    )
    merged_set = merge_yolo_proposals_by_score(pred_set)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_prediction_set(path, merged_set)
    return path
