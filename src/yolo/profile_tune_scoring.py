"""In-process train merged-view PQ scoring for YOLO profile selection (ADR 0005, 0007)."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common.merged_view_pq import MergedViewPqResult, compute_merged_view_pq
from common.test_inference import YoloInferenceProfileCandidate
from yolo.cross_tile_postprocess import (
    materialize_cross_tile_prediction_set,
    merged_instance_view_from_tiled_proposal_records,
)
from yolo.tiled_proposal_cache import TiledProposalRecord


@dataclass
class ProfileSelectionScoringTimings:
    cross_tile_association_s: float = 0.0
    metrics_s: float = 0.0

    @property
    def total_s(self) -> float:
        return self.cross_tile_association_s + self.metrics_s


def _log_timing(phase: str, elapsed_s: float) -> None:
    print(f"    {phase} {elapsed_s:.1f}s", flush=True)


def merged_instance_view_from_tiled_records(
    records: Sequence[TiledProposalRecord],
    *,
    height: int,
    width: int,
    timings: ProfileSelectionScoringTimings | None = None,
) -> np.ndarray:
    """Tiled proposals → cross-tile association → merged instance view (ADR 0005)."""
    t0 = time.perf_counter()
    pred_map = merged_instance_view_from_tiled_proposal_records(
        records, height=height, width=width
    )
    if timings is not None:
        timings.cross_tile_association_s = time.perf_counter() - t0
    return pred_map


def compute_train_pq(
    gt_instance_map: np.ndarray,
    records: Sequence[TiledProposalRecord],
    *,
    candidate: YoloInferenceProfileCandidate,
    height: int,
    width: int,
    log_timings: bool = False,
) -> MergedViewPqResult:
    """Train whole-section PQ for one variant/grid point (profile selection hot path)."""
    del candidate  # cross-tile thresholds are fixed; conf/mask_threshold affect detector cache only
    timings = ProfileSelectionScoringTimings() if log_timings else None
    pred_map = merged_instance_view_from_tiled_records(
        records,
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
    result = compute_merged_view_pq(gt, pred_map)
    if timings is not None:
        timings.metrics_s = time.perf_counter() - t0
        _log_timing("cross-tile association", timings.cross_tile_association_s)
        _log_timing("metrics", timings.metrics_s)
    return result


def materialize_cross_tile_prediction_set_from_records(
    records: Sequence[TiledProposalRecord],
    *,
    height: int,
    width: int,
    path: Path,
) -> Path:
    """Write canonical prediction set after cross-tile association."""
    return materialize_cross_tile_prediction_set(
        records, height=height, width=width, path=path
    )
