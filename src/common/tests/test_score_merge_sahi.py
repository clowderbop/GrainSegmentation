"""Direct score-merge paint from SAHI predictions (ADR 0007)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from common.instance_maps import score_merged_instance_map_from_sahi_predictions
from common.metrics import compute_aji
from common.prediction_set import (
    build_yolo_prediction_set_from_sahi_predictions,
    merge_yolo_proposals_by_score,
    prediction_set_to_merged_instance_view,
)
from common.tests.prediction_set_fixtures import assert_instance_map_partitions_equal
from common.tests.profile_tune_fixtures import (
    disjoint_sahi_proposals,
    overlapping_sahi_proposals,
    tiny_train_gt_map,
)


def _legacy_score_merged_map_via_prediction_set(
    predictions: list,
    *,
    height: int,
    width: int,
    mask_threshold: float,
) -> np.ndarray:
    pred_set = build_yolo_prediction_set_from_sahi_predictions(
        predictions,
        height=height,
        width=width,
        mask_threshold=mask_threshold,
    )
    merged_set = merge_yolo_proposals_by_score(pred_set)
    return prediction_set_to_merged_instance_view(merged_set)


@pytest.mark.parametrize(
    "proposals_fn",
    [disjoint_sahi_proposals, overlapping_sahi_proposals],
)
def test_score_merged_instance_map_direct_paint_matches_prediction_set_path(
    proposals_fn,
) -> None:
    height, width = 16, 16
    proposals = proposals_fn(height, width)
    mask_threshold = 0.5
    gt = tiny_train_gt_map(height, width)
    fast = score_merged_instance_map_from_sahi_predictions(
        proposals,
        height=height,
        width=width,
        mask_threshold=mask_threshold,
    )
    legacy = _legacy_score_merged_map_via_prediction_set(
        proposals,
        height=height,
        width=width,
        mask_threshold=mask_threshold,
    )
    assert_instance_map_partitions_equal(fast, legacy)
    assert compute_aji(gt, fast) == pytest.approx(compute_aji(gt, legacy), abs=1e-9)


def test_score_merged_instance_map_paints_incrementally() -> None:
    """ADR 0007: score-merge must not batch all decoded masks before painting."""
    height, width = 16, 16
    proposals = overlapping_sahi_proposals(height, width)

    def fail_batch_path(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError(
            "score_merged_instance_map_from_sahi_predictions must paint "
            "one mask at a time, not via yolo_detections_to_instance_map_by_score"
        )

    with patch(
        "common.instance_maps.yolo_detections_to_instance_map_by_score",
        side_effect=fail_batch_path,
    ):
        score_merged_instance_map_from_sahi_predictions(
            proposals,
            height=height,
            width=width,
            mask_threshold=0.5,
        )
