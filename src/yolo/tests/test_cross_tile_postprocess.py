"""Canonical cross-tile postprocess wiring (profile selection + whole predict)."""

from __future__ import annotations

from common.prediction_set import (
    prediction_set_to_merged_instance_view,
    segmentation_to_binary_mask,
)
from yolo.cross_tile_association import associate_tiled_proposals
from yolo.cross_tile_postprocess import (
    merged_instance_view_from_tiled_proposal_records,
    prediction_set_from_tiled_proposal_records,
)
from yolo.tests.cross_tile_association_fixtures import slice_boundary_duplicate_pair
from yolo.tests.profile_tune_fixtures import (
    tiled_proposal_records_disjoint_via_collector,
)
from yolo.tiled_proposal_cache import tiled_proposal_record_from_tile_mask


def test_prediction_set_from_disjoint_collector_records() -> None:
    """INTENT: postprocess builds a canonical prediction set from disjoint collector records."""
    height, width = 16, 16
    records = tiled_proposal_records_disjoint_via_collector(
        height, width, mask_threshold=0.5
    )
    for record in records:
        assert "tile_y0" in record
    pred_set = prediction_set_from_tiled_proposal_records(
        records, height=height, width=width
    )
    assert pred_set.height == height
    assert pred_set.width == width
    assert len(pred_set.detections) == 2


def test_merged_view_matches_direct_association_on_fixture() -> None:
    """INTENT: postprocess module output matches direct association on a fixture."""
    proposals, height, width = slice_boundary_duplicate_pair()
    direct = associate_tiled_proposals(proposals, height=height, width=width)
    records = [
        tiled_proposal_record_from_tile_mask(
            segmentation_to_binary_mask(p.segmentation),
            score=p.score,
            offset_y=p.offset_y,
            offset_x=p.offset_x,
            tile_y0=p.tile_y0,
            tile_x0=p.tile_x0,
            tile_y1=p.tile_y1,
            tile_x1=p.tile_x1,
        )
        for p in proposals
    ]
    via_module = prediction_set_from_tiled_proposal_records(
        records, height=height, width=width
    )
    assert len(via_module.detections) == len(direct.detections)
    merged = merged_instance_view_from_tiled_proposal_records(
        records, height=height, width=width
    )
    assert merged.shape == (height, width)
    assert merged.sum() == prediction_set_to_merged_instance_view(direct).sum()
