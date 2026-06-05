"""Phase progress logging across YOLO whole-section hot paths."""

from __future__ import annotations

import pytest

from yolo.cross_tile_association import associate_tiled_proposals
from yolo.tests.cross_tile_association_fixtures import slice_boundary_duplicate_pair
from yolo.tests.phase_logging_assertions import (
    assert_done_timing_lines,
    assert_substrings_in_order,
)


def test_associate_tiled_proposals_logs_association_sub_phases(
    capsys: pytest.CaptureFixture[str],
) -> None:
    proposals, height, width = slice_boundary_duplicate_pair()
    associate_tiled_proposals(proposals, height=height, width=width, log_timings=True)
    out = capsys.readouterr().out

    assert_substrings_in_order(
        out,
        "Enriching proposals …",
        "Enriching proposals done",
        "Building candidate pairs …",
        "Building candidate pairs done",
        "Merging predictions …",
        "Merging predictions done",
        "Cross-tile association done",
    )
    assert_done_timing_lines(out, min_count=4)


def test_merged_instance_view_logs_rasterize_phase(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from yolo.cross_tile_postprocess import merged_instance_view_from_tiled_proposal_records
    from yolo.tests.profile_tune_fixtures import (
        tiled_proposal_records_disjoint_via_collector,
    )

    height, width = 16, 16
    records = tiled_proposal_records_disjoint_via_collector(height, width, mask_threshold=0.5)
    merged_instance_view_from_tiled_proposal_records(
        records, height=height, width=width, log_timings=True
    )
    out = capsys.readouterr().out
    assert_substrings_in_order(
        out,
        "Enriching proposals …",
        "Cross-tile association done",
        "Rasterizing merged instance view …",
        "Rasterizing merged instance view done",
    )
    assert_done_timing_lines(out, min_count=5)
