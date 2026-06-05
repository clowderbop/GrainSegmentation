"""Crop-local proposal geometry for cross-tile association (scale issue 01)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

import yolo.cross_tile_association as cta
from common.prediction_set import assert_yolo_grains_non_overlapping, segmentation_to_binary_mask
from yolo.cross_tile_association import associate_tiled_proposals, mask_ios_crop_local
from yolo.tests.cross_tile_association_fixtures import slice_boundary_duplicate_pair
from yolo.tests.test_cross_tile_association import _foreground_mask_count

# Fixture local masks are 8x8 or smaller; budget catches full-section planes.
_LOCAL_MASK_PIXEL_BUDGET = 64 * 64


def _zeros_allocation_violations() -> tuple[list[tuple[int, ...]], object]:
    violations: list[tuple[int, ...]] = []
    original_zeros = np.zeros

    def spy_zeros(shape, *args, **kwargs):
        dims = (shape,) if isinstance(shape, int) else tuple(int(s) for s in shape)
        pixels = 1
        for dim in dims:
            pixels *= dim
        if pixels > _LOCAL_MASK_PIXEL_BUDGET:
            violations.append(dims)
        return original_zeros(shape, *args, **kwargs)

    return violations, spy_zeros


def test_enrich_and_pair_compare_avoid_full_section_mask_allocation() -> None:
    """Enrichment and pair comparison must not allocate section-sized boolean planes."""
    proposals, _, _ = slice_boundary_duplicate_pair()
    violations, spy_zeros = _zeros_allocation_violations()

    with patch.object(cta.np, "zeros", spy_zeros):
        enriched = cta._enrich_proposals(proposals)
        for left_index in range(len(enriched)):
            for right_index in range(left_index + 1, len(enriched)):
                cta._should_associate(enriched[left_index], enriched[right_index])

    assert violations == []
    for entry in enriched:
        assert entry.local_mask.size <= _LOCAL_MASK_PIXEL_BUDGET
        assert entry.area > 0
        assert 0.0 <= entry.centrality <= 1.0
        assert entry.proposal.score > 0.0


def test_associate_tiled_proposals_on_huge_section_smoke() -> None:
    """Behavioral smoke: association still merges fixtures on train-scale dimensions."""
    proposals, _, _ = slice_boundary_duplicate_pair()
    result = associate_tiled_proposals(proposals, height=10_000, width=52_000)
    assert_yolo_grains_non_overlapping(result)
    assert _foreground_mask_count(result) == 1


def test_crop_local_ios_matches_full_section_ios_on_fixture_pair() -> None:
    proposals, height, width = slice_boundary_duplicate_pair()
    left_seg = segmentation_to_binary_mask(proposals[0].segmentation)
    right_seg = segmentation_to_binary_mask(proposals[1].segmentation)

    full_left = np.zeros((height, width), dtype=bool)
    full_right = np.zeros((height, width), dtype=bool)
    oy0, ox0 = proposals[0].offset_y, proposals[0].offset_x
    oy1, ox1 = proposals[1].offset_y, proposals[1].offset_x
    lh, lw = left_seg.shape
    rh, rw = right_seg.shape
    full_left[oy0 : oy0 + lh, ox0 : ox0 + lw] = left_seg
    full_right[oy1 : oy1 + rh, ox1 : ox1 + rw] = right_seg

    intersection = int(np.count_nonzero(full_left & full_right))
    smaller = min(int(full_left.sum()), int(full_right.sum()))
    full_ios = intersection / smaller

    crop_ios = mask_ios_crop_local(
        left_seg,
        proposals[0].offset_y,
        proposals[0].offset_x,
        right_seg,
        proposals[1].offset_y,
        proposals[1].offset_x,
        left_area=int(left_seg.sum()),
        right_area=int(right_seg.sum()),
    )
    assert crop_ios == full_ios
