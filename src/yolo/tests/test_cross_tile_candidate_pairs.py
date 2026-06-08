"""Spatial candidate-pair generation for cross-tile association (scale issue 02)."""

from __future__ import annotations

import numpy as np

import yolo.cross_tile_association as cta
from yolo.cross_tile_association import generate_association_candidate_pairs
from yolo.tests.cross_tile_association_fixtures import (
    complementary_border_partials,
    overlapping_tile_central_vs_border,
    slice_boundary_duplicate_pair,
)


def test_slice_boundary_duplicate_fixture_emits_candidate_pair() -> None:
    """INTENT: slice-boundary duplicate fixture emits the expected association candidate pair."""
    proposals, _, _ = slice_boundary_duplicate_pair()
    enriched = cta._enrich_proposals(proposals)
    pairs = generate_association_candidate_pairs(enriched)
    assert (0, 1) in pairs


def _separated_tile_proposals(count: int) -> list:
    from yolo.tests.cross_tile_association_fixtures import _proposal

    tile_size = 16
    proposals = []
    for index in range(count):
        row = index // 32
        col = index % 32
        tile_y0 = row * tile_size
        tile_x0 = col * tile_size
        mask = np.zeros((4, 4), dtype=bool)
        mask[1:3, 1:3] = True
        proposals.append(
            _proposal(
                mask,
                score=0.5,
                offset_y=tile_y0 + 6,
                offset_x=tile_x0 + 6,
                tile_y0=tile_y0,
                tile_x0=tile_x0,
                tile_y1=tile_y0 + tile_size,
                tile_x1=tile_x0 + tile_size,
            )
        )
    return proposals


def _same_tile_duplicate_clusters(num_tiles: int) -> list:
    """Two overlapping proposals per non-overlapping tile → one pair per tile, O(tiles) total."""
    from yolo.tests.cross_tile_association_fixtures import _proposal

    tile_size = 16
    mask = np.zeros((6, 6), dtype=bool)
    mask[1:5, 1:5] = True
    proposals = []
    for tile_index in range(num_tiles):
        tile_y0 = 0
        tile_x0 = tile_index * tile_size
        offset_y = 5
        offset_x = tile_x0 + 5
        for score in (0.5 + tile_index * 1e-4, 0.6 + tile_index * 1e-4):
            proposals.append(
                _proposal(
                    mask.copy(),
                    score=score,
                    offset_y=offset_y,
                    offset_x=offset_x,
                    tile_y0=tile_y0,
                    tile_x0=tile_x0,
                    tile_y1=tile_y0 + tile_size,
                    tile_x1=tile_x0 + tile_size,
                )
            )
    return proposals


def test_spatially_separated_proposals_emit_no_candidate_pairs() -> None:
    """INTENT: spatially separated proposals emit no candidate pairs despite quadratic all-pairs count."""
    proposals = _separated_tile_proposals(120)
    enriched = cta._enrich_proposals(proposals)
    pairs = generate_association_candidate_pairs(enriched)
    assert pairs == []
    assert 120 * 119 // 2 == 7140  # all-pairs would be quadratic


def test_same_tile_overlapping_fixture_emits_candidate_pair() -> None:
    """INTENT: same-tile overlapping fixture emits an association candidate pair."""
    proposals, _, _, _ = overlapping_tile_central_vs_border()
    enriched = cta._enrich_proposals(proposals)
    pairs = generate_association_candidate_pairs(enriched)
    assert (0, 1) in pairs


def test_complementary_border_partials_emit_candidate_pair() -> None:
    """INTENT: complementary border partials emit an association candidate pair."""
    proposals, _, _, _ = complementary_border_partials()
    enriched = cta._enrich_proposals(proposals)
    pairs = generate_association_candidate_pairs(enriched)
    assert (0, 1) in pairs


def test_same_tile_clusters_emit_pairs_sub_quadratically() -> None:
    """INTENT: same-tile duplicate clusters emit O(tiles) pairs, not all-pairs quadratic growth."""
    num_tiles = 80
    proposals = _same_tile_duplicate_clusters(num_tiles)
    enriched = cta._enrich_proposals(proposals)
    pairs = generate_association_candidate_pairs(enriched)
    n = len(proposals)
    all_pairs = n * (n - 1) // 2

    assert len(pairs) > 0
    assert len(pairs) == num_tiles
    assert len(pairs) * 10 < all_pairs


def test_candidate_pair_growth_is_linear_not_quadratic() -> None:
    """INTENT: candidate pair count scales linearly when tile clusters double, not quadratically."""
    pair_counts: list[int] = []
    proposal_counts: list[int] = []
    for num_tiles in (25, 50, 100):
        proposals = _same_tile_duplicate_clusters(num_tiles)
        enriched = cta._enrich_proposals(proposals)
        pairs = generate_association_candidate_pairs(enriched)
        n = len(proposals)
        pair_counts.append(len(pairs))
        proposal_counts.append(n)
        all_pairs = n * (n - 1) // 2
        assert len(pairs) == num_tiles
        assert len(pairs) > 0
        assert len(pairs) * 8 < all_pairs

    ratio_50_25 = pair_counts[1] / pair_counts[0]
    ratio_100_50 = pair_counts[2] / pair_counts[1]
    assert 1.8 <= ratio_50_25 <= 2.2
    assert 1.8 <= ratio_100_50 <= 2.2

    all_pairs_25 = proposal_counts[0] * (proposal_counts[0] - 1) // 2
    all_pairs_50 = proposal_counts[1] * (proposal_counts[1] - 1) // 2
    assert all_pairs_50 / all_pairs_25 > 3.5
