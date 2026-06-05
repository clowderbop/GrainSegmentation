"""Synthetic tiled-proposal fixtures for cross-tile association tests."""

from __future__ import annotations

import numpy as np

from yolo.cross_tile_association import TiledAssociationProposal
from yolo.tiled_proposal_cache import tiled_proposal_record_from_tile_mask


def _proposal(
    mask: np.ndarray,
    *,
    score: float,
    offset_y: int,
    offset_x: int,
    tile_y0: int,
    tile_x0: int,
    tile_y1: int,
    tile_x1: int,
) -> TiledAssociationProposal:
    record = tiled_proposal_record_from_tile_mask(
        mask,
        score=score,
        offset_y=offset_y,
        offset_x=offset_x,
    )
    return TiledAssociationProposal.from_record(
        record,
        tile_y0=tile_y0,
        tile_x0=tile_x0,
        tile_y1=tile_y1,
        tile_x1=tile_x1,
    )


def slice_boundary_duplicate_pair(
    *,
    section_height: int = 24,
    section_width: int = 24,
    tile_size: int = 16,
    stride: int = 8,
) -> tuple[list[TiledAssociationProposal], int, int]:
    """Same grain detected as a central tile-0 mask and a tile-1 border partial."""
    tile0 = (0, 0, tile_size, tile_size)
    tile1 = (0, stride, tile_size + stride, tile_size)
    grain = np.zeros((section_height, section_width), dtype=bool)
    grain[6:14, 6:14] = True
    central = grain[6:14, 6:14]
    partial = grain[6:14, stride:14]
    return (
        [
            _proposal(
                central,
                score=0.72,
                offset_y=6,
                offset_x=6,
                tile_y0=tile0[0],
                tile_x0=tile0[1],
                tile_y1=tile0[2],
                tile_x1=tile0[3],
            ),
            _proposal(
                partial,
                score=0.81,
                offset_y=6,
                offset_x=stride,
                tile_y0=tile1[0],
                tile_x0=tile1[1],
                tile_y1=tile1[2],
                tile_x1=tile1[3],
            ),
        ],
        section_height,
        section_width,
    )


def adjacent_distinct_grains(
    *,
    section_height: int = 24,
    section_width: int = 24,
    tile_size: int = 16,
    stride: int = 8,
) -> tuple[list[TiledAssociationProposal], int, int]:
    """Two grains straddling the tile overlap seam without mask intersection."""
    tile0 = (0, 0, tile_size, tile_size)
    tile1 = (0, stride, tile_size + stride, tile_size)
    left = np.zeros((8, 6), dtype=bool)
    left[:, :] = True
    right = np.zeros((8, 6), dtype=bool)
    right[:, :] = True
    return (
        [
            _proposal(
                left,
                score=0.7,
                offset_y=6,
                offset_x=4,
                tile_y0=tile0[0],
                tile_x0=tile0[1],
                tile_y1=tile0[2],
                tile_x1=tile0[3],
            ),
            _proposal(
                right,
                score=0.75,
                offset_y=6,
                offset_x=12,
                tile_y0=tile1[0],
                tile_x0=tile1[1],
                tile_y1=tile1[2],
                tile_x1=tile1[3],
            ),
        ],
        section_height,
        section_width,
    )


def overlapping_tile_central_vs_border(
    *,
    section_height: int = 24,
    section_width: int = 24,
    tile_size: int = 16,
) -> tuple[list[TiledAssociationProposal], int, int, np.ndarray]:
    """Duplicate detections in one tile: central full mask vs border-trimmed partial."""
    tile = (0, 0, tile_size, tile_size)
    full = np.zeros((10, 10), dtype=bool)
    full[:, :] = True
    partial = full.copy()
    partial[:, :3] = False
    expected = np.zeros((section_height, section_width), dtype=bool)
    expected[5:15, 5:15] = True
    return (
        [
            _proposal(
                partial,
                score=0.95,
                offset_y=5,
                offset_x=5,
                tile_y0=tile[0],
                tile_x0=tile[1],
                tile_y1=tile[2],
                tile_x1=tile[3],
            ),
            _proposal(
                full,
                score=0.55,
                offset_y=5,
                offset_x=5,
                tile_y0=tile[0],
                tile_x0=tile[1],
                tile_y1=tile[2],
                tile_x1=tile[3],
            ),
        ],
        section_height,
        section_width,
        expected,
    )


def complementary_border_partials(
    *,
    section_height: int = 24,
    section_width: int = 24,
    tile_size: int = 16,
    stride: int = 8,
) -> tuple[list[TiledAssociationProposal], int, int, np.ndarray]:
    """Same grain as non-overlapping edge partials from adjacent tiles."""
    tile0 = (0, 0, tile_size, tile_size)
    tile1 = (0, stride, tile_size + stride, tile_size)
    left = np.zeros((8, 6), dtype=bool)
    left[:, :] = True
    right = np.zeros((8, 6), dtype=bool)
    right[:, :] = True
    expected = np.zeros((section_height, section_width), dtype=bool)
    expected[6:14, 4:14] = True
    return (
        [
            _proposal(
                left,
                score=0.6,
                offset_y=6,
                offset_x=4,
                tile_y0=tile0[0],
                tile_x0=tile0[1],
                tile_y1=tile0[2],
                tile_x1=tile0[3],
            ),
            _proposal(
                right,
                score=0.65,
                offset_y=6,
                offset_x=stride,
                tile_y0=tile1[0],
                tile_x0=tile1[1],
                tile_y1=tile1[2],
                tile_x1=tile1[3],
            ),
        ],
        section_height,
        section_width,
        expected,
    )


def near_boundary_low_overlap_pair(
    *,
    section_height: int = 24,
    section_width: int = 24,
    tile_size: int = 16,
    stride: int = 8,
) -> tuple[list[TiledAssociationProposal], int, int]:
    """Close bboxes across tiles with negligible mask overlap (over-merge guard)."""
    tile0 = (0, 0, tile_size, tile_size)
    tile1 = (0, stride, tile_size + stride, tile_size)
    left = np.zeros((8, 5), dtype=bool)
    left[:, :] = True
    right = np.zeros((8, 5), dtype=bool)
    right[:, :] = True
    return (
        [
            _proposal(
                left,
                score=0.66,
                offset_y=6,
                offset_x=7,
                tile_y0=tile0[0],
                tile_x0=tile0[1],
                tile_y1=tile0[2],
                tile_x1=tile0[3],
            ),
            _proposal(
                right,
                score=0.67,
                offset_y=6,
                offset_x=12,
                tile_y0=tile1[0],
                tile_x0=tile1[1],
                tile_y1=tile1[2],
                tile_x1=tile1[3],
            ),
        ],
        section_height,
        section_width,
    )
