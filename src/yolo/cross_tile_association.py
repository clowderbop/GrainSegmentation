"""Cross-tile instance association for tiled YOLO detector proposals.

Fixture-slice implementation: all-pairs association on full-section masks.
Spatial indexing and cluster-local fusion crops are deferred to production wiring.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from common.prediction_set import (
    PredictionSet,
    build_yolo_prediction_set_from_instance_map,
    segmentation_to_binary_mask,
)
from yolo.tiled_proposal_cache import TiledProposalRecord

# Fixed association thresholds (not profile-selection grid axes).
_MIN_PAIR_IOS = 0.35
_BORDER_PARTIAL_IOS = 0.15
_BORDER_MARGIN_FRAC = 0.08
_MIN_CENTRALITY_GAP_FOR_BORDER_MERGE = 0.12


@dataclass(frozen=True)
class TiledAssociationProposal:
    """One detector proposal with its source tile bounds in whole-image coordinates.

    Slice-local input type: wraps ``TiledProposalRecord`` fields plus tile bounds.
    Production wiring should unify with the v2 cache record rather than duplicating.
    """

    score: float
    bbox: list[float]
    segmentation: dict[str, Any]
    offset_y: int
    offset_x: int
    tile_y0: int
    tile_x0: int
    tile_y1: int
    tile_x1: int

    @classmethod
    def from_record(
        cls,
        record: TiledProposalRecord,
        *,
        tile_y0: int,
        tile_x0: int,
        tile_y1: int,
        tile_x1: int,
    ) -> TiledAssociationProposal:
        return cls(
            score=record["score"],
            bbox=record["bbox"],
            segmentation=record["segmentation"],
            offset_y=record["offset_y"],
            offset_x=record["offset_x"],
            tile_y0=tile_y0,
            tile_x0=tile_x0,
            tile_y1=tile_y1,
            tile_x1=tile_x1,
        )


@dataclass(frozen=True)
class _EnrichedProposal:
    index: int
    proposal: TiledAssociationProposal
    mask: np.ndarray
    area: int
    centrality: float
    touches_border: bool


def _full_mask(proposal: TiledAssociationProposal, *, height: int, width: int) -> np.ndarray:
    plane = np.zeros((height, width), dtype=bool)
    crop = segmentation_to_binary_mask(proposal.segmentation)
    oy, ox = proposal.offset_y, proposal.offset_x
    ch, cw = crop.shape
    plane[oy : oy + ch, ox : ox + cw] = crop
    return plane


def _mask_ios(left: np.ndarray, right: np.ndarray) -> float:
    intersection = int(np.count_nonzero(left & right))
    if intersection == 0:
        return 0.0
    smaller = min(int(left.sum()), int(right.sum()))
    if smaller == 0:
        return 0.0
    return intersection / smaller


def _mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return 0.0, 0.0
    return float(ys.mean()), float(xs.mean())


def _centrality(
    proposal: TiledAssociationProposal, mask: np.ndarray
) -> float:
    tile_h = max(proposal.tile_y1 - proposal.tile_y0, 1)
    tile_w = max(proposal.tile_x1 - proposal.tile_x0, 1)
    tile_cy = proposal.tile_y0 + tile_h / 2.0
    tile_cx = proposal.tile_x0 + tile_w / 2.0
    cy, cx = _mask_centroid(mask)
    dy = abs(cy - tile_cy) / (tile_h / 2.0)
    dx = abs(cx - tile_cx) / (tile_w / 2.0)
    distance = min(1.0, (dy * dy + dx * dx) ** 0.5)
    return 1.0 - distance


def _touches_tile_border(
    proposal: TiledAssociationProposal, mask: np.ndarray
) -> bool:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return False
    tile_h = proposal.tile_y1 - proposal.tile_y0
    tile_w = proposal.tile_x1 - proposal.tile_x0
    margin_y = max(1, int(round(tile_h * _BORDER_MARGIN_FRAC)))
    margin_x = max(1, int(round(tile_w * _BORDER_MARGIN_FRAC)))
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    local_y0 = y0 - proposal.tile_y0
    local_y1 = y1 - proposal.tile_y0
    local_x0 = x0 - proposal.tile_x0
    local_x1 = x1 - proposal.tile_x0
    return (
        local_y0 <= margin_y
        or local_y1 >= tile_h - 1 - margin_y
        or local_x0 <= margin_x
        or local_x1 >= tile_w - 1 - margin_x
    )


def _should_associate(left: _EnrichedProposal, right: _EnrichedProposal) -> bool:
    ios = _mask_ios(left.mask, right.mask)
    if ios >= _MIN_PAIR_IOS:
        return True
    if ios < _BORDER_PARTIAL_IOS:
        return False
    border_pair = left.touches_border or right.touches_border
    if not border_pair:
        return False
    if left.touches_border and right.touches_border:
        return True
    centrality_gap = abs(left.centrality - right.centrality)
    return centrality_gap >= _MIN_CENTRALITY_GAP_FOR_BORDER_MERGE


def _union_find_parent(parent: list[int], index: int) -> int:
    while parent[index] != index:
        parent[index] = parent[parent[index]]
        index = parent[index]
    return index


def _union_find_merge(parent: list[int], left: int, right: int) -> None:
    root_left = _union_find_parent(parent, left)
    root_right = _union_find_parent(parent, right)
    if root_left != root_right:
        parent[root_right] = root_left


def _cluster_members(parent: list[int], count: int) -> dict[int, list[int]]:
    clusters: dict[int, list[int]] = {}
    for index in range(count):
        root = _union_find_parent(parent, index)
        clusters.setdefault(root, []).append(index)
    return clusters


def _fuse_cluster(
    members: Sequence[_EnrichedProposal],
) -> tuple[np.ndarray, float]:
    fused = np.zeros_like(members[0].mask)
    for member in members:
        fused |= member.mask
    score = max(member.proposal.score for member in members)
    return fused, score


def _rasterize_clusters_non_overlapping(
    fused_clusters: list[tuple[np.ndarray, float]],
    *,
    height: int,
    width: int,
) -> tuple[np.ndarray, list[float]]:
    order = sorted(
        range(len(fused_clusters)),
        key=lambda index: fused_clusters[index][1],
    )
    instance_map = np.zeros((height, width), dtype=np.int32)
    scores: list[float] = []
    for label, cluster_index in enumerate(order, start=1):
        mask, score = fused_clusters[cluster_index]
        instance_map[mask] = label
        scores.append(score)
    return instance_map, scores


def associate_tiled_proposals(
    proposals: Sequence[TiledAssociationProposal],
    *,
    height: int,
    width: int,
) -> PredictionSet:
    """Fuse tiled detector proposals into a canonical non-overlapping prediction set."""
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid section size: {height}x{width}")
    if not proposals:
        return PredictionSet(
            schema_version=1,
            height=height,
            width=width,
            producer="yolo",
            detections=(),
        )

    enriched = _enrich_proposals(proposals, height=height, width=width)

    parent = list(range(len(enriched)))
    for left_index in range(len(enriched)):
        for right_index in range(left_index + 1, len(enriched)):
            if _should_associate(enriched[left_index], enriched[right_index]):
                _union_find_merge(parent, left_index, right_index)

    clusters = _cluster_members(parent, len(enriched))
    fused_clusters: list[tuple[np.ndarray, float]] = []
    for member_indices in clusters.values():
        members = [enriched[index] for index in member_indices]
        fused_clusters.append(_fuse_cluster(members))

    instance_map, scores = _rasterize_clusters_non_overlapping(
        fused_clusters, height=height, width=width
    )
    return build_yolo_prediction_set_from_instance_map(
        instance_map,
        score_for_label=lambda label_id: float(scores[label_id - 1]),
    )


def _enrich_proposals(
    proposals: Sequence[TiledAssociationProposal],
    *,
    height: int,
    width: int,
) -> list[_EnrichedProposal]:
    enriched: list[_EnrichedProposal] = []
    for index, proposal in enumerate(proposals):
        mask = _full_mask(proposal, height=height, width=width)
        enriched.append(
            _EnrichedProposal(
                index=index,
                proposal=proposal,
                mask=mask,
                area=int(mask.sum()),
                centrality=_centrality(proposal, mask),
                touches_border=_touches_tile_border(proposal, mask),
            )
        )
    return enriched


__all__ = ["TiledAssociationProposal", "associate_tiled_proposals"]
