"""Cross-tile instance association for tiled YOLO detector proposals.

Crop-local proposal geometry with all-pairs association and fixed IoS/centrality
thresholds. Spatial indexing and cluster-local fusion crops may be added later.
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

    Association input built from ``TiledProposalRecord`` (schema v3 tile bounds).
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
    def from_record(cls, record: TiledProposalRecord) -> TiledAssociationProposal:
        try:
            tile_y0 = int(record["tile_y0"])
            tile_x0 = int(record["tile_x0"])
            tile_y1 = int(record["tile_y1"])
            tile_x1 = int(record["tile_x1"])
        except KeyError as exc:
            raise ValueError(
                "Tiled proposal record missing source tile bounds metadata; "
                "re-run detector jobs to produce schema_version 3 caches"
            ) from exc
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
    local_mask: np.ndarray
    area: int
    centroid_y: float
    centroid_x: float
    centrality: float
    touches_border: bool


def mask_ios_crop_local(
    left_mask: np.ndarray,
    left_offset_y: int,
    left_offset_x: int,
    right_mask: np.ndarray,
    right_offset_y: int,
    right_offset_x: int,
    *,
    left_area: int,
    right_area: int,
) -> float:
    """Intersection-over-smaller for two crop-local masks in whole-image coordinates."""
    left_h, left_w = left_mask.shape
    right_h, right_w = right_mask.shape
    left_y1 = left_offset_y + left_h
    left_x1 = left_offset_x + left_w
    right_y1 = right_offset_y + right_h
    right_x1 = right_offset_x + right_w
    intersect_y0 = max(left_offset_y, right_offset_y)
    intersect_x0 = max(left_offset_x, right_offset_x)
    intersect_y1 = min(left_y1, right_y1)
    intersect_x1 = min(left_x1, right_x1)
    if intersect_y0 >= intersect_y1 or intersect_x0 >= intersect_x1:
        return 0.0
    left_slice = left_mask[
        intersect_y0 - left_offset_y : intersect_y1 - left_offset_y,
        intersect_x0 - left_offset_x : intersect_x1 - left_offset_x,
    ]
    right_slice = right_mask[
        intersect_y0 - right_offset_y : intersect_y1 - right_offset_y,
        intersect_x0 - right_offset_x : intersect_x1 - right_offset_x,
    ]
    intersection = int(np.count_nonzero(left_slice & right_slice))
    if intersection == 0:
        return 0.0
    smaller = min(left_area, right_area)
    if smaller == 0:
        return 0.0
    return intersection / smaller


def _mask_centroid_whole_image(
    local_mask: np.ndarray, *, offset_y: int, offset_x: int
) -> tuple[float, float]:
    ys, xs = np.where(local_mask)
    if ys.size == 0:
        return 0.0, 0.0
    return float(ys.mean() + offset_y), float(xs.mean() + offset_x)


def _centrality(
    proposal: TiledAssociationProposal,
    local_mask: np.ndarray,
    *,
    offset_y: int,
    offset_x: int,
) -> float:
    tile_h = max(proposal.tile_y1 - proposal.tile_y0, 1)
    tile_w = max(proposal.tile_x1 - proposal.tile_x0, 1)
    tile_cy = proposal.tile_y0 + tile_h / 2.0
    tile_cx = proposal.tile_x0 + tile_w / 2.0
    cy, cx = _mask_centroid_whole_image(local_mask, offset_y=offset_y, offset_x=offset_x)
    dy = abs(cy - tile_cy) / (tile_h / 2.0)
    dx = abs(cx - tile_cx) / (tile_w / 2.0)
    distance = min(1.0, (dy * dy + dx * dx) ** 0.5)
    return 1.0 - distance


def _touches_tile_border(
    proposal: TiledAssociationProposal,
    local_mask: np.ndarray,
    *,
    offset_y: int,
    offset_x: int,
) -> bool:
    ys, xs = np.where(local_mask)
    if ys.size == 0:
        return False
    tile_h = proposal.tile_y1 - proposal.tile_y0
    tile_w = proposal.tile_x1 - proposal.tile_x0
    margin_y = max(1, int(round(tile_h * _BORDER_MARGIN_FRAC)))
    margin_x = max(1, int(round(tile_w * _BORDER_MARGIN_FRAC)))
    y0, y1 = int(ys.min()) + offset_y, int(ys.max()) + offset_y
    x0, x1 = int(xs.min()) + offset_x, int(xs.max()) + offset_x
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


def _crop_local_ios(left: _EnrichedProposal, right: _EnrichedProposal) -> float:
    left_prop = left.proposal
    right_prop = right.proposal
    return mask_ios_crop_local(
        left.local_mask,
        left_prop.offset_y,
        left_prop.offset_x,
        right.local_mask,
        right_prop.offset_y,
        right_prop.offset_x,
        left_area=left.area,
        right_area=right.area,
    )


def _place_local_mask_in_section(
    local_mask: np.ndarray,
    *,
    offset_y: int,
    offset_x: int,
    height: int,
    width: int,
) -> np.ndarray:
    plane = np.zeros((height, width), dtype=bool)
    crop_h, crop_w = local_mask.shape
    plane[offset_y : offset_y + crop_h, offset_x : offset_x + crop_w] = local_mask
    return plane


def _should_associate(left: _EnrichedProposal, right: _EnrichedProposal) -> bool:
    ios = _crop_local_ios(left, right)
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
    *,
    height: int,
    width: int,
) -> tuple[np.ndarray, float]:
    fused = np.zeros((height, width), dtype=bool)
    for member in members:
        prop = member.proposal
        placed = _place_local_mask_in_section(
            member.local_mask,
            offset_y=prop.offset_y,
            offset_x=prop.offset_x,
            height=height,
            width=width,
        )
        fused |= placed
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

    enriched = _enrich_proposals(proposals)

    parent = list(range(len(enriched)))
    for left_index in range(len(enriched)):
        for right_index in range(left_index + 1, len(enriched)):
            if _should_associate(enriched[left_index], enriched[right_index]):
                _union_find_merge(parent, left_index, right_index)

    clusters = _cluster_members(parent, len(enriched))
    fused_clusters: list[tuple[np.ndarray, float]] = []
    for member_indices in clusters.values():
        members = [enriched[index] for index in member_indices]
        fused_clusters.append(_fuse_cluster(members, height=height, width=width))

    instance_map, scores = _rasterize_clusters_non_overlapping(
        fused_clusters, height=height, width=width
    )
    return build_yolo_prediction_set_from_instance_map(
        instance_map,
        score_for_label=lambda label_id: float(scores[label_id - 1]),
    )


def _enrich_proposals(
    proposals: Sequence[TiledAssociationProposal],
) -> list[_EnrichedProposal]:
    enriched: list[_EnrichedProposal] = []
    for index, proposal in enumerate(proposals):
        local_mask = segmentation_to_binary_mask(proposal.segmentation)
        area = int(local_mask.sum())
        offset_y, offset_x = proposal.offset_y, proposal.offset_x
        centroid_y, centroid_x = _mask_centroid_whole_image(
            local_mask, offset_y=offset_y, offset_x=offset_x
        )
        enriched.append(
            _EnrichedProposal(
                index=index,
                proposal=proposal,
                local_mask=local_mask,
                area=area,
                centroid_y=centroid_y,
                centroid_x=centroid_x,
                centrality=_centrality(
                    proposal, local_mask, offset_y=offset_y, offset_x=offset_x
                ),
                touches_border=_touches_tile_border(
                    proposal, local_mask, offset_y=offset_y, offset_x=offset_x
                ),
            )
        )
    return enriched


__all__ = [
    "TiledAssociationProposal",
    "associate_tiled_proposals",
    "mask_ios_crop_local",
]
