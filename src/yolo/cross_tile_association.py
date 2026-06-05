"""Cross-tile instance association for tiled YOLO detector proposals.

Crop-local proposal geometry, spatially bounded candidate-pair generation,
cluster-local fusion, and fixed IoS/centrality thresholds.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from yolo.phase_logging import (
    PHASE_BUILD_CANDIDATE_PAIRS,
    PHASE_CROSS_TILE_ASSOCIATION,
    PHASE_ENRICH_PROPOSALS,
    PHASE_MERGE_PREDICTIONS,
    log_phase_done,
    log_phase_start,
)

from common.prediction_set import (
    GRAIN_CLASS_ID,
    PredictionSet,
    binary_mask_to_segmentation,
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


@dataclass(frozen=True)
class _FusedCluster:
    local_mask: np.ndarray
    offset_y: int
    offset_x: int
    score: float

    @property
    def bbox_yxyx(self) -> tuple[int, int, int, int]:
        crop_h, crop_w = self.local_mask.shape
        return (
            self.offset_y,
            self.offset_x,
            self.offset_y + crop_h,
            self.offset_x + crop_w,
        )


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


def _bbox_yxyx(proposal: TiledAssociationProposal) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = proposal.bbox
    return float(y0), float(x0), float(y1), float(x1)


def _tile_bounds_yxyx(proposal: TiledAssociationProposal) -> tuple[int, int, int, int]:
    return proposal.tile_y0, proposal.tile_x0, proposal.tile_y1, proposal.tile_x1


def _rects_intersect(
    left: tuple[float, ...], right: tuple[float, ...]
) -> bool:
    ly0, lx0, ly1, lx1 = left
    ry0, rx0, ry1, rx1 = right
    return ly0 < ry1 and ry0 < ly1 and lx0 < rx1 and rx0 < lx1


def _tiles_overlap(
    left: tuple[int, int, int, int], right: tuple[int, int, int, int]
) -> bool:
    return _rects_intersect(
        (float(left[0]), float(left[1]), float(left[2]), float(left[3])),
        (float(right[0]), float(right[1]), float(right[2]), float(right[3])),
    )


def _expanded_bbox(entry: _EnrichedProposal) -> tuple[float, float, float, float]:
    y0, x0, y1, x1 = _bbox_yxyx(entry.proposal)
    if not entry.touches_border:
        return y0, x0, y1, x1
    prop = entry.proposal
    tile_h = max(prop.tile_y1 - prop.tile_y0, 1)
    tile_w = max(prop.tile_x1 - prop.tile_x0, 1)
    margin_y = max(1, int(round(tile_h * _BORDER_MARGIN_FRAC)))
    margin_x = max(1, int(round(tile_w * _BORDER_MARGIN_FRAC)))
    return (
        y0 - margin_y,
        x0 - margin_x,
        y1 + margin_y,
        x1 + margin_x,
    )


def generate_association_candidate_pairs(
    enriched: Sequence[_EnrichedProposal],
) -> list[tuple[int, int]]:
    """Return deduplicated proposal index pairs that may be slice-boundary duplicates."""
    count = len(enriched)
    if count < 2:
        return []
    bboxes = [_expanded_bbox(entry) for entry in enriched]
    tiles = [_tile_bounds_yxyx(entry.proposal) for entry in enriched]
    pairs: list[tuple[int, int]] = []
    for left_index in range(count):
        for right_index in range(left_index + 1, count):
            same_tile = tiles[left_index] == tiles[right_index]
            if not same_tile and not _tiles_overlap(
                tiles[left_index], tiles[right_index]
            ):
                continue
            if not _rects_intersect(bboxes[left_index], bboxes[right_index]):
                continue
            pairs.append((left_index, right_index))
    return pairs


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


def _cluster_crop_bounds(
    members: Sequence[_EnrichedProposal],
) -> tuple[int, int, int, int]:
    y0 = min(member.proposal.offset_y for member in members)
    x0 = min(member.proposal.offset_x for member in members)
    y1 = max(
        member.proposal.offset_y + member.local_mask.shape[0] for member in members
    )
    x1 = max(
        member.proposal.offset_x + member.local_mask.shape[1] for member in members
    )
    return y0, x0, y1, x1


def _fuse_cluster(members: Sequence[_EnrichedProposal]) -> _FusedCluster:
    y0, x0, y1, x1 = _cluster_crop_bounds(members)
    crop_h, crop_w = y1 - y0, x1 - x0
    fused = np.zeros((crop_h, crop_w), dtype=bool)
    for member in members:
        prop = member.proposal
        local_y = prop.offset_y - y0
        local_x = prop.offset_x - x0
        mask_h, mask_w = member.local_mask.shape
        fused[local_y : local_y + mask_h, local_x : local_x + mask_w] |= member.local_mask
    score = max(member.proposal.score for member in members)
    return _FusedCluster(local_mask=fused, offset_y=y0, offset_x=x0, score=score)


def _bbox_intersects(
    left: tuple[int, int, int, int], right: tuple[int, int, int, int]
) -> bool:
    return _rects_intersect(
        (float(left[0]), float(left[1]), float(left[2]), float(left[3])),
        (float(right[0]), float(right[1]), float(right[2]), float(right[3])),
    )


def _subtract_occluder_from_cluster(
    target: _FusedCluster, occluder: _FusedCluster
) -> np.ndarray:
    """Clear ``target`` pixels covered by ``occluder`` inside their bbox intersection."""
    if not _bbox_intersects(target.bbox_yxyx, occluder.bbox_yxyx):
        return target.local_mask
    ty0, tx0, ty1, tx1 = target.bbox_yxyx
    oy0, ox0, oy1, ox1 = occluder.bbox_yxyx
    iy0 = max(ty0, oy0)
    ix0 = max(tx0, ox0)
    iy1 = min(ty1, oy1)
    ix1 = min(tx1, ox1)
    clipped = target.local_mask.copy()
    target_slice = clipped[iy0 - ty0 : iy1 - ty0, ix0 - tx0 : ix1 - tx0]
    occluder_slice = occluder.local_mask[iy0 - oy0 : iy1 - oy0, ix0 - ox0 : ix1 - ox0]
    target_slice[occluder_slice] = False
    return clipped


def _resolve_cluster_overlaps_by_score(
    clusters: Sequence[_FusedCluster],
) -> list[_FusedCluster]:
    """Prefer higher-score grains where cluster bboxes overlap (score-paint semantics)."""
    ordered = sorted(clusters, key=lambda cluster: cluster.score, reverse=True)
    kept: list[_FusedCluster] = []
    for cluster in ordered:
        local_mask = cluster.local_mask
        for occluder in kept:
            local_mask = _subtract_occluder_from_cluster(
                _FusedCluster(
                    local_mask=local_mask,
                    offset_y=cluster.offset_y,
                    offset_x=cluster.offset_x,
                    score=cluster.score,
                ),
                occluder,
            )
        if local_mask.any():
            kept.append(
                _FusedCluster(
                    local_mask=local_mask,
                    offset_y=cluster.offset_y,
                    offset_x=cluster.offset_x,
                    score=cluster.score,
                )
            )
    return kept


def _build_yolo_prediction_set_from_fused_clusters(
    clusters: Sequence[_FusedCluster],
    *,
    height: int,
    width: int,
) -> PredictionSet:
    detections: list[dict[str, Any]] = []
    for cluster in _resolve_cluster_overlaps_by_score(clusters):
        crop_h, crop_w = cluster.local_mask.shape
        detections.append(
            {
                "segmentation": binary_mask_to_segmentation(
                    cluster.local_mask, height=crop_h, width=crop_w
                ),
                "offset_y": cluster.offset_y,
                "offset_x": cluster.offset_x,
                "score": float(cluster.score),
                "category_id": GRAIN_CLASS_ID,
            }
        )
    return PredictionSet(
        schema_version=1,
        height=height,
        width=width,
        producer="yolo",
        detections=tuple(detections),
    )


def associate_tiled_proposals(
    proposals: Sequence[TiledAssociationProposal],
    *,
    height: int,
    width: int,
    log_timings: bool = False,
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

    t_assoc = time.perf_counter()

    if log_timings:
        log_phase_start(PHASE_ENRICH_PROPOSALS)
    t0 = time.perf_counter()
    enriched = _enrich_proposals(proposals)
    if log_timings:
        log_phase_done(
            PHASE_ENRICH_PROPOSALS,
            time.perf_counter() - t0,
            detail=f"n={len(enriched)}",
        )

    if log_timings:
        log_phase_start(PHASE_BUILD_CANDIDATE_PAIRS)
    t0 = time.perf_counter()
    candidate_pairs = generate_association_candidate_pairs(enriched)
    if log_timings:
        log_phase_done(
            PHASE_BUILD_CANDIDATE_PAIRS,
            time.perf_counter() - t0,
            detail=f"{len(candidate_pairs)} pairs",
        )

    if log_timings:
        log_phase_start(PHASE_MERGE_PREDICTIONS)
    t0 = time.perf_counter()
    parent = list(range(len(enriched)))
    for left_index, right_index in candidate_pairs:
        if _should_associate(enriched[left_index], enriched[right_index]):
            _union_find_merge(parent, left_index, right_index)

    clusters = _cluster_members(parent, len(enriched))
    fused_clusters: list[_FusedCluster] = []
    for member_indices in clusters.values():
        members = [enriched[index] for index in member_indices]
        fused_clusters.append(_fuse_cluster(members))

    result = _build_yolo_prediction_set_from_fused_clusters(
        fused_clusters, height=height, width=width
    )
    if log_timings:
        log_phase_done(
            PHASE_MERGE_PREDICTIONS,
            time.perf_counter() - t0,
            detail=f"{len(fused_clusters)} clusters",
        )
        log_phase_done(
            PHASE_CROSS_TILE_ASSOCIATION,
            time.perf_counter() - t_assoc,
        )
    return result


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
    "generate_association_candidate_pairs",
    "mask_ios_crop_local",
]
