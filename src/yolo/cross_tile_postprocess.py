"""Canonical YOLO cross-tile postprocess (profile selection and whole predict).

Consumes **tiled detector proposals** with source tile bounds and produces a
non-overlapping **instance prediction set** or **merged instance view**.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from common.prediction_set import (
    PredictionSet,
    prediction_set_to_merged_instance_view,
    save_prediction_set,
)

from yolo.cross_tile_association import (
    TiledAssociationProposal,
    associate_tiled_proposals,
)
from yolo.tiled_proposal_cache import TiledProposalRecord


def tiled_association_proposals_from_records(
    records: Sequence[TiledProposalRecord],
) -> list[TiledAssociationProposal]:
    """Build association inputs from v3 tiled proposal cache records."""
    return [TiledAssociationProposal.from_record(record) for record in records]


def prediction_set_from_tiled_proposal_records(
    records: Sequence[TiledProposalRecord],
    *,
    height: int,
    width: int,
) -> PredictionSet:
    """Fuse tiled proposals into the canonical YOLO instance prediction set."""
    proposals = tiled_association_proposals_from_records(records)
    return associate_tiled_proposals(proposals, height=height, width=width)


def merged_instance_view_from_tiled_proposal_records(
    records: Sequence[TiledProposalRecord],
    *,
    height: int,
    width: int,
) -> np.ndarray:
    """Rasterize cross-tile association output for train PQ / diagnostics."""
    pred_set = prediction_set_from_tiled_proposal_records(
        records, height=height, width=width
    )
    return prediction_set_to_merged_instance_view(pred_set)


def materialize_cross_tile_prediction_set(
    records: Sequence[TiledProposalRecord],
    *,
    height: int,
    width: int,
    path: Path,
) -> Path:
    """Write canonical prediction set after cross-tile association."""
    pred_set = prediction_set_from_tiled_proposal_records(
        records, height=height, width=width
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    save_prediction_set(path, pred_set)
    return path


__all__ = [
    "materialize_cross_tile_prediction_set",
    "merged_instance_view_from_tiled_proposal_records",
    "prediction_set_from_tiled_proposal_records",
    "tiled_association_proposals_from_records",
]
