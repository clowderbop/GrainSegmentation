"""YOLO profile-tune test helpers (re-export neutral fixtures from common)."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np

from common.instance_metric_bundle import INSTANCE_METRIC_BUNDLE_KEYS
from common.test_inference import YoloInferenceProfileCandidate


def constant_metric_bundle(value: float) -> dict[str, float]:
    return {key: float(value) for key in INSTANCE_METRIC_BUNDLE_KEYS}


def instance_metrics_report_for_pq(pq: float) -> dict[str, object]:
    return {"samples": [{"sample_id": "train", **constant_metric_bundle(pq)}]}
from common.tests.profile_tune_fixtures import (  # noqa: F401
    FakeBbox,
    FakeSahiPrediction,
    V1SahiPickleStub,
    disjoint_sahi_proposals,
    disjoint_tile_local_proposals,
    overlapping_sahi_proposals,
    tiny_train_gt_map,
)


def candidate_for_variant(variant: str) -> YoloInferenceProfileCandidate:
    """One distinct grid point per registry variant (ADR 0005 parity fixtures)."""
    by_variant: dict[str, YoloInferenceProfileCandidate] = {
        "PPL": YoloInferenceProfileCandidate(
            postprocess_type="GREEDYNMM",
            match_metric="IOS",
            match_threshold=0.4,
            conf=0.15,
            mask_threshold=0.4,
        ),
        "PPLPPXblend": YoloInferenceProfileCandidate(
            postprocess_type="NMM",
            match_metric="IOU",
            match_threshold=0.5,
            conf=0.25,
            mask_threshold=0.5,
        ),
        "PPL+PPXblend": YoloInferenceProfileCandidate(
            postprocess_type="GREEDYNMM",
            match_metric="IOU",
            match_threshold=0.6,
            conf=0.35,
            mask_threshold=0.6,
        ),
        "PPL+AllPPX": YoloInferenceProfileCandidate(
            postprocess_type="NMM",
            match_metric="IOS",
            match_threshold=0.5,
            conf=0.25,
            mask_threshold=0.5,
        ),
    }
    return by_variant[variant]


def collect_v2_records_with_mocked_slices(
    section_height: int,
    section_width: int,
    slice_yields: Sequence[tuple[int, int, int, int, list[Any]]],
    *,
    mask_threshold: float,
    slice_height: int,
    slice_width: int,
) -> list:
    """Run ``collect_tiled_detector_proposals`` with inference mocked; tile-local preds only."""
    from yolo.tiled_proposal_cache import collect_tiled_detector_proposals

    image = np.zeros((section_height, section_width, 3), dtype=np.uint8)

    def fake_iter(
        _image: np.ndarray, _model: object, *, full_shape: list[int] | None, **_kwargs: object
    ) -> Iterator[tuple[int, int, int, int, list[Any]]]:
        assert full_shape is None
        yield from slice_yields

    with patch(
        "yolo.sliced_detection.iter_whole_slice_predictions",
        side_effect=fake_iter,
    ):
        return collect_tiled_detector_proposals(
            image,
            MagicMock(),
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=0.0,
            overlap_width_ratio=0.0,
            mask_threshold=mask_threshold,
        )


def v2_records_from_overlapping_masks(height: int, width: int) -> list:
    """v2 records for overlapping proposals (slice-merge mask union path)."""
    from yolo.tiled_proposal_cache import tiled_proposal_record_from_binary_mask

    return [
        tiled_proposal_record_from_binary_mask(
            np.asarray(pred.mask.bool_mask, dtype=bool),
            score=float(pred.score.value),
        )
        for pred in overlapping_sahi_proposals(height, width)
    ]


def v2_records_from_disjoint_via_collector(
    height: int, width: int, *, mask_threshold: float
) -> list:
    """Build v2 records via collector encode on tile-local fixture masks."""
    preds = disjoint_tile_local_proposals(height, width)
    return collect_v2_records_with_mocked_slices(
        height,
        width,
        [(0, 0, width, height, preds)],
        mask_threshold=mask_threshold,
        slice_height=height,
        slice_width=width,
    )


def write_on_disk_v1_proposal_cache(
    cache_dir: Path, *, meta: dict[str, object]
) -> None:
    """Write a schema v1 cache layout (dense SAHI pickle + v1 meta sidecar)."""
    import json
    import pickle

    height = int(meta["height"])
    width = int(meta["width"])
    dense = np.zeros((height, width), dtype=bool)
    dense[0:4, 0:4] = True
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    v1_meta = dict(meta)
    v1_meta["schema_version"] = 1
    with (cache_dir / "proposals.pkl").open("wb") as handle:
        pickle.dump([V1SahiPickleStub(bool_mask=dense)], handle, protocol=pickle.HIGHEST_PROTOCOL)
    (cache_dir / "proposals.meta.json").write_text(
        json.dumps(v1_meta, indent=2), encoding="utf-8"
    )
