"""YOLO profile-tune test helpers (re-export neutral fixtures from common)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from common.test_inference import YoloInferenceProfileCandidate
from common.tests.profile_tune_fixtures import (  # noqa: F401
    FakeBbox,
    FakeSahiPrediction,
    V1SahiPickleStub,
    disjoint_sahi_proposals,
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
