"""Synthetic variant scorers for profile-tune grid dry-runs and tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from common.instance_metric_bundle import INSTANCE_METRIC_BUNDLE_KEYS
from common.test_inference import YoloInferenceProfileCandidate

VariantScorer = Callable[[str, YoloInferenceProfileCandidate, Path], Path]


def dry_run_scorer(
    variant_scores: dict[tuple[str, str], float],
) -> VariantScorer:
    def score(variant: str, candidate: YoloInferenceProfileCandidate, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        key = (variant, candidate.candidate_id())
        pq = variant_scores.get(key, 0.0)
        sample = {"sample_id": "train", **{metric: pq for metric in INSTANCE_METRIC_BUNDLE_KEYS}}
        metrics_path = out_dir / "instance_metrics.json"
        metrics_path.write_text(
            json.dumps({"samples": [sample]}, indent=2),
            encoding="utf-8",
        )
        return metrics_path

    return score
