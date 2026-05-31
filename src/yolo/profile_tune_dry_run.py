"""Synthetic variant scorers for profile-tune grid dry-runs and tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from common.test_inference import YoloInferenceProfileCandidate

VariantScorer = Callable[[str, YoloInferenceProfileCandidate, Path], Path]


def dry_run_scorer(
    variant_scores: dict[tuple[str, str], float],
) -> VariantScorer:
    def score(variant: str, candidate: YoloInferenceProfileCandidate, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        key = (variant, candidate.candidate_id())
        aji = variant_scores.get(key, 0.0)
        metrics_path = out_dir / "instance_metrics.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "samples": [{"sample_id": "train", "aji": aji}],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return metrics_path

    return score
