"""YOLO inference profile train selection (staged grid search, ADR 0005)."""

from __future__ import annotations

import csv
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from common import yaml_validate as yv
from common.test_inference import (
    YoloInferenceProfileCandidate,
    load_test_inference_recipe,
    parse_yolo_profile_candidate_mapping,
    rewrite_yolo_profile_in_recipe_text,
)
from common.variants import repo_root

_TUNE_GRID_RELATIVE = Path("configs") / "yolo_inference_profile_tune.yaml"


@dataclass(frozen=True)
class Stage1FixedKnobs:
    conf: float
    mask_threshold: float


@dataclass(frozen=True)
class Stage1Grid:
    postprocess_type: tuple[str, ...]
    match_metric: tuple[str, ...]
    match_threshold: tuple[float, ...]


@dataclass(frozen=True)
class Stage2Grid:
    conf: tuple[float, ...]
    mask_threshold: tuple[float, ...]


@dataclass(frozen=True)
class TuneGridSpec:
    stage1_fixed: Stage1FixedKnobs
    stage1: Stage1Grid
    stage2: Stage2Grid


def tune_grid_path(path: Path | None = None) -> Path:
    return path or (repo_root() / _TUNE_GRID_RELATIVE)


def load_tune_grid(path: Path | None = None) -> TuneGridSpec:
    resolved = tune_grid_path(path)
    with resolved.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    doc = yv.require_mapping(raw, context=str(resolved))
    stage1_raw = yv.require_mapping(doc.get("stage1"), context="stage1")
    stage2_raw = yv.require_mapping(doc.get("stage2"), context="stage2")
    recipe = load_test_inference_recipe()
    stage1_fixed = Stage1FixedKnobs(
        conf=recipe.yolo.conf,
        mask_threshold=recipe.yolo.profile.mask_threshold,
    )
    return TuneGridSpec(
        stage1_fixed=stage1_fixed,
        stage1=Stage1Grid(
            postprocess_type=yv.require_str_list(
                stage1_raw.get("postprocess_type"), context="stage1.postprocess_type"
            ),
            match_metric=yv.require_str_list(
                stage1_raw.get("match_metric"), context="stage1.match_metric"
            ),
            match_threshold=yv.require_float_list(
                stage1_raw.get("match_threshold"), context="stage1.match_threshold"
            ),
        ),
        stage2=Stage2Grid(
            conf=yv.require_float_list(stage2_raw.get("conf"), context="stage2.conf"),
            mask_threshold=yv.require_float_list(
                stage2_raw.get("mask_threshold"), context="stage2.mask_threshold"
            ),
        ),
    )


def iter_stage1_candidates(spec: TuneGridSpec) -> Iterable[YoloInferenceProfileCandidate]:
    fixed = spec.stage1_fixed
    for ppt, metric, threshold in itertools.product(
        spec.stage1.postprocess_type,
        spec.stage1.match_metric,
        spec.stage1.match_threshold,
    ):
        yield YoloInferenceProfileCandidate(
            postprocess_type=ppt,
            match_metric=metric,
            match_threshold=threshold,
            conf=fixed.conf,
            mask_threshold=fixed.mask_threshold,
        )


def iter_stage2_candidates(
    spec: TuneGridSpec, stage1_winner: YoloInferenceProfileCandidate
) -> Iterable[YoloInferenceProfileCandidate]:
    for conf, mask_threshold in itertools.product(spec.stage2.conf, spec.stage2.mask_threshold):
        yield YoloInferenceProfileCandidate(
            postprocess_type=stage1_winner.postprocess_type,
            match_metric=stage1_winner.match_metric,
            match_threshold=stage1_winner.match_threshold,
            conf=conf,
            mask_threshold=mask_threshold,
        )


def extract_mean_aji_from_report(report: dict[str, Any]) -> float:
    mean = report.get("mean")
    if isinstance(mean, dict) and "aji" in mean:
        return float(mean["aji"])
    samples = report.get("samples")
    if isinstance(samples, list) and samples:
        aji_values = [
            float(row["aji"])
            for row in samples
            if isinstance(row, dict) and "aji" in row
        ]
        if aji_values:
            return float(sum(aji_values) / len(aji_values))
    raise ValueError("instance metrics report has no AJI in mean or samples")


def mean_aji_across_variants(variant_scores: dict[str, float]) -> float:
    if not variant_scores:
        raise ValueError("variant_scores must not be empty")
    return float(sum(variant_scores.values()) / len(variant_scores))


def select_best_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("rows must not be empty")
    return max(rows, key=lambda row: float(row["mean_aji"]))


def load_instance_metrics_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def score_candidate_across_variants(
    variant_reports: dict[str, Path],
) -> tuple[float, dict[str, float]]:
    per_variant: dict[str, float] = {}
    for variant, report_path in variant_reports.items():
        per_variant[variant] = extract_mean_aji_from_report(
            load_instance_metrics_report(report_path)
        )
    return mean_aji_across_variants(per_variant), per_variant


def write_stage_results_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    variant_names: tuple[str, ...],
) -> None:
    if not rows:
        raise ValueError("rows must not be empty")
    fieldnames = [
        "candidate_id",
        "postprocess_type",
        "match_metric",
        "match_threshold",
        "conf",
        "mask_threshold",
        "mean_aji",
    ] + [f"aji__{variant}" for variant in variant_names]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_stage_winner_json(
    path: Path,
    *,
    stage: int,
    candidate: YoloInferenceProfileCandidate,
    mean_aji: float,
    per_variant: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage,
        "mean_aji": mean_aji,
        "profile": candidate.to_dict(),
        "per_variant_aji": per_variant,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def candidate_from_winner_json(payload: dict[str, Any]) -> YoloInferenceProfileCandidate:
    profile = yv.require_mapping(payload.get("profile"), context="profile")
    return parse_yolo_profile_candidate_mapping(profile, context="profile")


def load_stage_winner(
    path: Path, *, expected_stage: int | None = None
) -> YoloInferenceProfileCandidate:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if expected_stage is not None:
        stage = payload.get("stage")
        if stage != expected_stage:
            raise ValueError(
                f"winner JSON must be from tune stage {expected_stage}, got {stage!r}"
            )
    return candidate_from_winner_json(payload)


def promote_profile_to_recipe(
    profile: YoloInferenceProfileCandidate,
    recipe_path: Path,
) -> None:
    original_text = recipe_path.read_text(encoding="utf-8")
    recipe_path.write_text(
        rewrite_yolo_profile_in_recipe_text(original_text, profile),
        encoding="utf-8",
    )
    load_test_inference_recipe.cache_clear()
    try:
        loaded = load_test_inference_recipe(recipe_path)
    except ValueError as exc:
        recipe_path.write_text(original_text, encoding="utf-8")
        raise ValueError(f"promoted recipe failed validation: {exc}") from exc
    promoted = loaded.yolo
    if (
        promoted.conf != profile.conf
        or promoted.profile.mask_threshold != profile.mask_threshold
        or promoted.profile.postprocess_type != profile.postprocess_type
        or promoted.profile.match_metric != profile.match_metric
        or promoted.profile.match_threshold != profile.match_threshold
    ):
        recipe_path.write_text(original_text, encoding="utf-8")
        raise ValueError("promoted recipe does not match winning profile")
