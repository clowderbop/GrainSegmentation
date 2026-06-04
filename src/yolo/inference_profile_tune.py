"""YOLO inference profile train selection (factorial grid from YAML, ADR 0005)."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
from collections.abc import Callable
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import Any, Iterable

import yaml

from common import yaml_validate as yv
from common.instance_metric_bundle import INSTANCE_METRIC_BUNDLE_KEYS
from common.instance_eval_report import (
    extract_instance_metric_bundle_from_report,
    extract_metric_from_report,
    load_instance_eval_report,
)
from common.test_inference import (
    YoloInferenceProfileCandidate,
    load_test_inference_recipe,
    parse_yolo_profile_candidate_mapping,
    rewrite_yolo_profile_in_recipe_text,
)
from common.variants import repo_root
from yolo.profile_tune_work import weights_path
from yolo.tiled_proposal_cache import recipe_whole_window_fingerprint, weights_sha256

VariantScorer = Callable[[str, YoloInferenceProfileCandidate, Path], Path]

_TUNE_GRID_RELATIVE = Path("configs") / "yolo_inference_profile_tune.yaml"


@dataclass(frozen=True)
class ProfileTuneGrid:
    postprocess_type: tuple[str, ...]
    match_metric: tuple[str, ...]
    match_threshold: tuple[float, ...]
    conf: tuple[float, ...]
    mask_threshold: tuple[float, ...]


@dataclass(frozen=True)
class TuneGridSpec:
    grid: ProfileTuneGrid


def tune_grid_path(path: Path | None = None) -> Path:
    return path or (repo_root() / _TUNE_GRID_RELATIVE)


def load_tune_grid(path: Path | None = None) -> TuneGridSpec:
    resolved = tune_grid_path(path)
    with resolved.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    doc = yv.require_mapping(raw, context=str(resolved))
    grid_raw = yv.require_mapping(doc.get("grid"), context="grid")
    return TuneGridSpec(
        grid=ProfileTuneGrid(
            postprocess_type=yv.require_str_list(
                grid_raw.get("postprocess_type"), context="grid.postprocess_type"
            ),
            match_metric=yv.require_str_list(
                grid_raw.get("match_metric"), context="grid.match_metric"
            ),
            match_threshold=yv.require_float_list(
                grid_raw.get("match_threshold"), context="grid.match_threshold"
            ),
            conf=yv.require_float_list(grid_raw.get("conf"), context="grid.conf"),
            mask_threshold=yv.require_float_list(
                grid_raw.get("mask_threshold"), context="grid.mask_threshold"
            ),
        ),
    )


def count_grid_candidates(spec: TuneGridSpec) -> int:
    """Number of profile points in the Cartesian product of grid axes."""
    grid = spec.grid
    return prod(
        (
            len(grid.postprocess_type),
            len(grid.match_metric),
            len(grid.match_threshold),
            len(grid.conf),
            len(grid.mask_threshold),
        )
    )


def count_detector_jobs(spec: TuneGridSpec, variant_count: int) -> int:
    """GPU detector jobs: one per (variant, conf, mask_threshold)."""
    grid = spec.grid
    return variant_count * len(grid.conf) * len(grid.mask_threshold)


def iter_detector_jobs(
    spec: TuneGridSpec, variants: tuple[str, ...]
) -> Iterable[tuple[str, float, float]]:
    """One GPU detector job per (variant, conf, mask_threshold)."""
    for variant in variants:
        for conf, mask_threshold in itertools.product(spec.grid.conf, spec.grid.mask_threshold):
            yield variant, conf, mask_threshold


def detector_job_at_index(
    spec: TuneGridSpec, variants: tuple[str, ...], array_index: int
) -> tuple[str, float, float]:
    if array_index < 1:
        raise ValueError(f"array index must be >= 1, got {array_index}")
    jobs = list(iter_detector_jobs(spec, variants))
    if array_index > len(jobs):
        raise ValueError(
            f"array index {array_index} out of range for {len(jobs)} detector jobs"
        )
    return jobs[array_index - 1]


def iter_grid_candidates(spec: TuneGridSpec) -> Iterable[YoloInferenceProfileCandidate]:
    grid = spec.grid
    for (
        postprocess_type,
        match_metric,
        match_threshold,
        conf,
        mask_threshold,
    ) in itertools.product(
        grid.postprocess_type,
        grid.match_metric,
        grid.match_threshold,
        grid.conf,
        grid.mask_threshold,
    ):
        yield YoloInferenceProfileCandidate(
            postprocess_type=postprocess_type,
            match_metric=match_metric,
            match_threshold=match_threshold,
            conf=conf,
            mask_threshold=mask_threshold,
        )


PROFILE_SELECTION_OBJECTIVE = "pq"


def variant_metric_column(metric_key: str, variant: str) -> str:
    return f"{metric_key}__{variant}"


def extract_mean_pq_from_report(report: dict[str, Any]) -> float:
    return extract_metric_from_report(report, PROFILE_SELECTION_OBJECTIVE)


def mean_pq_across_variants(variant_pq: dict[str, float]) -> float:
    if not variant_pq:
        raise ValueError("variant_pq must not be empty")
    return float(sum(variant_pq.values()) / len(variant_pq))


def select_best_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("rows must not be empty")
    return max(rows, key=lambda row: float(row["mean_pq"]))


def flatten_per_variant_bundles(
    per_variant_bundles: dict[str, dict[str, float]],
) -> dict[str, float]:
    flat: dict[str, float] = {}
    for variant, bundle in per_variant_bundles.items():
        for key, value in bundle.items():
            flat[variant_metric_column(key, variant)] = float(value)
    return flat


def load_instance_metrics_report(path: Path) -> dict[str, Any]:
    return load_instance_eval_report(path)


def score_candidate_across_variants(
    variant_reports: dict[str, Path],
) -> tuple[float, dict[str, dict[str, float]]]:
    per_variant_bundles: dict[str, dict[str, float]] = {}
    for variant, report_path in variant_reports.items():
        per_variant_bundles[variant] = extract_instance_metric_bundle_from_report(
            load_instance_metrics_report(report_path)
        )
    per_variant_pq = {
        variant: bundle[PROFILE_SELECTION_OBJECTIVE]
        for variant, bundle in per_variant_bundles.items()
    }
    return mean_pq_across_variants(per_variant_pq), per_variant_bundles


def grid_results_fieldnames(variant_names: tuple[str, ...]) -> list[str]:
    base = [
        "candidate_id",
        "postprocess_type",
        "match_metric",
        "match_threshold",
        "conf",
        "mask_threshold",
        "mean_pq",
    ]
    metric_columns = [
        variant_metric_column(key, variant)
        for variant in variant_names
        for key in INSTANCE_METRIC_BUNDLE_KEYS
    ]
    return base + metric_columns


def grid_result_row_from_candidate_scoring(
    *,
    candidate: YoloInferenceProfileCandidate,
    mean_pq: float,
    per_variant_bundles: dict[str, dict[str, float]],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "candidate_id": candidate.candidate_id(),
        **candidate.to_dict(),
        "mean_pq": mean_pq,
        **flatten_per_variant_bundles(per_variant_bundles),
    }
    return row


def write_grid_results_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    variant_names: tuple[str, ...],
) -> None:
    if not rows:
        raise ValueError("rows must not be empty")
    fieldnames = grid_results_fieldnames(variant_names)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_grid_result_row(
    path: Path,
    row: dict[str, Any],
    *,
    variant_names: tuple[str, ...],
) -> None:
    fieldnames = grid_results_fieldnames(variant_names)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_grid_results_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def tune_grid_fingerprint(grid_config: Path | None) -> str:
    resolved = tune_grid_path(grid_config)
    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    return digest


@dataclass(frozen=True)
class GridResumeContext:
    grainseg_root: Path
    run_root: Path
    grid_config: Path | None


def variant_eval_resume_meta_path(metrics_path: Path) -> Path:
    return metrics_path.with_name(f"{metrics_path.stem}.resume.json")


def variant_eval_fingerprint(
    *,
    candidate: YoloInferenceProfileCandidate,
    variant: str,
    context: GridResumeContext,
) -> dict[str, Any]:
    weights = weights_path(context.grainseg_root, variant, context.run_root)
    recipe = load_test_inference_recipe()
    return {
        "candidate_id": candidate.candidate_id(),
        "postprocess_type": candidate.postprocess_type,
        "match_metric": candidate.match_metric,
        "match_threshold": candidate.match_threshold,
        "conf": candidate.conf,
        "mask_threshold": candidate.mask_threshold,
        "variant": variant,
        "weights_sha256": weights_sha256(weights),
        "recipe_window_fingerprint": recipe_whole_window_fingerprint(recipe),
        "tune_grid_fingerprint": tune_grid_fingerprint(context.grid_config),
    }


def write_variant_eval_resume_meta(metrics_path: Path, fingerprint: dict[str, Any]) -> None:
    variant_eval_resume_meta_path(metrics_path).write_text(
        json.dumps(fingerprint, indent=2),
        encoding="utf-8",
    )


def metrics_resume_valid(metrics_path: Path, *, expected: dict[str, Any]) -> bool:
    if not metrics_path.is_file():
        return False
    meta_path = variant_eval_resume_meta_path(metrics_path)
    if not meta_path.is_file():
        return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for key, expected_value in expected.items():
        if meta.get(key) != expected_value:
            return False
    return True


def should_skip_variant_eval(
    metrics_path: Path,
    *,
    resume: bool,
    expected_fingerprint: dict[str, Any] | None,
) -> bool:
    if not resume:
        return False
    if expected_fingerprint is None:
        return metrics_path.is_file()
    return metrics_resume_valid(metrics_path, expected=expected_fingerprint)


def finalize_grid_winner(
    grid_dir: Path,
    rows: list[dict[str, Any]],
    *,
    variant_names: tuple[str, ...],
) -> YoloInferenceProfileCandidate:
    if not rows:
        raise ValueError("rows must not be empty")
    write_grid_results_csv(grid_dir / "results.csv", rows, variant_names=variant_names)
    best_row = select_best_candidate(rows)
    winner = YoloInferenceProfileCandidate(
        postprocess_type=str(best_row["postprocess_type"]),
        match_metric=str(best_row["match_metric"]),
        match_threshold=float(best_row["match_threshold"]),
        conf=float(best_row["conf"]),
        mask_threshold=float(best_row["mask_threshold"]),
    )
    write_grid_winner_json(
        grid_dir / "winner.json",
        candidate=winner,
        mean_pq=float(best_row["mean_pq"]),
        per_variant_pq={
            variant: float(best_row[variant_metric_column(PROFILE_SELECTION_OBJECTIVE, variant)])
            for variant in variant_names
        },
    )
    return winner


def run_grid_search(
    *,
    candidates: list[YoloInferenceProfileCandidate],
    variants: tuple[str, ...],
    output_dir: Path,
    score_variant: VariantScorer,
    resume: bool = False,
    resume_context: GridResumeContext | None = None,
    on_variant_scored: Callable[
        [YoloInferenceProfileCandidate, str, Path], None
    ]
    | None = None,
) -> tuple[YoloInferenceProfileCandidate, list[dict[str, object]]]:
    grid_dir = output_dir / "grid"
    results_csv = grid_dir / "results.csv"
    rows: list[dict[str, object]] = []
    if resume:
        rows = [dict(row) for row in load_grid_results_csv(results_csv)]
    completed_ids = {str(row["candidate_id"]) for row in rows}

    for candidate in candidates:
        if resume and candidate.candidate_id() in completed_ids:
            continue
        candidate_dir = grid_dir / "candidates" / candidate.candidate_id()
        variant_reports: dict[str, Path] = {}
        for variant in variants:
            variant_out = candidate_dir / variant
            metrics_path = variant_out / "instance_metrics.json"
            fingerprint = (
                variant_eval_fingerprint(
                    candidate=candidate, variant=variant, context=resume_context
                )
                if resume_context is not None
                else None
            )
            if should_skip_variant_eval(
                metrics_path, resume=resume, expected_fingerprint=fingerprint
            ):
                if not metrics_path.is_file():
                    raise FileNotFoundError(
                        f"Resume expected metrics at {metrics_path}"
                    )
                variant_reports[variant] = metrics_path
            else:
                variant_reports[variant] = score_variant(variant, candidate, variant_out)
                if resume_context is not None and fingerprint is not None:
                    write_variant_eval_resume_meta(metrics_path, fingerprint)
            if on_variant_scored is not None:
                on_variant_scored(candidate, variant, variant_reports[variant])
        mean_pq, per_variant_bundles = score_candidate_across_variants(variant_reports)
        row = grid_result_row_from_candidate_scoring(
            candidate=candidate,
            mean_pq=mean_pq,
            per_variant_bundles=per_variant_bundles,
        )
        rows.append(row)
        write_grid_results_csv(results_csv, rows, variant_names=variants)
        print(
            f"grid {candidate.candidate_id()}: mean_pq={mean_pq:.6f}",
            flush=True,
        )

    if not rows:
        raise ValueError("no grid results to finalize")
    winner = finalize_grid_winner(grid_dir, rows, variant_names=variants)
    return winner, rows


def write_grid_winner_json(
    path: Path,
    *,
    candidate: YoloInferenceProfileCandidate,
    mean_pq: float,
    per_variant_pq: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "selection_objective": PROFILE_SELECTION_OBJECTIVE,
        "mean_pq": mean_pq,
        "profile": candidate.to_dict(),
        "per_variant_pq": per_variant_pq,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def candidate_from_winner_json(payload: dict[str, Any]) -> YoloInferenceProfileCandidate:
    profile = yv.require_mapping(payload.get("profile"), context="profile")
    return parse_yolo_profile_candidate_mapping(profile, context="profile")


def load_grid_winner(path: Path) -> YoloInferenceProfileCandidate:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return candidate_from_winner_json(payload)


def profile_selection_row_path(grid_dir: Path, candidate_id: str) -> Path:
    return grid_dir / "rows" / f"{candidate_id}.json"


def write_profile_selection_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2), encoding="utf-8")


def load_profile_selection_row(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def clear_profile_selection_rows(grid_dir: Path) -> None:
    rows_dir = grid_dir / "rows"
    if not rows_dir.is_dir():
        return
    for path in rows_dir.glob("*.json"):
        path.unlink()


def rows_to_grid_results(
    rows: list[dict[str, Any]], *, variant_names: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Normalize profile selection result rows for results.csv."""
    normalized: list[dict[str, Any]] = []
    for row in rows:
        csv_row: dict[str, Any] = {
            "candidate_id": row["candidate_id"],
            "postprocess_type": row["postprocess_type"],
            "match_metric": row["match_metric"],
            "match_threshold": row["match_threshold"],
            "conf": row["conf"],
            "mask_threshold": row["mask_threshold"],
            "mean_pq": row["mean_pq"],
        }
        for variant in variant_names:
            for key in INSTANCE_METRIC_BUNDLE_KEYS:
                column = variant_metric_column(key, variant)
                csv_row[column] = row[column]
        normalized.append(csv_row)
    return normalized


def recompute_winner_from_csv(
    output_dir: Path, *, variant_names: tuple[str, ...] | None = None
) -> YoloInferenceProfileCandidate:
    """Recompute grid/winner.json from an existing grid/results.csv."""
    from common.variants import all_variant_names

    variants = variant_names or all_variant_names()
    results_csv = output_dir / "grid" / "results.csv"
    rows = load_grid_results_csv(results_csv)
    if not rows:
        raise ValueError(f"No rows in {results_csv}")
    return finalize_grid_winner(output_dir / "grid", rows, variant_names=variants)


def candidate_at_grid_index(
    spec: TuneGridSpec, array_index: int
) -> YoloInferenceProfileCandidate:
    if array_index < 1:
        raise ValueError(f"array index must be >= 1, got {array_index}")
    candidates = list(iter_grid_candidates(spec))
    if array_index > len(candidates):
        raise ValueError(
            f"array index {array_index} out of range for {len(candidates)} candidates"
        )
    return candidates[array_index - 1]


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
