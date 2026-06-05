"""YOLO inference profile train selection (factorial grid from YAML, ADR 0005)."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from common import yaml_validate as yv
from common.merged_view_pq import (
    MERGED_VIEW_PQ_COUNT_KEYS,
    MERGED_VIEW_PQ_RESULT_KEYS,
    MergedViewPqResult,
    merged_view_pq_from_instance_metric_bundle,
)
from common.instance_eval_report import (
    extract_instance_metric_bundle_from_report,
    extract_metric_from_report,
    load_instance_eval_report,
)
from common.test_inference import (
    YoloInferenceProfileCandidate,
    load_test_inference_recipe,
    profile_tune_candidate_from_conf,
    profile_tune_fixed_mask_threshold,
    rewrite_yolo_conf_in_recipe_text,
)
from common.variants import repo_root
from yolo.profile_tune_work import weights_path
from yolo.tiled_proposal_cache import recipe_whole_window_fingerprint, weights_sha256

VariantScorer = Callable[[str, YoloInferenceProfileCandidate, Path], Path]

_TUNE_GRID_RELATIVE = Path("config") / "yolo_inference_profile_tune.yaml"


@dataclass(frozen=True)
class ProfileTuneGrid:
    conf: tuple[float, ...]


@dataclass(frozen=True)
class TuneGridSpec:
    grid: ProfileTuneGrid


_REMOVED_GRID_AXES = (
    "postprocess_type",
    "match_metric",
    "match_threshold",
    "mask_threshold",
)


def tune_grid_path(path: Path | None = None) -> Path:
    return path or (repo_root() / _TUNE_GRID_RELATIVE)


def load_tune_grid(path: Path | None = None) -> TuneGridSpec:
    resolved = tune_grid_path(path)
    with resolved.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    doc = yv.require_mapping(raw, context=str(resolved))
    grid_raw = yv.require_mapping(doc.get("grid"), context="grid")
    for axis in _REMOVED_GRID_AXES:
        if axis not in grid_raw:
            continue
        values = grid_raw.get(axis)
        if axis == "mask_threshold":
            if isinstance(values, list) and len(values) > 1:
                raise ValueError(
                    "grid.mask_threshold must not list multiple values; "
                    "mask threshold is fixed in config/test_inference.yaml"
                )
        raise ValueError(
            f"grid.{axis} is no longer a profile-selection axis "
            f"(removed after cross-tile postprocess); tune conf only"
        )
    return TuneGridSpec(
        grid=ProfileTuneGrid(
            conf=yv.require_float_list(grid_raw.get("conf"), context="grid.conf"),
        ),
    )


def count_grid_candidates(spec: TuneGridSpec) -> int:
    """Number of profile points (conf values only)."""
    return len(spec.grid.conf)


def count_detector_jobs(spec: TuneGridSpec, variant_count: int) -> int:
    """GPU detector array tasks: one per input configuration (variant)."""
    del spec  # grid shape does not change variant task count
    return variant_count


def detector_keys_per_variant(spec: TuneGridSpec) -> int:
    return len(spec.grid.conf)


def iter_detector_keys(spec: TuneGridSpec) -> Iterable[float]:
    """Detector conf values processed inside one variant GPU task."""
    yield from spec.grid.conf


def variant_at_detector_array_index(variants: tuple[str, ...], array_index: int) -> str:
    if array_index < 1:
        raise ValueError(f"array index must be >= 1, got {array_index}")
    if array_index > len(variants):
        raise ValueError(
            f"array index {array_index} out of range for {len(variants)} variants"
        )
    return variants[array_index - 1]


def iter_detector_jobs(
    spec: TuneGridSpec, variants: tuple[str, ...]
) -> Iterable[tuple[str, float]]:
    """Flat detector grid: every (variant, conf) cache key."""
    for variant in variants:
        for conf in iter_detector_keys(spec):
            yield variant, conf


def detector_job_at_index(
    spec: TuneGridSpec, variants: tuple[str, ...], array_index: int
) -> tuple[str, float]:
    """Resolve a flat detector grid index (tests, legacy callers)."""
    if array_index < 1:
        raise ValueError(f"array index must be >= 1, got {array_index}")
    jobs = list(iter_detector_jobs(spec, variants))
    if array_index > len(jobs):
        raise ValueError(
            f"array index {array_index} out of range for {len(jobs)} detector jobs"
        )
    return jobs[array_index - 1]


def iter_grid_candidates(spec: TuneGridSpec) -> Iterable[YoloInferenceProfileCandidate]:
    for conf in spec.grid.conf:
        yield profile_tune_candidate_from_conf(conf)


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


def _coerce_merged_view_pq_value(key: str, value: Any) -> float | int:
    if key in MERGED_VIEW_PQ_COUNT_KEYS:
        return int(round(float(value)))
    return float(value)


def merged_view_pq_result_from_grid_row(
    row: dict[str, Any],
    *,
    variant: str,
) -> MergedViewPqResult:
    result: dict[str, float | int] = {}
    for key in MERGED_VIEW_PQ_RESULT_KEYS:
        column = variant_metric_column(key, variant)
        if column not in row:
            raise KeyError(f"Missing {column!r} in grid row")
        result[key] = _coerce_merged_view_pq_value(key, row[column])
    return result  # type: ignore[return-value]


def per_variant_pq_results_from_grid_row(
    row: dict[str, Any],
    *,
    variant_names: tuple[str, ...],
) -> dict[str, MergedViewPqResult]:
    return {
        variant: merged_view_pq_result_from_grid_row(row, variant=variant)
        for variant in variant_names
    }


def mean_merged_view_pq_across_variants(
    results: list[MergedViewPqResult],
) -> dict[str, float | int]:
    if not results:
        raise ValueError("results must not be empty")
    out: dict[str, float | int] = {}
    for key in MERGED_VIEW_PQ_RESULT_KEYS:
        if key in MERGED_VIEW_PQ_COUNT_KEYS:
            out[key] = int(round(float(np.mean([r[key] for r in results]))))
        else:
            out[key] = float(np.mean([float(r[key]) for r in results]))
    return out


def flatten_per_variant_pq_results(
    per_variant_results: dict[str, MergedViewPqResult],
) -> dict[str, float | int]:
    flat: dict[str, float | int] = {}
    for variant, result in per_variant_results.items():
        for key in MERGED_VIEW_PQ_RESULT_KEYS:
            flat[variant_metric_column(key, variant)] = result[key]
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
        "conf",
        "mask_threshold",
    ]
    mean_columns = [f"mean_{key}" for key in MERGED_VIEW_PQ_RESULT_KEYS]
    per_variant_columns = [
        variant_metric_column(key, variant)
        for variant in variant_names
        for key in MERGED_VIEW_PQ_RESULT_KEYS
    ]
    return base + mean_columns + per_variant_columns


def grid_result_row_from_candidate_scoring(
    *,
    candidate: YoloInferenceProfileCandidate,
    per_variant_pq_results: dict[str, MergedViewPqResult],
) -> dict[str, Any]:
    mean_pq_fields = mean_merged_view_pq_across_variants(
        list(per_variant_pq_results.values())
    )
    row: dict[str, Any] = {
        "candidate_id": candidate.candidate_id(),
        **candidate.to_dict(),
        **{f"mean_{key}": value for key, value in mean_pq_fields.items()},
        **flatten_per_variant_pq_results(per_variant_pq_results),
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
    winner = profile_tune_candidate_from_conf(float(best_row["conf"]))
    write_grid_winner_json(
        grid_dir / "winner.json",
        candidate=winner,
        mean_pq=float(best_row["mean_pq"]),
        per_variant_pq_results=per_variant_pq_results_from_grid_row(
            best_row, variant_names=variant_names
        ),
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
            per_variant_pq_results={
                variant: merged_view_pq_from_instance_metric_bundle(bundle)
                for variant, bundle in per_variant_bundles.items()
            },
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
    per_variant_pq_results: dict[str, MergedViewPqResult],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fixed_mask = profile_tune_fixed_mask_threshold()
    payload = {
        "selection_objective": PROFILE_SELECTION_OBJECTIVE,
        "mean_pq": mean_pq,
        "conf": candidate.conf,
        "mask_threshold": fixed_mask,
        "profile_selection_axes": ["conf"],
        "fixed_mask_threshold": fixed_mask,
        "removed_grid_axes": list(_REMOVED_GRID_AXES),
        "per_variant_pq_results": per_variant_pq_results,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def candidate_from_winner_json(payload: dict[str, Any]) -> YoloInferenceProfileCandidate:
    if "conf" in payload:
        return profile_tune_candidate_from_conf(
            yv.require_float(payload.get("conf"), context="conf")
        )
    profile = yv.require_mapping(payload.get("profile"), context="profile")
    return profile_tune_candidate_from_conf(
        yv.require_float(profile.get("conf"), context="profile.conf")
    )


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
            "conf": row["conf"],
            "mask_threshold": row["mask_threshold"],
        }
        for key in MERGED_VIEW_PQ_RESULT_KEYS:
            csv_row[f"mean_{key}"] = row[f"mean_{key}"]
        for variant in variant_names:
            for key in MERGED_VIEW_PQ_RESULT_KEYS:
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
    fixed_mask = profile_tune_fixed_mask_threshold()
    recipe_path.write_text(
        rewrite_yolo_conf_in_recipe_text(
            original_text, conf=profile.conf, mask_threshold=fixed_mask
        ),
        encoding="utf-8",
    )
    load_test_inference_recipe.cache_clear()
    try:
        loaded = load_test_inference_recipe(recipe_path)
    except ValueError as exc:
        recipe_path.write_text(original_text, encoding="utf-8")
        raise ValueError(f"promoted recipe failed validation: {exc}") from exc
    promoted = loaded.yolo
    if promoted.conf != profile.conf or promoted.profile.mask_threshold != fixed_mask:
        recipe_path.write_text(original_text, encoding="utf-8")
        raise ValueError("promoted recipe does not match winning profile")
