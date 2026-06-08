"""YOLO inference profile train selection (factorial grid from YAML, ADR 0005)."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from common import yaml_validate as yv
from common.merged_view_pq import (
    MERGED_VIEW_PQ_RESULT_KEYS,
    MergedViewPqResult,
    flatten_merged_view_pq_results_by_suffix,
    mean_merged_view_pq_results,
    merged_view_pq_column_name,
    merged_view_pq_result_from_prefixed_columns,
)
from common.test_inference import (
    YoloInferenceProfileCandidate,
    load_test_inference_recipe,
    profile_tune_candidate_from_conf,
    profile_tune_fixed_mask_threshold,
    rewrite_yolo_profile_in_recipe_text,
)
from common.variants import repo_root

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


def iter_grid_candidates(spec: TuneGridSpec) -> Iterable[YoloInferenceProfileCandidate]:
    for conf in spec.grid.conf:
        yield profile_tune_candidate_from_conf(conf)


PROFILE_SELECTION_OBJECTIVE = "pq"


def mean_pq_across_variants(variant_pq: dict[str, float]) -> float:
    if not variant_pq:
        raise ValueError("variant_pq must not be empty")
    return float(sum(variant_pq.values()) / len(variant_pq))


def select_best_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("rows must not be empty")
    return max(rows, key=lambda row: float(row["mean_pq"]))


def per_variant_pq_results_from_grid_row(
    row: dict[str, Any],
    *,
    variant_names: tuple[str, ...],
) -> dict[str, MergedViewPqResult]:
    return {
        variant: merged_view_pq_result_from_prefixed_columns(row, suffix=variant)
        for variant in variant_names
    }


def grid_results_fieldnames(variant_names: tuple[str, ...]) -> list[str]:
    base = [
        "candidate_id",
        "conf",
        "mask_threshold",
    ]
    mean_columns = [f"mean_{key}" for key in MERGED_VIEW_PQ_RESULT_KEYS]
    per_variant_columns = [
        merged_view_pq_column_name(key, variant)
        for variant in variant_names
        for key in MERGED_VIEW_PQ_RESULT_KEYS
    ]
    return base + mean_columns + per_variant_columns


def grid_result_row_from_candidate_scoring(
    *,
    candidate: YoloInferenceProfileCandidate,
    per_variant_pq_results: dict[str, MergedViewPqResult],
) -> dict[str, Any]:
    mean_pq_fields = mean_merged_view_pq_results(list(per_variant_pq_results.values()))
    row: dict[str, Any] = {
        "candidate_id": candidate.candidate_id(),
        **candidate.to_dict(),
        **{f"mean_{key}": value for key, value in mean_pq_fields.items()},
        **flatten_merged_view_pq_results_by_suffix(per_variant_pq_results),
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
        "profile_selection_axes": ["conf"],
        "fixed_mask_threshold": fixed_mask,
        "removed_grid_axes": list(_REMOVED_GRID_AXES),
        "per_variant_pq_results": per_variant_pq_results,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def candidate_from_winner_json(
    payload: dict[str, Any],
) -> YoloInferenceProfileCandidate:
    return profile_tune_candidate_from_conf(
        yv.require_float(payload.get("conf"), context="conf")
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
                column = merged_view_pq_column_name(key, variant)
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
        rewrite_yolo_profile_in_recipe_text(
            original_text,
            YoloInferenceProfileCandidate(conf=profile.conf, mask_threshold=fixed_mask),
            keys=("conf", "mask_threshold"),
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
