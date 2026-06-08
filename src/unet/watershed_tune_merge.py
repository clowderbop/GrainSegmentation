"""Merge shard watershed tune grid CSVs into canonical tune artifacts."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from unet.extraction_tune_scoring import (
    WatershedParamSet,
    format_watershed_param_set,
    select_best_watershed_tune_row,
    watershed_best_json_summary,
    watershed_param_set_from_tune_row,
    watershed_tune_fieldnames,
)
from unet.watershed_tune_grid import (
    WatershedTuneGrid,
    iter_watershed_tune_param_sets,
    watershed_tune_candidate_count,
)


@dataclass(frozen=True)
class WatershedTuneMergeResult:
    merged_rows: list[dict[str, Any]]
    fieldnames: list[str]
    best_row: dict[str, Any]
    best_params: WatershedParamSet
    best_json: dict[str, Any]


def load_watershed_tune_grid_csv(path: Path | str) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"watershed tune grid CSV has no data rows: {path}")
    return rows


def merge_watershed_tune_shard_csvs(
    shard_csv_paths: Sequence[Path | str],
    *,
    grid: WatershedTuneGrid | None = None,
    sample_ids: Sequence[str] | None = None,
    sanitize_sample_id: Callable[[str], str],
) -> WatershedTuneMergeResult:
    if not shard_csv_paths:
        raise ValueError("shard_csv_paths must not be empty")
    if sample_ids is None:
        raise ValueError("sample_ids is required to build watershed best JSON")

    fieldnames = watershed_tune_fieldnames(
        sample_ids, sanitize_sample_id=sanitize_sample_id
    )
    merged_rows: list[dict[str, Any]] = []
    seen: dict[WatershedParamSet, Path] = {}
    for shard_csv_path in shard_csv_paths:
        path = Path(shard_csv_path)
        for row in load_watershed_tune_grid_csv(path):
            params = watershed_param_set_from_tune_row(row)
            if params in seen:
                raise ValueError(
                    "duplicate WatershedParamSet across shard CSVs: "
                    f"{format_watershed_param_set(params)} "
                    f"(first in {seen[params]}, also in {path})"
                )
            seen[params] = path
            merged_rows.append(row)

    if grid is not None:
        expected = watershed_tune_candidate_count(grid)
        if len(merged_rows) != expected:
            raise ValueError(
                f"expected {expected} watershed tune rows, found {len(merged_rows)}"
            )
        order = {
            params: index
            for index, params in enumerate(iter_watershed_tune_param_sets(grid))
        }
        for row in merged_rows:
            params = watershed_param_set_from_tune_row(row)
            if params not in order:
                raise ValueError(
                    "watershed tune row is not in configured grid: "
                    f"{format_watershed_param_set(params)}"
                )
        merged_rows.sort(key=lambda row: order[watershed_param_set_from_tune_row(row)])

    best_row = select_best_watershed_tune_row(merged_rows)
    best_params = watershed_param_set_from_tune_row(best_row)
    best_json = watershed_best_json_summary(
        best_row,
        best_params,
        sample_ids,
        sanitize_sample_id=sanitize_sample_id,
    )
    return WatershedTuneMergeResult(
        merged_rows=merged_rows,
        fieldnames=fieldnames,
        best_row=best_row,
        best_params=best_params,
        best_json=best_json,
    )


def write_watershed_tune_merge_artifacts(
    result: WatershedTuneMergeResult,
    *,
    output_csv: Path,
    output_json: Path,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=result.fieldnames)
        writer.writeheader()
        writer.writerows(result.merged_rows)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(result.best_json, handle, indent=2)
