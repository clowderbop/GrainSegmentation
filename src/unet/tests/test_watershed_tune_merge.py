"""Shard grid CSV merge contracts (issue 03)."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

from unet import tune_watershed
from unet.extraction_tune_scoring import (
    select_best_watershed_tune_row,
    watershed_tune_fieldnames,
)
from unet.tests.tune_watershed_cli_fixtures import (
    make_tune_collect_args,
    tune_watershed_argv,
    watershed_param_set_from_csv_row,
    write_mini_tune_grid,
)
from unet.watershed_tune_grid import load_watershed_tune_grid
from unet.watershed_tune_grid_shard import (
    iter_watershed_tune_param_sets_for_shard,
    iter_watershed_tune_shards,
)
from common.merged_view_pq import MERGED_VIEW_PQ_RESULT_KEYS
from unet.watershed_tune_merge import (
    merge_watershed_tune_shard_csvs,
    write_watershed_tune_merge_artifacts,
)
from unet.watershed_tune_merge_cli import main as merge_main


def _write_watershed_tune_csv(
    path: Path,
    rows: list[dict[str, str]],
    *,
    sample_ids: list[str],
) -> None:
    fieldnames = watershed_tune_fieldnames(
        sample_ids, sanitize_sample_id=lambda sample_id: sample_id
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _split_monolithic_rows_into_shard_csvs(
    rows: list[dict[str, str]],
    *,
    grid_path: Path,
    output_dir: Path,
) -> list[Path]:
    grid = load_watershed_tune_grid(grid_path).grid
    sample_ids = ["train"]
    shard_paths: list[Path] = []
    for shard in iter_watershed_tune_shards(grid):
        shard_params = set(iter_watershed_tune_param_sets_for_shard(grid, shard))
        shard_rows = [
            row
            for row in rows
            if watershed_param_set_from_csv_row(row) in shard_params
        ]
        shard_path = output_dir / f"watershed_grid_shard_{shard.index}.csv"
        _write_watershed_tune_csv(shard_path, shard_rows, sample_ids=sample_ids)
        shard_paths.append(shard_path)
    return shard_paths


def test_merge_shard_csvs_matches_monolithic_best_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """INTENT: merging shard CSVs picks the same best WatershedParamSet and mean_pq as monolithic tune."""
    grid_path = tmp_path / "mini_grid.yaml"
    write_mini_tune_grid(grid_path)
    args = make_tune_collect_args(tmp_path, paint_semantic_region=True)
    monolithic_csv = tmp_path / "monolithic.csv"

    monkeypatch.setattr(
        sys,
        "argv",
        tune_watershed_argv(
            args,
            output_csv=monolithic_csv,
            grid_config=grid_path,
        ),
    )
    tune_watershed.main()

    with monolithic_csv.open(encoding="utf-8") as handle:
        monolithic_rows = list(csv.DictReader(handle))

    monolithic_best = select_best_watershed_tune_row(monolithic_rows)
    shard_csvs = _split_monolithic_rows_into_shard_csvs(
        monolithic_rows,
        grid_path=grid_path,
        output_dir=tmp_path / "shards",
    )
    grid = load_watershed_tune_grid(grid_path).grid

    result = merge_watershed_tune_shard_csvs(
        shard_csvs,
        grid=grid,
        sample_ids=["train"],
        sanitize_sample_id=lambda sample_id: sample_id,
    )

    assert watershed_param_set_from_csv_row(result.best_row) == watershed_param_set_from_csv_row(
        monolithic_best
    )
    assert float(result.best_row["mean_pq"]) == pytest.approx(
        float(monolithic_best["mean_pq"])
    )


def test_merge_rejects_duplicate_watershed_param_set_across_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """INTENT: merge fails with a clear error when shard CSVs contain the same WatershedParamSet."""
    grid_path = tmp_path / "mini_grid.yaml"
    write_mini_tune_grid(grid_path)
    args = make_tune_collect_args(tmp_path, paint_semantic_region=True)
    monolithic_csv = tmp_path / "monolithic.csv"

    monkeypatch.setattr(
        sys,
        "argv",
        tune_watershed_argv(
            args,
            output_csv=monolithic_csv,
            grid_config=grid_path,
        ),
    )
    tune_watershed.main()

    with monolithic_csv.open(encoding="utf-8") as handle:
        monolithic_rows = list(csv.DictReader(handle))

    shard_dir = tmp_path / "shards"
    shard_csvs = _split_monolithic_rows_into_shard_csvs(
        monolithic_rows,
        grid_path=grid_path,
        output_dir=shard_dir,
    )
    duplicate_path = shard_dir / "duplicate.csv"
    _write_watershed_tune_csv(
        duplicate_path,
        [monolithic_rows[0]],
        sample_ids=["train"],
    )
    grid = load_watershed_tune_grid(grid_path).grid

    with pytest.raises(ValueError, match="duplicate WatershedParamSet"):
        merge_watershed_tune_shard_csvs(
            [*shard_csvs, duplicate_path],
            grid=grid,
            sample_ids=["train"],
            sanitize_sample_id=lambda sample_id: sample_id,
        )


def test_merge_rejects_incomplete_shard_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """INTENT: merge fails when shard CSVs cover fewer combos than the configured grid."""
    grid_path = tmp_path / "mini_grid.yaml"
    write_mini_tune_grid(grid_path)
    args = make_tune_collect_args(tmp_path, paint_semantic_region=True)
    monolithic_csv = tmp_path / "monolithic.csv"

    monkeypatch.setattr(
        sys,
        "argv",
        tune_watershed_argv(
            args,
            output_csv=monolithic_csv,
            grid_config=grid_path,
        ),
    )
    tune_watershed.main()

    with monolithic_csv.open(encoding="utf-8") as handle:
        monolithic_rows = list(csv.DictReader(handle))

    shard_dir = tmp_path / "shards"
    shard_csvs = _split_monolithic_rows_into_shard_csvs(
        monolithic_rows,
        grid_path=grid_path,
        output_dir=shard_dir,
    )
    grid = load_watershed_tune_grid(grid_path).grid

    with pytest.raises(ValueError, match="expected 4 watershed tune rows, found 2"):
        merge_watershed_tune_shard_csvs(
            shard_csvs[:1],
            grid=grid,
            sample_ids=["train"],
            sanitize_sample_id=lambda sample_id: sample_id,
        )


def test_merge_writes_full_grid_csv_and_merged_view_pq_best_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """INTENT: merge writes every combo row to the merged CSV and ADR 0003 best JSON fields."""
    grid_path = tmp_path / "mini_grid.yaml"
    write_mini_tune_grid(grid_path)
    args = make_tune_collect_args(tmp_path, paint_semantic_region=True)
    monolithic_csv = tmp_path / "monolithic.csv"

    monkeypatch.setattr(
        sys,
        "argv",
        tune_watershed_argv(
            args,
            output_csv=monolithic_csv,
            grid_config=grid_path,
        ),
    )
    tune_watershed.main()

    with monolithic_csv.open(encoding="utf-8") as handle:
        monolithic_rows = list(csv.DictReader(handle))

    shard_csvs = _split_monolithic_rows_into_shard_csvs(
        monolithic_rows,
        grid_path=grid_path,
        output_dir=tmp_path / "shards",
    )
    merged_csv = tmp_path / "watershed_grid_merged.csv"
    best_json = tmp_path / "watershed_best_12345.json"
    grid = load_watershed_tune_grid(grid_path).grid

    result = merge_watershed_tune_shard_csvs(
        shard_csvs,
        grid=grid,
        sample_ids=["train"],
        sanitize_sample_id=lambda sample_id: sample_id,
    )
    write_watershed_tune_merge_artifacts(
        result,
        output_csv=merged_csv,
        output_json=best_json,
    )

    with merged_csv.open(encoding="utf-8") as handle:
        merged_rows = list(csv.DictReader(handle))

    assert len(merged_rows) == len(monolithic_rows)
    fieldnames = watershed_tune_fieldnames(
        ["train"], sanitize_sample_id=lambda sample_id: sample_id
    )
    assert set(merged_rows[0].keys()) == set(fieldnames)
    for key in MERGED_VIEW_PQ_RESULT_KEYS:
        assert f"mean_{key}" in merged_rows[0]

    payload = json.loads(best_json.read_text(encoding="utf-8"))
    assert payload["selection_objective"] == "pq"
    for key in MERGED_VIEW_PQ_RESULT_KEYS:
        assert f"best_mean_{key}" in payload
        assert f"best_per_sample_{key}" in payload
        assert "train" in payload[f"best_per_sample_{key}"]


def test_merge_cli_writes_merged_csv_and_best_json_from_shard_glob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """INTENT: merge CLI resolves shard CSV glob, validates grid, and writes canonical artifacts."""
    grid_path = tmp_path / "mini_grid.yaml"
    write_mini_tune_grid(grid_path)
    args = make_tune_collect_args(tmp_path, paint_semantic_region=True)
    monolithic_csv = tmp_path / "monolithic.csv"

    monkeypatch.setattr(
        sys,
        "argv",
        tune_watershed_argv(
            args,
            output_csv=monolithic_csv,
            grid_config=grid_path,
        ),
    )
    tune_watershed.main()

    with monolithic_csv.open(encoding="utf-8") as handle:
        monolithic_rows = list(csv.DictReader(handle))

    shard_dir = tmp_path / "shards"
    _split_monolithic_rows_into_shard_csvs(
        monolithic_rows,
        grid_path=grid_path,
        output_dir=shard_dir,
    )
    merged_csv = tmp_path / "watershed_grid_run.csv"
    best_json = tmp_path / "watershed_best_99999.json"

    merge_main(
        [
            "--shard-csv-glob",
            str(shard_dir / "watershed_grid_shard_*.csv"),
            "--grid-config",
            str(grid_path),
            "--manifest",
            str(args.manifest),
            "--output-csv",
            str(merged_csv),
            "--output-json",
            str(best_json),
        ]
    )

    assert merged_csv.is_file()
    assert best_json.is_file()
    with merged_csv.open(encoding="utf-8") as handle:
        merged_rows = list(csv.DictReader(handle))
    assert len(merged_rows) == len(monolithic_rows)
    payload = json.loads(best_json.read_text(encoding="utf-8"))
    assert payload["selection_objective"] == "pq"


def test_merge_cli_writes_merged_csv_from_explicit_shard_csv_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """INTENT: merge CLI accepts repeatable --shard-csv paths without a glob."""
    grid_path = tmp_path / "mini_grid.yaml"
    write_mini_tune_grid(grid_path)
    args = make_tune_collect_args(tmp_path, paint_semantic_region=True)
    monolithic_csv = tmp_path / "monolithic.csv"

    monkeypatch.setattr(
        sys,
        "argv",
        tune_watershed_argv(
            args,
            output_csv=monolithic_csv,
            grid_config=grid_path,
        ),
    )
    tune_watershed.main()

    with monolithic_csv.open(encoding="utf-8") as handle:
        monolithic_rows = list(csv.DictReader(handle))

    shard_csvs = _split_monolithic_rows_into_shard_csvs(
        monolithic_rows,
        grid_path=grid_path,
        output_dir=tmp_path / "shards",
    )
    merged_csv = tmp_path / "watershed_grid_run.csv"
    best_json = tmp_path / "watershed_best_99999.json"

    merge_main(
        [
            *(arg for path in shard_csvs for arg in ("--shard-csv", str(path))),
            "--grid-config",
            str(grid_path),
            "--sample-ids",
            "train",
            "--output-csv",
            str(merged_csv),
            "--output-json",
            str(best_json),
        ]
    )

    with merged_csv.open(encoding="utf-8") as handle:
        merged_rows = list(csv.DictReader(handle))
    assert len(merged_rows) == len(monolithic_rows)


def test_merge_cli_rejects_missing_shard_csv_sources(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """INTENT: merge CLI reports a clear argument error when no shard CSV inputs are supplied."""
    with pytest.raises(SystemExit):
        merge_main(
            [
                "--output-csv",
                str(tmp_path / "out.csv"),
                "--output-json",
                str(tmp_path / "out.json"),
                "--sample-ids",
                "train",
            ]
        )
    assert "at least one shard CSV is required" in capsys.readouterr().err
