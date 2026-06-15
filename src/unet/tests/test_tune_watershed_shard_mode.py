"""Shard-mode watershed tune CLI contracts (issue 02)."""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import pytest

from unet import tune_watershed
from unet.tune_watershed import _parse_args
from common.merged_view_pq import MERGED_VIEW_PQ_RESULT_KEYS
from unet.extraction_tune_scoring import watershed_tune_fieldnames
from unet.watershed_tune_grid import (
    iter_watershed_tune_param_sets,
    load_watershed_tune_grid,
)
from unet.watershed_tune_grid_shard import (
    iter_watershed_tune_param_sets_for_shard,
    iter_watershed_tune_shards,
    watershed_tune_shard_combo_count,
)
from unet.tests.tune_watershed_cli_fixtures import (
    MICRO_GPKG,
    make_tune_collect_args,
    speckle_prone_tune_collect_args,
    tune_watershed_argv,
    watershed_param_sets_from_csv_rows,
    write_mini_tune_grid,
)

_COMBO_MEAN_TIMING_RE = re.compile(
    r"shard \d+/\d+, combo \d+/\d+ mean PQ=.*\) \d+\.\d+s"
)


def test_shard_mode_scores_only_shard_subset_on_fixtures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """INTENT: shard-mode tune scores only that shard's combos, not the full grid."""
    grid_path = tmp_path / "mini_grid.yaml"
    write_mini_tune_grid(grid_path)
    args = make_tune_collect_args(tmp_path, paint_semantic_region=True)
    out_csv = tmp_path / "shard_grid.csv"

    grid = load_watershed_tune_grid(grid_path).grid
    shard = next(iter(iter_watershed_tune_shards(grid)))
    expected_params = list(iter_watershed_tune_param_sets_for_shard(grid, shard))

    monkeypatch.setattr(
        sys,
        "argv",
        tune_watershed_argv(
            args,
            output_csv=out_csv,
            grid_config=grid_path,
            shard=shard,
        ),
    )
    tune_watershed.main()

    with out_csv.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == len(expected_params)
    assert watershed_param_sets_from_csv_rows(rows) == expected_params


def test_shard_mode_writes_shard_csv_and_skips_best_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """INTENT: shard-mode tune writes the shard grid CSV and never emits best JSON."""
    grid_path = tmp_path / "mini_grid.yaml"
    write_mini_tune_grid(grid_path)
    args = make_tune_collect_args(tmp_path, paint_semantic_region=True)
    out_csv = tmp_path / "watershed_grid_run_shard_1.csv"
    out_json = tmp_path / "watershed_best_12345.json"

    grid = load_watershed_tune_grid(grid_path).grid
    shard = next(iter(iter_watershed_tune_shards(grid)))

    monkeypatch.setattr(
        sys,
        "argv",
        tune_watershed_argv(
            args,
            output_csv=out_csv,
            grid_config=grid_path,
            output_json=out_json,
            shard=shard,
        ),
    )
    tune_watershed.main()

    assert out_csv.is_file()
    assert not out_json.exists()


def test_shard_mode_logs_per_combo_timing_lines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """INTENT: shard-mode logs retain per-combo timing and merged-view PQ audit lines."""
    grid_path = tmp_path / "mini_grid.yaml"
    write_mini_tune_grid(grid_path)
    args = make_tune_collect_args(tmp_path, paint_semantic_region=True)
    out_csv = tmp_path / "shard_grid.csv"

    grid = load_watershed_tune_grid(grid_path).grid
    shard = next(iter(iter_watershed_tune_shards(grid)))
    shard_size = watershed_tune_shard_combo_count(grid, shard)

    monkeypatch.setattr(
        sys,
        "argv",
        tune_watershed_argv(
            args,
            output_csv=out_csv,
            grid_config=grid_path,
            shard=shard,
        ),
    )
    tune_watershed.main()

    out = capsys.readouterr().out
    assert f"shard {shard.index}/" in out
    assert f"combo 1/{shard_size}" in out
    assert f"combo {shard_size}/{shard_size}" in out
    assert "running watershed" in out
    assert "running metrics" in out
    assert "PQ=" in out
    assert _COMBO_MEAN_TIMING_RE.search(out)
    assert " | best:" not in out


def test_shard_mode_csv_rows_retain_merged_view_pq_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """INTENT: shard-mode grid CSV rows keep stable MergedViewPqResult mean and per-sample fields."""
    grid_path = tmp_path / "mini_grid.yaml"
    write_mini_tune_grid(grid_path)
    args = make_tune_collect_args(tmp_path, paint_semantic_region=True)
    out_csv = tmp_path / "shard_grid.csv"

    grid = load_watershed_tune_grid(grid_path).grid
    shard = next(iter(iter_watershed_tune_shards(grid)))

    monkeypatch.setattr(
        sys,
        "argv",
        tune_watershed_argv(
            args,
            output_csv=out_csv,
            grid_config=grid_path,
            shard=shard,
        ),
    )
    tune_watershed.main()

    sample_id = "train"
    fieldnames = watershed_tune_fieldnames(
        [sample_id], sanitize_sample_id=lambda sid: sid
    )
    with out_csv.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows
    for row in rows:
        assert set(row.keys()) == set(fieldnames)
        for key in MERGED_VIEW_PQ_RESULT_KEYS:
            assert f"mean_{key}" in row
            assert f"{key}__{sample_id}" in row
            assert int(row["mean_gt_instance_count"]) >= 0
            assert int(row["mean_pred_instance_count"]) >= 0


def test_shard_mode_surfaces_catastrophic_over_segmentation_in_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """INTENT: shard-mode CSV rows expose very low PQ and high pred counts when extraction fails."""
    grid_path = tmp_path / "speckle_grid.yaml"
    write_mini_tune_grid(
        grid_path,
        grid={
            "min_distance": [5],
            "h_maxima": [0],
            "boundary_dilate_iter": [0],
            "watershed_connectivity": [1],
            "min_area_px": [0],
            "exclude_border": [0],
            "ridge_level": [None],
        },
    )
    args = speckle_prone_tune_collect_args(tmp_path)
    out_csv = tmp_path / "shard_grid.csv"

    grid = load_watershed_tune_grid(grid_path).grid
    shard = next(iter(iter_watershed_tune_shards(grid)))

    monkeypatch.setattr(
        sys,
        "argv",
        tune_watershed_argv(
            args,
            output_csv=out_csv,
            grid_config=grid_path,
            shard=shard,
        ),
    )
    tune_watershed.main()

    with out_csv.open(encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))

    assert int(row["mean_pred_instance_count"]) > int(row["mean_gt_instance_count"])
    assert float(row["mean_pq"]) < 0.15


def test_shard_mode_combo_order_matches_monolithic_subset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """INTENT: shard-mode iteration order matches monolithic ordering for that shard subset."""
    grid_path = tmp_path / "mini_grid.yaml"
    write_mini_tune_grid(grid_path)
    args = make_tune_collect_args(tmp_path, paint_semantic_region=True)
    out_csv = tmp_path / "shard_grid.csv"

    grid = load_watershed_tune_grid(grid_path).grid
    shard = next(s for s in iter_watershed_tune_shards(grid) if s.index == 2)
    monolithic = list(iter_watershed_tune_param_sets(grid))
    expected = [
        params
        for params in monolithic
        if params.min_distance == shard.min_distance
        and params.boundary_dilate_iter == shard.boundary_dilate_iter
    ]

    monkeypatch.setattr(
        sys,
        "argv",
        tune_watershed_argv(
            args,
            output_csv=out_csv,
            grid_config=grid_path,
            shard=shard,
        ),
    )
    tune_watershed.main()

    with out_csv.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert watershed_param_sets_from_csv_rows(rows) == expected


def test_monolithic_mode_writes_best_json_and_grid_progress_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """INTENT: monolithic tune without shard args still writes best JSON and [N/grid] progress."""
    grid_path = tmp_path / "mini_grid.yaml"
    write_mini_tune_grid(grid_path)
    args = make_tune_collect_args(tmp_path, paint_semantic_region=True)
    out_csv = tmp_path / "grid.csv"
    out_json = tmp_path / "best.json"

    grid = load_watershed_tune_grid(grid_path).grid
    grid_size = len(list(iter_watershed_tune_param_sets(grid)))

    monkeypatch.setattr(
        sys,
        "argv",
        tune_watershed_argv(
            args,
            output_csv=out_csv,
            grid_config=grid_path,
            output_json=out_json,
        ),
    )
    tune_watershed.main()

    assert out_csv.is_file()
    assert out_json.is_file()
    out = capsys.readouterr().out
    assert f"[1/{grid_size}]" in out
    assert f"[{grid_size}/{grid_size}]" in out
    assert " | best:" in out
    assert "Best watershed parameters" in out


@pytest.mark.parametrize(
    ("extra_argv", "expected_message"),
    [
        (
            ["--shard-index", "1"],
            "shard mode requires --shard-index, --shard-min-distance, and "
            "--shard-boundary-dilate-iter together",
        ),
        (
            [
                "--shard-index",
                "0",
                "--shard-min-distance",
                "5",
                "--shard-boundary-dilate-iter",
                "0",
            ],
            "shard-index must be >= 1",
        ),
    ],
)
def test_tune_watershed_rejects_invalid_shard_cli_args(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    extra_argv: list[str],
    expected_message: str,
) -> None:
    """INTENT: tune_watershed rejects partial or invalid shard CLI arguments."""
    with pytest.raises(SystemExit):
        _parse_args(
            [
                "--preds-dir",
                str(tmp_path),
                "--manifest",
                "m.json",
                "--gt-gpkg",
                str(MICRO_GPKG),
                "--output-csv",
                "out.csv",
                *extra_argv,
            ]
        )
    assert expected_message in capsys.readouterr().err


def test_tune_watershed_rejects_shard_descriptor_not_in_grid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """INTENT: tune_watershed rejects shard descriptors that do not match the loaded grid."""
    grid_path = tmp_path / "mini_grid.yaml"
    write_mini_tune_grid(grid_path)
    args = make_tune_collect_args(tmp_path, paint_semantic_region=True)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tune_watershed",
            "--preds-dir",
            str(args.preds_dir),
            "--manifest",
            str(args.manifest),
            "--gt-gpkg",
            str(args.gt_gpkg),
            "--output-csv",
            str(tmp_path / "out.csv"),
            "--grid-config",
            str(grid_path),
            "--shard-index",
            "99",
            "--shard-min-distance",
            "5",
            "--shard-boundary-dilate-iter",
            "0",
        ],
    )

    with pytest.raises(ValueError, match="shard descriptor does not match any shard"):
        tune_watershed.main()
