"""Tests for profile selection finalize job (ADR 0005)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from common.test_inference import YoloInferenceProfileCandidate
from yolo.inference_profile_tune import (
    load_grid_results_csv,
    load_grid_winner,
    profile_selection_row_path,
    write_profile_selection_row,
)
from yolo.profile_tune_finalize import collect_profile_selection_rows, main


def _write_mini_grid(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "grid": {
                    "postprocess_type": ["GREEDYNMM", "NMM"],
                    "match_metric": ["IOS"],
                    "match_threshold": [0.5],
                    "conf": [0.25],
                    "mask_threshold": [0.5],
                },
            }
        ),
        encoding="utf-8",
    )


def test_finalize_merges_rows_into_results_csv_and_winner(tmp_path: Path) -> None:
    _write_mini_grid(tmp_path / "grid.yaml")
    output_dir = tmp_path / "run"
    grid_dir = output_dir / "grid"
    better = YoloInferenceProfileCandidate(
        postprocess_type="GREEDYNMM",
        match_metric="IOS",
        match_threshold=0.5,
        conf=0.25,
        mask_threshold=0.5,
    )
    worse = YoloInferenceProfileCandidate(
        postprocess_type="NMM",
        match_metric="IOS",
        match_threshold=0.5,
        conf=0.25,
        mask_threshold=0.5,
    )
    for candidate, aji in ((better, 0.9), (worse, 0.4)):
        write_profile_selection_row(
            profile_selection_row_path(grid_dir, candidate.candidate_id()),
            {
                "candidate_id": candidate.candidate_id(),
                **candidate.to_dict(),
                "mean_aji": aji,
                "aji__PPL": aji,
                "fingerprint": {},
            },
        )

    main(
        [
            "--output-dir",
            str(output_dir),
            "--grid-config",
            str(tmp_path / "grid.yaml"),
            "--variants",
            "PPL",
            "--expected-candidate-count",
            "2",
        ]
    )

    rows = load_grid_results_csv(grid_dir / "results.csv")
    assert len(rows) == 2
    winner = load_grid_winner(grid_dir / "winner.json")
    assert winner == better


def test_collect_profile_selection_rows_sorted_by_filename(tmp_path: Path) -> None:
    grid_dir = tmp_path / "grid"
    for name in ("b_row", "a_row"):
        path = grid_dir / "rows" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"candidate_id": name}), encoding="utf-8")
    rows = collect_profile_selection_rows(grid_dir)
    assert [row["candidate_id"] for row in rows] == ["a_row", "b_row"]


def test_finalize_recompute_winner_from_csv(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    grid_dir = tmp_path / "grid"
    grid_dir.mkdir(parents=True)
    (grid_dir / "results.csv").write_text(
        "candidate_id,postprocess_type,match_metric,match_threshold,conf,mask_threshold,mean_aji,aji__PPL\n"
        "winner,GREEDYNMM,IOS,0.5,0.2,0.45,0.95,0.95\n",
        encoding="utf-8",
    )
    main(
        [
            "--output-dir",
            str(tmp_path),
            "--recompute-winner-from-csv",
            "--variants",
            "PPL",
        ]
    )
    captured = json.loads(capsys.readouterr().out.strip())
    assert captured["conf"] == pytest.approx(0.2)
    assert (grid_dir / "winner.json").is_file()
