"""Tests for profile selection finalize job (ADR 0005)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from common.merged_view_pq import MERGED_VIEW_PQ_RESULT_KEYS
from common.test_inference import (
    YoloInferenceProfileCandidate,
    load_test_inference_recipe,
    profile_tune_candidate_from_conf,
)
from yolo.inference_profile_tune import (
    grid_result_row_from_candidate_scoring,
    grid_results_fieldnames,
    load_grid_results_csv,
    load_grid_winner,
    profile_selection_row_path,
    write_profile_selection_row,
)
from yolo.profile_tune_finalize import collect_profile_selection_rows, main
from yolo.tests.profile_tune_fixtures import constant_merged_view_pq_result


def _write_mini_grid(path: Path) -> None:
    path.write_text(
        yaml.safe_dump({"grid": {"conf": [0.25, 0.30]}}),
        encoding="utf-8",
    )


def test_finalize_merges_rows_into_results_csv_and_winner(tmp_path: Path) -> None:
    _write_mini_grid(tmp_path / "grid.yaml")
    output_dir = tmp_path / "run"
    grid_dir = output_dir / "grid"
    better = profile_tune_candidate_from_conf(0.25)
    worse = profile_tune_candidate_from_conf(0.30)
    for candidate, pq in ((better, 0.9), (worse, 0.4)):
        write_profile_selection_row(
            profile_selection_row_path(grid_dir, candidate.candidate_id()),
            {
                **grid_result_row_from_candidate_scoring(
                    candidate=candidate,
                    per_variant_pq_results={"PPL": constant_merged_view_pq_result(pq)},
                ),
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
    payload = json.loads((grid_dir / "winner.json").read_text(encoding="utf-8"))
    assert payload["mean_pq"] == pytest.approx(0.9)
    assert payload["selection_objective"] == "pq"
    assert "per_variant_pq" not in payload
    ppl = payload["per_variant_pq_results"]["PPL"]
    assert tuple(ppl.keys()) == MERGED_VIEW_PQ_RESULT_KEYS
    assert ppl["pq"] == pytest.approx(0.9)


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
    fieldnames = grid_results_fieldnames(("PPL",))
    mean_values = ",".join("0.95" for _ in MERGED_VIEW_PQ_RESULT_KEYS)
    per_variant_values = ",".join("0.95" for _ in MERGED_VIEW_PQ_RESULT_KEYS)
    (grid_dir / "results.csv").write_text(
        ",".join(fieldnames) + "\n"
        f"winner,0.2,0.4,{mean_values},{per_variant_values}\n",
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
