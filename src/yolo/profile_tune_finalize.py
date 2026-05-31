"""CLI: merge profile selection result rows into grid audit artifacts (ADR 0005)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from yolo.profile_tune_cli import parse_profile_tune_variants
from yolo.inference_profile_tune import (
    finalize_grid_winner,
    load_grid_results_csv,
    load_grid_winner,
    load_profile_selection_row,
    load_tune_grid,
    rows_to_grid_results,
)


def collect_profile_selection_rows(grid_dir: Path) -> list[dict]:
    rows_dir = grid_dir / "rows"
    if not rows_dir.is_dir():
        raise FileNotFoundError(f"Missing profile selection rows directory: {rows_dir}")
    paths = sorted(rows_dir.glob("*.json"))
    if not paths:
        raise ValueError(f"No profile selection rows under {rows_dir}")
    return [load_profile_selection_row(path) for path in paths]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid-config", type=Path, default=None)
    parser.add_argument("--variants", default=None)
    parser.add_argument(
        "--expected-candidate-count",
        type=int,
        default=None,
        help="Fail when row count differs (default: full grid size from config).",
    )
    parser.add_argument(
        "--recompute-winner-from-csv",
        action="store_true",
        help="Write grid/winner.json from existing grid/results.csv and exit.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    variants = parse_profile_tune_variants(args.variants)
    grid_dir = args.output_dir / "grid"

    if args.recompute_winner_from_csv:
        from yolo.inference_profile_tune import recompute_winner_from_csv

        winner = recompute_winner_from_csv(args.output_dir, variant_names=variants)
        print(json.dumps(winner.to_dict(), indent=2))
        return

    spec = load_tune_grid(args.grid_config)
    expected = args.expected_candidate_count
    if expected is None:
        from yolo.inference_profile_tune import iter_grid_candidates

        expected = len(list(iter_grid_candidates(spec)))

    row_payloads = collect_profile_selection_rows(grid_dir)
    if len(row_payloads) != expected:
        raise ValueError(
            f"Expected {expected} profile selection rows, found {len(row_payloads)}"
        )

    csv_rows = rows_to_grid_results(row_payloads, variant_names=variants)
    winner = finalize_grid_winner(grid_dir, csv_rows, variant_names=variants)
    print(json.dumps(winner.to_dict(), indent=2))
    loaded = load_grid_winner(grid_dir / "winner.json")
    assert loaded == winner
    assert load_grid_results_csv(grid_dir / "results.csv")


if __name__ == "__main__":
    main()
