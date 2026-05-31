"""CLI: list grid candidate ids (one per line) for SLURM array submission."""

from __future__ import annotations

import argparse
from pathlib import Path

from yolo.inference_profile_tune import iter_grid_candidates, load_tune_grid, tune_grid_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid-config",
        type=Path,
        default=None,
        help=f"Search grid YAML (default: {tune_grid_path()}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    spec = load_tune_grid(args.grid_config)
    for candidate in iter_grid_candidates(spec):
        print(candidate.candidate_id())


if __name__ == "__main__":
    main()
