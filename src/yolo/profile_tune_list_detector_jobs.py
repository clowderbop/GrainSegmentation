"""CLI: print detector SLURM array variants (one input configuration per line) for submit."""

from __future__ import annotations

import argparse
from pathlib import Path

from yolo.inference_profile_tune import load_tune_grid, tune_grid_path
from yolo.profile_tune_cli import parse_profile_tune_variants


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid-config",
        type=Path,
        default=None,
        help=f"Search grid YAML (default: {tune_grid_path()}).",
    )
    parser.add_argument(
        "--variants",
        default=None,
        help="Comma-separated registry variants (default: all).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    spec = load_tune_grid(args.grid_config)
    variants = parse_profile_tune_variants(args.variants)
    del spec
    for variant in variants:
        print(variant)


if __name__ == "__main__":
    main()
