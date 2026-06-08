"""CLI: print watershed tune shard count for SLURM array submission."""

from __future__ import annotations

import argparse
from pathlib import Path

from unet.watershed_tune_grid import watershed_tune_grid_path
from unet.watershed_tune_grid_shard import watershed_tune_shard_count_for_grid_config


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid-config",
        type=Path,
        default=None,
        help=f"Watershed tune grid YAML (default: {watershed_tune_grid_path()}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    print(watershed_tune_shard_count_for_grid_config(args.grid_config))


if __name__ == "__main__":
    main()
