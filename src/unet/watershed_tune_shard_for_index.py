"""CLI: resolve one watershed tune shard descriptor from a 1-based array index."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from unet.watershed_tune_grid import load_watershed_tune_grid, watershed_tune_grid_path
from unet.watershed_tune_grid_shard import iter_watershed_tune_shards


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid-config",
        type=Path,
        default=None,
        help=f"Watershed tune grid YAML (default: {watershed_tune_grid_path()}).",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        required=True,
        help="1-based shard index (matches SLURM_ARRAY_TASK_ID)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    grid = load_watershed_tune_grid(watershed_tune_grid_path(args.grid_config)).grid
    for shard in iter_watershed_tune_shards(grid):
        if shard.index == args.shard_index:
            print(shard.min_distance, shard.boundary_dilate_iter)
            return
    print(
        f"no shard with index {args.shard_index} for grid",
        file=sys.stderr,
    )
    raise SystemExit(1)


if __name__ == "__main__":
    main()
