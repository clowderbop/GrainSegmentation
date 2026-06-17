"""CLI: print SLURM wall time for watershed tune jobs from grid YAML."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from unet.slurm_watershed_tune import (
    WatershedTuneWalltimeRole,
    watershed_tune_walltime_for_role,
    watershed_tune_walltimes_for_grid_config,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    role_group = parser.add_mutually_exclusive_group(required=True)
    role_group.add_argument(
        "--role",
        choices=("monolithic", "shard", "merge"),
        help="Tune job shape: monolithic tune, one shard, or merge.",
    )
    role_group.add_argument(
        "--all",
        action="store_true",
        help="Print shard, monolithic, and merge walltimes on one line.",
    )
    parser.add_argument(
        "--grid-config",
        type=Path,
        default=None,
        help="Watershed tune grid YAML (required for monolithic and shard roles).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.all:
        shard, monolithic, merge = watershed_tune_walltimes_for_grid_config(
            args.grid_config
        )
        print(f"{shard} {monolithic} {merge}")
        return
    role: WatershedTuneWalltimeRole = args.role
    print(watershed_tune_walltime_for_role(role, grid_config=args.grid_config))


if __name__ == "__main__":
    main(sys.argv[1:])
