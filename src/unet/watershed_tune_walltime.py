"""CLI: print SLURM wall time for watershed tune jobs from grid YAML."""

from __future__ import annotations

import argparse
from pathlib import Path

from unet.slurm_watershed_tune import (
    WatershedTuneWalltimeRole,
    watershed_tune_walltime_for_role,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--role",
        choices=("monolithic", "shard", "merge"),
        required=True,
        help="Tune job shape: monolithic tune, one shard, or merge.",
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
    role: WatershedTuneWalltimeRole = args.role
    print(watershed_tune_walltime_for_role(role, grid_config=args.grid_config))


if __name__ == "__main__":
    main()
