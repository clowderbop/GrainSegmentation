#!/usr/bin/env python3
"""Regenerate golden instance_map.npz after intentional painter or fixture changes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common.gpkg_instance_map import gpkg_to_merged_instance_map

_FIXTURE_DIR = Path(__file__).resolve().parent
_DEFAULT_GPKG = _FIXTURE_DIR / "micro_labels.gpkg"
_DEFAULT_NPZ = _FIXTURE_DIR / "instance_map.npz"
_DEFAULT_HEIGHT = 48
_DEFAULT_WIDTH = 64


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpkg", type=Path, default=_DEFAULT_GPKG)
    parser.add_argument("--output", type=Path, default=_DEFAULT_NPZ)
    parser.add_argument("--height", type=int, default=_DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=_DEFAULT_WIDTH)
    args = parser.parse_args(argv)

    instance_map = gpkg_to_merged_instance_map(
        args.gpkg, height=args.height, width=args.width
    )
    np.savez_compressed(args.output, instance_map=instance_map)
    unique_ids = sorted(int(v) for v in np.unique(instance_map) if v != 0)
    print(
        f"wrote {args.output} shape={instance_map.shape} "
        f"instances={len(unique_ids)} ids={unique_ids}"
    )


if __name__ == "__main__":
    main()
