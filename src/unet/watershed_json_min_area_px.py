"""Print min_area_px from a watershed_best_*.json file (for SLURM CC eval)."""

from __future__ import annotations

import sys
from pathlib import Path

from unet.watershed_best_params import load_watershed_best_params


def main() -> None:
    if len(sys.argv) != 2:
        print(
            "usage: watershed_json_min_area_px.py <watershed_best.json>",
            file=sys.stderr,
        )
        sys.exit(2)
    path = Path(sys.argv[1])
    if not path.is_file():
        print(f"not a file: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        params = load_watershed_best_params(path)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    print(params.min_area_px)


if __name__ == "__main__":
    main()
