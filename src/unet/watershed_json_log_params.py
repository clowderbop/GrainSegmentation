"""Print tuned watershed params from watershed_best_*.json (for SLURM eval logs)."""

from __future__ import annotations

import sys
from pathlib import Path

from unet.extraction_tune_scoring import format_watershed_param_set
from unet.watershed_best_params import load_watershed_best_params


def main() -> None:
    if len(sys.argv) != 2:
        print(
            "usage: watershed_json_log_params.py <watershed_best.json>",
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
    print(f"  {format_watershed_param_set(params)}")


if __name__ == "__main__":
    main()
