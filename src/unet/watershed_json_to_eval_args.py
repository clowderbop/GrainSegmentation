from __future__ import annotations

import sys
from pathlib import Path

from unet.watershed_best_params import load_watershed_best_params


def main() -> None:
    if len(sys.argv) != 2:
        print(
            "usage: watershed_json_to_eval_args.py <watershed_best.json>",
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

    min_distance = params.min_distance
    boundary_dilate_iter = params.boundary_dilate_iter
    watershed_connectivity = params.watershed_connectivity
    min_area_px = params.min_area_px
    exclude_border = params.exclude_border
    ridge_level = params.ridge_level

    lines: list[str] = [
        "--instance-method",
        "watershed",
        "--watershed-min-distance",
        str(min_distance),
        "--watershed-h-maxima",
        str(params.h_maxima),
        "--watershed-boundary-dilate-iter",
        str(boundary_dilate_iter),
        "--watershed-connectivity",
        str(watershed_connectivity),
        "--watershed-min-area-px",
        str(min_area_px),
    ]
    if exclude_border:
        lines.append("--watershed-exclude-border")
    else:
        lines.append("--no-watershed-exclude-border")
    if ridge_level is not None:
        lines.extend(["--watershed-ridge-level", str(float(ridge_level))])

    for token in lines:
        print(token)


if __name__ == "__main__":
    main()
