"""CLI: print detector SLURM job parameters (variant, conf, mask_threshold) for submit."""

from __future__ import annotations

import argparse
from pathlib import Path

from common.variants import all_variant_names
from yolo.inference_profile_tune import iter_detector_jobs, load_tune_grid, tune_grid_path


def _parse_variants(raw: str | None) -> tuple[str, ...]:
    if raw is None:
        return all_variant_names()
    names = tuple(v.strip() for v in raw.split(",") if v.strip())
    if not names:
        raise ValueError("--variants must list at least one registry variant")
    return names


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
    variants = _parse_variants(args.variants)
    for variant, conf, mask_threshold in iter_detector_jobs(spec, variants):
        print(f"{variant}\t{conf}\t{mask_threshold}")


if __name__ == "__main__":
    main()
