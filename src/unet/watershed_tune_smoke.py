"""Lightweight watershed tune smoke and timing harness (pre-SLURM sanity check)."""

from __future__ import annotations

import argparse
import time
from typing import Sequence

from unet.extraction_tune_scoring import (
    WatershedParamSet,
    format_merged_view_pq_audit_line,
    format_watershed_param_set,
    mean_train_pq_for_watershed_params,
)
from unet.watershed_tune_fixtures import (
    TRAIN_WHOLE_SECTION_SHAPE,
    large_shape_sparse_two_grain_masks,
)

DEFAULT_SMOKE_SHAPE = (1_000, 5_200)
DEFAULT_SMOKE_PARAMS = WatershedParamSet(5, 0, 1, 0, False, None, h_maxima=0)


def default_smoke_watershed_params() -> WatershedParamSet:
    """Fixed minimal combo for pre-SLURM smoke (independent of tune grid YAML)."""
    return DEFAULT_SMOKE_PARAMS


def run_watershed_tune_smoke(
    params: WatershedParamSet | None = None,
    *,
    height: int = DEFAULT_SMOKE_SHAPE[0],
    width: int = DEFAULT_SMOKE_SHAPE[1],
    sample_id: str = "train",
) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
    """Score one watershed combo on synthetic large-shape masks (no production data)."""
    if params is None:
        params = default_smoke_watershed_params()
    gt, semantic = large_shape_sparse_two_grain_masks(height=height, width=width)
    print(
        f"watershed tune smoke: shape=({height}, {width}) "
        f"{format_watershed_param_set(params)}",
        flush=True,
    )
    t0 = time.perf_counter()
    mean_pq, per_sample = mean_train_pq_for_watershed_params(
        [gt],
        [semantic],
        params,
        sample_ids=[sample_id],
        log=True,
    )
    elapsed = time.perf_counter() - t0
    print(
        f"watershed tune smoke complete: {format_merged_view_pq_audit_line(mean_pq)} "
        f"({format_watershed_param_set(params)}) {elapsed:.1f}s",
        flush=True,
    )
    return mean_pq, per_sample


def _build_arg_parser() -> argparse.ArgumentParser:
    default_params = default_smoke_watershed_params()
    parser = argparse.ArgumentParser(
        description=(
            "Run one watershed tune scoring combo on synthetic large-shape masks. "
            "Use as a pre-SLURM sanity check without cached preds or GPKG access."
        )
    )
    parser.add_argument(
        "--full-shape",
        action="store_true",
        help=(
            f"Use train whole-section geometry {TRAIN_WHOLE_SECTION_SHAPE} "
            "(slow; prefer srun for this size)."
        ),
    )
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument(
        "--h-maxima",
        type=int,
        default=default_params.h_maxima,
    )
    parser.add_argument(
        "--min-distance",
        type=int,
        default=default_params.min_distance,
    )
    parser.add_argument(
        "--boundary-dilate-iter",
        type=int,
        default=default_params.boundary_dilate_iter,
    )
    parser.add_argument(
        "--watershed-connectivity",
        type=int,
        default=default_params.watershed_connectivity,
        choices=[1, 2],
    )
    parser.add_argument(
        "--min-area-px",
        type=int,
        default=default_params.min_area_px,
    )
    parser.add_argument(
        "--exclude-border",
        type=int,
        default=int(default_params.exclude_border),
        choices=[0, 1],
    )
    parser.add_argument("--ridge-level", type=float, default=default_params.ridge_level)
    parser.add_argument("--sample-id", default="train")
    return parser


def _resolve_shape(args: argparse.Namespace) -> tuple[int, int]:
    if args.full_shape:
        return TRAIN_WHOLE_SECTION_SHAPE
    height = args.height if args.height is not None else DEFAULT_SMOKE_SHAPE[0]
    width = args.width if args.width is not None else DEFAULT_SMOKE_SHAPE[1]
    return height, width


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    params = WatershedParamSet(
        min_distance=args.min_distance,
        boundary_dilate_iter=args.boundary_dilate_iter,
        watershed_connectivity=args.watershed_connectivity,
        min_area_px=args.min_area_px,
        exclude_border=bool(args.exclude_border),
        ridge_level=args.ridge_level,
        h_maxima=args.h_maxima,
    )
    height, width = _resolve_shape(args)
    run_watershed_tune_smoke(
        params,
        height=height,
        width=width,
        sample_id=args.sample_id,
    )


if __name__ == "__main__":
    main()
