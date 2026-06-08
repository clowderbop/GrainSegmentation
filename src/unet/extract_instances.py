"""Extract instance labels from U-Net semantic predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from common.arg_errors import raise_cli_argument_error
from common.manifest_io import load_dataset_manifest
from common.prediction_set import (
    build_unet_prediction_set_from_instance_map,
    prediction_set_path,
    save_prediction_set,
)
from common.run_provenance import write_run_provenance
from common.semantic_instance import semantic_to_instance_label_map
from unet.instance_masks import (
    build_watershed_semantic_prep,
    watershed_area_filter,
    watershed_base_extraction,
)
from unet.prediction_cache import prediction_tiff_path


def list_semantic_predictions(semantic_dir: Path) -> list[str]:
    sample_ids: list[str] = []
    for path in sorted(semantic_dir.glob("*_pred.tif")):
        stem = path.name[: -len("_pred.tif")]
        if stem:
            sample_ids.append(stem)
    return sample_ids


def _resolve_sample_ids(args: argparse.Namespace) -> list[str]:
    doc = load_dataset_manifest(args.manifest)
    manifest_ids = [row.sample_id for row in doc.samples]
    available = set(list_semantic_predictions(args.semantic_dir))
    missing = [sid for sid in manifest_ids if sid not in available]
    if missing:
        raise ValueError(
            "Manifest sample_id(s) missing semantic predictions "
            f"under {args.semantic_dir}: {', '.join(missing)}"
        )
    extra = sorted(available - set(manifest_ids))
    if extra:
        print(
            f"Note: ignoring {len(extra)} semantic prediction(s) not listed in manifest.",
            file=sys.stderr,
        )
    return manifest_ids


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract instance prediction sets from semantic prediction TIFFs.",
    )
    parser.add_argument("--semantic-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--instance-method",
        choices=("cc", "watershed"),
        default="cc",
    )
    parser.add_argument("--watershed-min-distance", type=int, default=1)
    parser.add_argument("--watershed-boundary-dilate-iter", type=int, default=0)
    parser.add_argument("--watershed-connectivity", type=int, choices=(1, 2), default=1)
    parser.add_argument("--watershed-min-area-px", type=int, default=0)
    parser.add_argument(
        "--watershed-exclude-border",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--watershed-ridge-level", type=float, default=None)
    parser.add_argument("--min-area-px", type=int, default=0)
    return parser


def _validate_args(
    args: argparse.Namespace, parser: argparse.ArgumentParser | None = None
) -> None:
    if args.watershed_min_distance < 1:
        raise_cli_argument_error("watershed_min_distance must be >= 1", parser=parser)
    if args.watershed_boundary_dilate_iter < 0:
        raise_cli_argument_error(
            "watershed_boundary_dilate_iter must be >= 0", parser=parser
        )
    if args.watershed_min_area_px < 0:
        raise_cli_argument_error("watershed_min_area_px must be >= 0", parser=parser)
    if args.min_area_px < 0:
        raise_cli_argument_error("min_area_px must be >= 0", parser=parser)
    if args.watershed_ridge_level is not None and not np.isfinite(
        args.watershed_ridge_level
    ):
        raise_cli_argument_error(
            "watershed_ridge_level must be finite when set", parser=parser
        )


def _export_min_area_px(args: argparse.Namespace) -> int:
    if args.instance_method == "watershed":
        return args.watershed_min_area_px
    return args.min_area_px


def _instances_from_semantic(
    semantic: np.ndarray, args: argparse.Namespace
) -> np.ndarray:
    if args.instance_method == "cc":
        return semantic_to_instance_label_map(semantic, min_area_px=args.min_area_px)
    prep = build_watershed_semantic_prep(semantic)
    base = watershed_base_extraction(
        prep,
        min_distance=args.watershed_min_distance,
        boundary_dilate_iter=args.watershed_boundary_dilate_iter,
        watershed_connectivity=args.watershed_connectivity,
        exclude_border=args.watershed_exclude_border,
        ridge_level=args.watershed_ridge_level,
    )
    return watershed_area_filter(base, args.watershed_min_area_px)


def _load_semantic_tiff(path: Path) -> np.ndarray:
    arr = tifffile.imread(path)
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[2] == 1:
            arr = arr[:, :, 0]
        else:
            raise ValueError(f"Semantic prediction must be single-channel: {path}")
    if arr.ndim != 2:
        raise ValueError(f"Semantic prediction must be 2D: {path}")
    return arr.astype(np.int32)


def _run_provenance_payload(args: argparse.Namespace) -> dict[str, Any]:
    export_min_area_px = _export_min_area_px(args)
    payload: dict[str, Any] = {
        "producer": "unet",
        "instance_method": args.instance_method,
        "min_area_px": export_min_area_px,
        "manifest": str(args.manifest.resolve()),
    }
    if args.instance_method == "watershed":
        payload.update(
            {
                "watershed_min_distance": args.watershed_min_distance,
                "watershed_boundary_dilate_iter": args.watershed_boundary_dilate_iter,
                "watershed_connectivity": args.watershed_connectivity,
                "watershed_min_area_px": args.watershed_min_area_px,
                "watershed_exclude_border": args.watershed_exclude_border,
                "watershed_ridge_level": args.watershed_ridge_level,
            }
        )
    return payload


def run_extract_instances(args: argparse.Namespace) -> None:
    _validate_args(args)

    sample_ids = _resolve_sample_ids(args)
    if not sample_ids:
        print(f"ERROR: No *_pred.tif files in {args.semantic_dir}", file=sys.stderr)
        sys.exit(1)

    export_min_area_px = _export_min_area_px(args)
    provenance_path = write_run_provenance(
        args.output_dir, _run_provenance_payload(args)
    )
    print(f"Wrote run provenance to {provenance_path}")

    n_samples = len(sample_ids)
    print(
        f"Instance extraction: {n_samples} sample(s), "
        f"method={args.instance_method}, min_area_px={export_min_area_px}"
    )
    for idx, sample_id in enumerate(sample_ids):
        print(f"Extracting {sample_id} ({idx + 1}/{n_samples})...")
        pred_path = prediction_tiff_path(args.semantic_dir, sample_id)
        semantic = _load_semantic_tiff(pred_path)
        instance_map = _instances_from_semantic(semantic, args)
        prediction_set = build_unet_prediction_set_from_instance_map(instance_map)
        out_path = prediction_set_path(args.output_dir, sample_id)
        save_prediction_set(out_path, prediction_set)
        n_inst = len(prediction_set.detections)
        print(f"Wrote {out_path} ({n_inst} extracted grains)")


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    run_extract_instances(args)


if __name__ == "__main__":
    main()
