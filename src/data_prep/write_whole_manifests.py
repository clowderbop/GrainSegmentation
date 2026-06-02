#!/usr/bin/env python3
"""Write U-Net whole-section manifests under dataset/{train,test}/manifests/.

Run once against an existing scratch tree after preprocessing (see
``docs/runbooks/preprocessing.md``), especially after ``create_multichannel_input_tiffs.sh``
so stacked YOLO TIFFs exist on disk but are **not** referenced by these manifests.

Example (from repo root)::

    uv run --directory src/data_prep python write_whole_manifests.py \\
        --grainseg-root "$SCRATCH/GrainSeg"

Only per-channel ``train_*.tif`` / ``test_*.tif`` paths from ``config/variants.yaml`` are
included. Stacked mosaics (e.g. ``train_PPL+AllPPX.tif``) are excluded by design.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common.manifest_io import (
    build_unet_whole_manifest,
    default_whole_manifest_path,
    validate_dataset_manifest,
    write_dataset_manifest,
)
from common.variants import all_variant_names, default_grainseg_root


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grainseg-root",
        type=Path,
        default=None,
        help="GrainSeg scratch root (default: $SCRATCH/GrainSeg)",
    )
    parser.add_argument(
        "--split",
        choices=("train", "test", "both"),
        default="both",
        help="Which dataset splits to write",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=None,
        help="Variant name (repeatable). Default: all registry variants.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print paths without writing JSON",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    grainseg = args.grainseg_root or default_grainseg_root()
    grainseg = grainseg.resolve()

    splits: tuple[str, ...]
    if args.split == "both":
        splits = ("train", "test")
    else:
        splits = (args.split,)

    variants = tuple(args.variant) if args.variant else all_variant_names()
    written: list[Path] = []

    for split in splits:
        for variant in variants:
            manifest = build_unet_whole_manifest(
                split=split,  # type: ignore[arg-type]
                variant=variant,
                grainseg_root=grainseg,
            )
            validate_dataset_manifest(manifest)
            out_path = default_whole_manifest_path(grainseg, split, variant)  # type: ignore[arg-type]
            if args.dry_run:
                print(f"would write {out_path} ({len(manifest.samples)} sample(s))")
            else:
                write_dataset_manifest(out_path, manifest)
                print(out_path)
                written.append(out_path)

    if args.dry_run:
        print(f"Dry run: {len(variants) * len(splits)} manifest(s) validated.", file=sys.stderr)
    else:
        print(f"Wrote {len(written)} manifest(s) under {grainseg}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
