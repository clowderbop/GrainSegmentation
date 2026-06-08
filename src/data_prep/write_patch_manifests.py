#!/usr/bin/env python3
"""Write YOLO patch manifests and optional U-Net manifests on an existing scratch tree.

After ``create_patch_datasets.sh`` (or to refresh manifests without re-patchifying)::

    uv run --directory src/data_prep python write_patch_manifests.py \\
        --grainseg-root "$SCRATCH/GrainSeg"

Add ``--write-unet-manifests`` to run ``crop_unet_masks_from_yolo_patches`` for test
patches and write ``dataset/test/unet_from_yolo/{variant}/manifest.json``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common.patch_manifests import (
    default_unet_patch_root,
    write_all_yolo_dataset_yamls,
    write_yolo_patch_manifest,
)
from common.variants import all_variant_names, default_grainseg_root, get_variant


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
        help="Which dataset splits to write YOLO patch manifests for",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=None,
        help="Variant name (repeatable). Default: all registry variants.",
    )
    parser.add_argument(
        "--write-yolo-yamls",
        action="store_true",
        help="Also write Ultralytics data.yaml files under each patch variant dir",
    )
    parser.add_argument(
        "--write-unet-manifests",
        action="store_true",
        help="Crop test U-Net patches from YOLO manifests and write unet_from_yolo manifests",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without writing",
    )
    return parser.parse_args()


def _crop_unet_test_patches(
    *,
    grainseg_root: Path,
    variant: str,
    dry_run: bool,
) -> None:
    from crop_unet_masks_from_yolo_patches import crop_from_yolo_manifest

    spec = get_variant(variant)
    split = "test"
    yolo_manifest = (
        grainseg_root / "dataset" / split / "patches" / variant / "manifest.json"
    )
    if not yolo_manifest.is_file():
        raise FileNotFoundError(f"YOLO patch manifest not found: {yolo_manifest}")

    ref_tiff = grainseg_root / spec.paths.test_mosaic_stacked
    ref_mask = grainseg_root / spec.paths.test_labels_raster
    unet_root = grainseg_root / default_unet_patch_root(split, variant)
    if dry_run:
        print(f"would crop U-Net test patches: {variant} -> {unet_root}")
        return
    crop_from_yolo_manifest(
        yolo_manifest_path=yolo_manifest,
        variant=variant,
        dataset_split=split,
        grainseg_root=grainseg_root,
        reference_tiff=ref_tiff,
        reference_mask=ref_mask,
        output_images_dir=unet_root / "images",
        output_masks_dir=unet_root / "masks",
        write_manifest=True,
    )


def main() -> int:
    args = _parse_args()
    grainseg = (args.grainseg_root or default_grainseg_root()).resolve()

    splits: tuple[str, ...]
    if args.split == "both":
        splits = ("train", "test")
    else:
        splits = (args.split,)

    variants = tuple(args.variant) if args.variant else all_variant_names()
    written: list[Path] = []

    for split in splits:
        patches_parent = grainseg / "dataset" / split / "patches"
        for variant in variants:
            if args.dry_run:
                print(f"would write YOLO manifest: {split} {variant}")
                continue
            path = write_yolo_patch_manifest(
                variant=variant,
                split=split,  # type: ignore[arg-type]
                grainseg_root=grainseg,
            )
            written.append(path)
            print(f"Wrote {path}")

        if args.write_yolo_yamls and not args.dry_run:
            held_out = split == "test"
            yaml_paths = write_all_yolo_dataset_yamls(
                patches_parent,
                held_out=held_out,
                variants=variants,
            )
            for yaml_path in yaml_paths:
                print(f"Wrote {yaml_path}")

    if args.write_unet_manifests:
        for variant in variants:
            _crop_unet_test_patches(
                grainseg_root=grainseg,
                variant=variant,
                dry_run=args.dry_run,
            )

    if args.dry_run:
        return 0
    if not written and not args.write_unet_manifests:
        print("No manifests written.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
