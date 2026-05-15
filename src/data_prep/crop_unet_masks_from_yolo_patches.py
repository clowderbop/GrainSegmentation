from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import tifffile

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from split_tiff_gpkg_to_yolo import _region_bounds, load_image_channel_first

PATCH_STEM_RE = re.compile(r"^region_(\d+)_y(\d+)_x(\d+)$")


def parse_patch_stem(name_without_ext: str) -> tuple[int, int, int]:
    match = PATCH_STEM_RE.match(name_without_ext)
    if match is None:
        raise ValueError(
            f"Expected stem like region_0123_y01234_x05678, got: {name_without_ext!r}"
        )
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def compute_tile_patch_bounds(
    region_idx: int,
    patch_y0: int,
    patch_x0: int,
    *,
    height: int,
    width: int,
    tile_size: int,
    patch_size: int,
) -> tuple[int, int, int, int]:
    """Return (y0, y1, x0, x1) crop from the full mask, before zero-padding."""
    regions = _region_bounds(height, width, tile_size)
    if region_idx < 0 or region_idx >= len(regions):
        raise ValueError(
            f"region_idx {region_idx} out of range for "
            f"{len(regions)} tiles (H={height}, W={width}, tile_size={tile_size})"
        )
    ry0, ry1, rx0, rx1 = regions[region_idx]
    if patch_y0 < ry0 or patch_y0 >= ry1 or patch_x0 < rx0 or patch_x0 >= rx1:
        raise ValueError(
            f"Patch origin (y={patch_y0}, x={patch_x0}) outside "
            f"region {region_idx} bounds y=[{ry0},{ry1}), x=[{rx0},{rx1})"
        )
    patch_y1 = min(patch_y0 + patch_size, ry1)
    patch_x1 = min(patch_x0 + patch_size, rx1)
    return patch_y0, patch_y1, patch_x0, patch_x1


def extract_padded_patch_2d(
    mask: np.ndarray,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    patch_size: int,
) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D, got shape {mask.shape}")
    patch = mask[y0:y1, x0:x1]
    out = np.zeros((patch_size, patch_size), dtype=mask.dtype)
    h, w = patch.shape
    out[:h, :w] = patch
    return out


def _raster_hw(path: Path) -> tuple[int, int]:
    """Return (height, width) for a mosaic TIFF (matches patchify reference)."""
    arr = load_image_channel_first(path)
    _, h, w = arr.shape
    return int(h), int(w)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Crop full UNet mask to YOLO val patch grid; copy images with "
            "evaluate-friendly stems."
        )
    )
    parser.add_argument(
        "--reference-tiff",
        type=Path,
        required=True,
        help="Same mosaic GeoTIFF used by patchify for this variant (sets H, W).",
    )
    parser.add_argument(
        "--reference-mask",
        type=Path,
        required=True,
        help="Full-scene raster mask [0,2] aligned to --reference-tiff.",
    )
    parser.add_argument(
        "--yolo-images-dir",
        type=Path,
        required=True,
        help="YOLO patch image dir (e.g. images/test for held-out mosaics; filenames region_*_y*_x*).",
    )
    parser.add_argument(
        "--output-images-dir",
        type=Path,
        required=True,
        help="Writes {stem}{image_suffix}.tif (copies of each patch image).",
    )
    parser.add_argument(
        "--output-masks-dir",
        type=Path,
        required=True,
        help="Writes {stem}{mask_stem_suffix}.tif (cropped + padded masks).",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=1024,
        help="Patch edge length (default matches patchify).",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=4096,
        help="Tile size used in patchify (default 4096).",
    )
    parser.add_argument(
        "--image-suffix",
        default="_PPL",
        help="Suffix before extension for UNet image stems (default _PPL).",
    )
    parser.add_argument(
        "--mask-stem-suffix",
        default="_labels",
        help="Inserted before .tif for mask filenames (default _labels).",
    )
    args = parser.parse_args(argv)

    height, width = _raster_hw(args.reference_tiff)
    mask = tifffile.imread(args.reference_mask)
    if mask.ndim != 2:
        raise ValueError(f"Reference mask must be 2D, got {mask.shape}")
    if mask.shape != (height, width):
        raise ValueError(
            f"Mask shape {mask.shape} does not match reference image size "
            f"({height}, {width}) from {args.reference_tiff}"
        )

    args.output_images_dir.mkdir(parents=True, exist_ok=True)
    args.output_masks_dir.mkdir(parents=True, exist_ok=True)

    patches = sorted(
        list(args.yolo_images_dir.glob("*.tif"))
        + list(args.yolo_images_dir.glob("*.tiff"))
    )
    if not patches:
        print(f"No TIFF patches in {args.yolo_images_dir}", file=sys.stderr)
        sys.exit(1)

    for img_path in patches:
        stem = img_path.stem
        region_idx, py0, px0 = parse_patch_stem(stem)
        y0, y1, x0, x1 = compute_tile_patch_bounds(
            region_idx,
            py0,
            px0,
            height=height,
            width=width,
            tile_size=args.tile_size,
            patch_size=args.patch_size,
        )
        patch_mask = extract_padded_patch_2d(mask, y0, y1, x0, x1, args.patch_size)

        out_img = args.output_images_dir / f"{stem}{args.image_suffix}{img_path.suffix}"
        out_msk = (
            args.output_masks_dir / f"{stem}{args.mask_stem_suffix}{img_path.suffix}"
        )
        shutil.copy2(img_path, out_img)
        tifffile.imwrite(out_msk, patch_mask, compression="deflate")

    print(f"Wrote {len(patches)} image / mask pairs under output dirs.")


if __name__ == "__main__":
    main()
