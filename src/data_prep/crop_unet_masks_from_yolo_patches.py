from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import tifffile

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
_SRC_ROOT = _SCRIPT_DIR.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from common.image_io import load_tiff_channel_first
from common.patching import (
    extract_padded_patch_2d,
    parse_region_patch_stem,
    tile_patch_bounds,
)

load_image_channel_first = load_tiff_channel_first
parse_patch_stem = parse_region_patch_stem
compute_tile_patch_bounds = tile_patch_bounds


def _raster_hw(path: Path) -> tuple[int, int]:
    arr = load_image_channel_first(path)
    _, h, w = arr.shape
    return int(h), int(w)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        )
    parser.add_argument(
        "--reference-tiff",
        type=Path,
        required=True,
        )
    parser.add_argument(
        "--reference-mask",
        type=Path,
        required=True,
        )
    parser.add_argument(
        "--yolo-images-dir",
        type=Path,
        required=True,
        )
    parser.add_argument(
        "--output-images-dir",
        type=Path,
        required=True,
        )
    parser.add_argument(
        "--output-masks-dir",
        type=Path,
        required=True,
        )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=1024,
        )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=4096,
        )
    parser.add_argument(
        "--image-suffix",
        default="_PPL",
        )
    parser.add_argument(
        "--mask-stem-suffix",
        default="_labels",
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
