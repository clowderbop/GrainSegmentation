from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import cast

import numpy as np
import tifffile

from common.image_io import load_tiff_channel_first
from common.manifest_io import (
    ManifestSplit,
    load_dataset_manifest,
    manifest_path_base_dir,
    resolve_manifest_path,
)
from common.patch_manifests import (
    default_unet_patch_root,
    write_unet_patch_manifest,
)
from common.patching import (
    extract_padded_patch_2d,
    extract_padded_patch_channel_first,
    parse_region_patch_stem,
    tile_patch_bounds,
)
from common.variants import get_variant


def _raster_hw(path: Path) -> tuple[int, int]:
    arr = load_tiff_channel_first(path)
    _, h, w = arr.shape
    return int(h), int(w)


def _channel_mosaic_path(
    grainseg_root: Path, split: str, suffix: str, variant: str
) -> Path:
    spec = get_variant(variant)
    template = (
        spec.paths.train_channel_template
        if split == "train"
        else spec.paths.test_channel_template
    )
    return grainseg_root / template.format(suffix=suffix)


def _resolve_channel_mosaics(
    grainseg_root: Path, split: str, variant: str, suffixes: tuple[str, ...]
) -> dict[str, np.ndarray]:
    loaded: dict[str, np.ndarray] = {}
    for suffix in suffixes:
        path = _channel_mosaic_path(grainseg_root, split, suffix, variant)
        if not path.is_file():
            raise FileNotFoundError(f"Channel mosaic not found: {path}")
        loaded[suffix] = load_tiff_channel_first(path)
    return loaded


def _crop_mask_patch(
    mask: np.ndarray,
    stem: str,
    *,
    height: int,
    width: int,
    tile_size: int,
    patch_size: int,
) -> np.ndarray:
    region_idx, py0, px0 = parse_region_patch_stem(stem)
    y0, y1, x0, x1 = tile_patch_bounds(
        region_idx,
        py0,
        px0,
        height=height,
        width=width,
        tile_size=tile_size,
        patch_size=patch_size,
    )
    return extract_padded_patch_2d(mask, y0, y1, x0, x1, patch_size)


def _process_single_input_row(
    *,
    row_image: Path,
    stem: str,
    mask: np.ndarray,
    height: int,
    width: int,
    tile_size: int,
    patch_size: int,
    output_images_dir: Path,
    output_masks_dir: Path,
    image_suffix: str,
    mask_stem_suffix: str,
) -> None:
    ext = row_image.suffix or ".tif"
    out_img = output_images_dir / f"{stem}{image_suffix}{ext}"
    out_msk = output_masks_dir / f"{stem}{mask_stem_suffix}{ext}"
    shutil.copy2(row_image, out_img)
    patch_mask = _crop_mask_patch(
        mask,
        stem,
        height=height,
        width=width,
        tile_size=tile_size,
        patch_size=patch_size,
    )
    tifffile.imwrite(out_msk, patch_mask, compression="deflate")


def _process_multi_input_row(
    *,
    stem: str,
    channel_mosaics: dict[str, np.ndarray],
    input_suffixes: tuple[str, ...],
    mask: np.ndarray,
    height: int,
    width: int,
    tile_size: int,
    patch_size: int,
    output_images_dir: Path,
    output_masks_dir: Path,
    mask_stem_suffix: str,
) -> None:
    region_idx, py0, px0 = parse_region_patch_stem(stem)
    y0, y1, x0, x1 = tile_patch_bounds(
        region_idx,
        py0,
        px0,
        height=height,
        width=width,
        tile_size=tile_size,
        patch_size=patch_size,
    )
    ext = ".tif"
    for suffix in input_suffixes:
        mosaic = channel_mosaics[suffix]
        patch = extract_padded_patch_channel_first(mosaic, (y0, y1, x0, x1), patch_size)
        out_img = output_images_dir / f"{stem}{suffix}{ext}"
        tifffile.imwrite(
            out_img,
            np.clip(patch, 0, 255).astype(np.uint8, copy=False),
            metadata={"axes": "CYX"},
            compression="deflate",
        )
    patch_mask = extract_padded_patch_2d(mask, y0, y1, x0, x1, patch_size)
    out_msk = output_masks_dir / f"{stem}{mask_stem_suffix}{ext}"
    tifffile.imwrite(out_msk, patch_mask, compression="deflate")


def crop_from_yolo_manifest(
    *,
    yolo_manifest_path: Path,
    variant: str,
    dataset_split: str,
    grainseg_root: Path,
    reference_tiff: Path,
    reference_mask: Path,
    output_images_dir: Path,
    output_masks_dir: Path,
    patch_size: int = 1024,
    tile_size: int = 4096,
    image_suffix: str = "_PPL",
    mask_stem_suffix: str = "_labels",
    write_manifest: bool = True,
) -> int:
    doc = load_dataset_manifest(yolo_manifest_path)
    if doc.variant != variant:
        raise ValueError(
            f"Manifest variant {doc.variant!r} != requested variant {variant!r}"
        )
    spec = get_variant(variant)
    base = manifest_path_base_dir(doc)

    height, width = _raster_hw(reference_tiff)
    mask = tifffile.imread(reference_mask)
    if mask.ndim != 2:
        raise ValueError(f"Reference mask must be 2D, got {mask.shape}")
    if mask.shape != (height, width):
        raise ValueError(
            f"Mask shape {mask.shape} does not match reference image size "
            f"({height}, {width}) from {reference_tiff}"
        )

    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_masks_dir.mkdir(parents=True, exist_ok=True)

    channel_mosaics: dict[str, np.ndarray] | None = None
    if spec.unet.num_inputs > 1:
        channel_mosaics = _resolve_channel_mosaics(
            grainseg_root, dataset_split, variant, spec.unet.input_suffixes
        )

    count = 0
    for row in doc.samples:
        if row.image is None:
            raise ValueError(
                f"YOLO patch manifest row {row.sample_id!r} must use image field"
            )
        yolo_image = resolve_manifest_path(row.image, base)
        stem = row.sample_id
        if spec.unet.num_inputs == 1:
            _process_single_input_row(
                row_image=yolo_image,
                stem=stem,
                mask=mask,
                height=height,
                width=width,
                tile_size=tile_size,
                patch_size=patch_size,
                output_images_dir=output_images_dir,
                output_masks_dir=output_masks_dir,
                image_suffix=image_suffix,
                mask_stem_suffix=mask_stem_suffix,
            )
        else:
            assert channel_mosaics is not None
            _process_multi_input_row(
                stem=stem,
                channel_mosaics=channel_mosaics,
                input_suffixes=spec.unet.input_suffixes,
                mask=mask,
                height=height,
                width=width,
                tile_size=tile_size,
                patch_size=patch_size,
                output_images_dir=output_images_dir,
                output_masks_dir=output_masks_dir,
                mask_stem_suffix=mask_stem_suffix,
            )
        count += 1

    if write_manifest:
        write_unet_patch_manifest(
            variant=variant,
            split=cast(ManifestSplit, dataset_split),
            grainseg_root=grainseg_root,
            yolo_manifest=doc,
        )

    return count


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Crop U-Net patch images/masks from YOLO patch manifest rows.",
    )
    parser.add_argument(
        "--yolo-manifest",
        type=Path,
        required=True,
        help="YOLO patch manifest.json (image + gt_txt rows)",
    )
    parser.add_argument("--variant", required=True)
    parser.add_argument(
        "--dataset-split",
        choices=("train", "test"),
        required=True,
    )
    parser.add_argument(
        "--grainseg-root",
        type=Path,
        required=True,
        help="GrainSeg root for multi-input channel mosaics and manifest output",
    )
    parser.add_argument("--reference-tiff", type=Path, required=True)
    parser.add_argument("--reference-mask", type=Path, required=True)
    parser.add_argument("--yolo-images-dir", type=Path, default=None)
    parser.add_argument("--output-images-dir", type=Path, default=None)
    parser.add_argument("--output-masks-dir", type=Path, default=None)
    parser.add_argument("--patch-size", type=int, default=1024)
    parser.add_argument("--tile-size", type=int, default=4096)
    parser.add_argument("--image-suffix", default="_PPL")
    parser.add_argument("--mask-stem-suffix", default="_labels")
    parser.add_argument(
        "--no-write-manifest",
        action="store_true",
        help="Skip writing dataset/{split}/unet_from_yolo/{variant}/manifest.json",
    )
    args = parser.parse_args(argv)

    grainseg = args.grainseg_root.resolve()
    unet_rel = default_unet_patch_root(args.dataset_split, args.variant)
    unet_root = grainseg / unet_rel

    yolo_images = args.yolo_images_dir
    if yolo_images is None:
        split_name = "test" if args.dataset_split == "test" else "train"
        yolo_images = (
            grainseg
            / "dataset"
            / args.dataset_split
            / "patches"
            / args.variant
            / "images"
            / split_name
        )

    out_images = args.output_images_dir or (unet_root / "images")
    out_masks = args.output_masks_dir or (unet_root / "masks")

    count = crop_from_yolo_manifest(
        yolo_manifest_path=args.yolo_manifest.resolve(),
        variant=args.variant,
        dataset_split=args.dataset_split,
        grainseg_root=grainseg,
        reference_tiff=args.reference_tiff.resolve(),
        reference_mask=args.reference_mask.resolve(),
        output_images_dir=out_images,
        output_masks_dir=out_masks,
        patch_size=args.patch_size,
        tile_size=args.tile_size,
        image_suffix=args.image_suffix,
        mask_stem_suffix=args.mask_stem_suffix,
        write_manifest=not args.no_write_manifest,
    )
    print(f"Wrote {count} image / mask pairs under {out_images} and {out_masks}.")


if __name__ == "__main__":
    main()
