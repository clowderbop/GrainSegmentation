"""Build YOLO and U-Net patch dataset manifests (Phase 4)."""

from __future__ import annotations

from pathlib import Path

from common.manifest_io import (
    DatasetManifest,
    ManifestSampleRow,
    ManifestSplit,
    default_patch_manifest_path,
    patch_dir_relpath,
    validate_dataset_manifest,
    write_dataset_manifest,
)
from common.variants import all_variant_names, get_variant

_PATCH_IMAGE_SUFFIXES = {".tif", ".tiff"}


def labels_gpkg_relpath(split: ManifestSplit) -> str:
    if split == "train":
        return "dataset/train/train_labels.gpkg"
    return "dataset/test/test_labels.gpkg"


def yolo_split_names(*, test: bool) -> tuple[str, ...]:
    return ("test",) if test else ("train", "val")


def render_yolo_dataset_yaml(
    variant: str,
    *,
    held_out: bool = False,
) -> str:
    """Ultralytics data.yaml body (path relative to yaml file)."""
    spec = get_variant(variant)
    if held_out:
        train_p = val_p = test_p = "images/test"
    else:
        train_p = "images/train"
        val_p = "images/val"
        test_p = ""

    lines = [
        "path: .",
        f"train: {train_p}",
        f"val: {val_p}",
        f"test: {test_p}",
    ]
    if spec.yolo.input_channels != 3:
        lines.append(f"channels: {spec.yolo.input_channels}")
    lines.extend(["", "names:", "  0: grain", ""])
    return "\n".join(lines)


def write_yolo_dataset_yaml_file(
    variant: str,
    patch_root: Path,
    *,
    held_out: bool = False,
) -> Path:
    spec = get_variant(variant)
    yaml_name = spec.yolo.yaml_name
    out_path = patch_root / yaml_name
    out_path.write_text(
        render_yolo_dataset_yaml(variant, held_out=held_out),
        encoding="utf-8",
    )
    return out_path


def write_all_yolo_dataset_yamls(
    patches_parent: Path,
    *,
    held_out: bool = False,
    variants: tuple[str, ...] | None = None,
) -> list[Path]:
    written: list[Path] = []
    for variant in variants or all_variant_names():
        variant_dir = patches_parent / variant
        if not variant_dir.is_dir():
            raise FileNotFoundError(f"Patch variant directory not found: {variant_dir}")
        written.append(
            write_yolo_dataset_yaml_file(variant, variant_dir, held_out=held_out)
        )
    return written


def _scan_split_images(
    images_dir: Path,
    labels_dir: Path,
    *,
    patch_root_rel: str,
    split_name: str,
    labels_gpkg_rel: str,
) -> list[ManifestSampleRow]:
    if not images_dir.is_dir():
        return []

    rows: list[ManifestSampleRow] = []
    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in _PATCH_IMAGE_SUFFIXES:
            continue
        stem = image_path.stem
        label_path = labels_dir / f"{stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(
                f"Missing YOLO label for patch {stem}: {label_path}"
            )
        rows.append(
            ManifestSampleRow(
                sample_id=stem,
                image=f"{patch_root_rel}/images/{split_name}/{image_path.name}",
                gt_txt=f"{patch_root_rel}/labels/{split_name}/{label_path.name}",
                gt_gpkg=labels_gpkg_rel,
                gt_origin="patch_stem",
            )
        )
    return rows


def build_yolo_patch_manifest(
    *,
    variant: str,
    split: ManifestSplit,
    grainseg_root: str | Path,
    patch_root: str | Path | None = None,
) -> DatasetManifest:
    """Scan ``dataset/{split}/patches/{variant}/`` and build a YOLO patch manifest."""
    grainseg = Path(grainseg_root).resolve()
    if patch_root is None:
        patch_rel = patch_dir_relpath(split, variant)
        patch_dir = grainseg / patch_rel
    else:
        patch_dir = Path(patch_root).resolve()
        patch_rel = patch_dir.relative_to(grainseg)
    if not patch_dir.is_dir():
        raise FileNotFoundError(f"Patch directory not found: {patch_dir}")

    gpkg_rel = labels_gpkg_relpath(split)
    rows: list[ManifestSampleRow] = []
    for split_name in yolo_split_names(test=split == "test"):
        image_dir = patch_dir / "images" / split_name
        label_dir = patch_dir / "labels" / split_name
        rows.extend(
            _scan_split_images(
                image_dir,
                label_dir,
                patch_root_rel=str(patch_rel).replace("\\", "/"),
                split_name=split_name,
                labels_gpkg_rel=gpkg_rel,
            )
        )

    if not rows:
        raise ValueError(f"No patch samples found under {patch_dir}")

    manifest = DatasetManifest(
        schema_version=1,
        variant=variant,
        unit="patches",
        grainseg_root=str(grainseg),
        path_base="grainseg_root",
        samples=tuple(rows),
    )
    validate_dataset_manifest(manifest)
    return manifest


def write_yolo_patch_manifest(
    *,
    variant: str,
    split: ManifestSplit,
    grainseg_root: str | Path,
    patch_root: str | Path | None = None,
) -> Path:
    manifest = build_yolo_patch_manifest(
        variant=variant,
        split=split,
        grainseg_root=grainseg_root,
        patch_root=patch_root,
    )
    out_path = (
        Path(patch_root).resolve() / "manifest.json"
        if patch_root is not None
        else default_patch_manifest_path(grainseg_root, split, variant)
    )
    write_dataset_manifest(out_path, manifest)
    return out_path


def build_unet_patch_manifest(
    *,
    variant: str,
    split: ManifestSplit,
    grainseg_root: str | Path,
    yolo_manifest: DatasetManifest,
    unet_root_rel: str,
    image_suffix: str = "_PPL",
    mask_stem_suffix: str = "_labels",
) -> DatasetManifest:
    """Build U-Net patch manifest from YOLO rows and ``unet_from_yolo`` layout."""
    spec = get_variant(variant)
    rows: list[ManifestSampleRow] = []
    unet_rel = unet_root_rel.replace("\\", "/").rstrip("/")

    for row in yolo_manifest.samples:
        if row.image is None:
            raise ValueError(
                f"U-Net patch manifest requires YOLO image rows; got multi-input row {row.sample_id!r}"
            )
        stem = row.sample_id
        ext = Path(row.image).suffix or ".tif"
        image_name = f"{stem}{image_suffix}{ext}"
        mask_name = f"{stem}{mask_stem_suffix}{ext}"

        if spec.unet.num_inputs == 1:
            rows.append(
                ManifestSampleRow(
                    sample_id=stem,
                    image=f"{unet_rel}/images/{image_name}",
                    mask=f"{unet_rel}/masks/{mask_name}",
                    gt_gpkg=row.gt_gpkg,
                    gt_origin="patch_stem",
                )
            )
        else:
            channel_paths = [
                f"{unet_rel}/images/{stem}{suffix}{ext}"
                for suffix in spec.unet.input_suffixes
            ]
            rows.append(
                ManifestSampleRow(
                    sample_id=stem,
                    images=tuple(channel_paths),
                    mask=f"{unet_rel}/masks/{mask_name}",
                    gt_gpkg=row.gt_gpkg,
                    gt_origin="patch_stem",
                )
            )

    manifest = DatasetManifest(
        schema_version=1,
        variant=variant,
        unit="patches",
        grainseg_root=str(Path(grainseg_root).resolve()),
        path_base="grainseg_root",
        samples=tuple(rows),
    )
    validate_dataset_manifest(manifest)
    return manifest


def write_unet_patch_manifest(
    *,
    variant: str,
    split: ManifestSplit,
    grainseg_root: str | Path,
    yolo_manifest: DatasetManifest | Path,
    unet_root_rel: str | None = None,
) -> Path:
    if isinstance(yolo_manifest, Path):
        from common.manifest_io import load_dataset_manifest

        yolo_doc = load_dataset_manifest(yolo_manifest)
    else:
        yolo_doc = yolo_manifest

    if unet_root_rel is None:
        unet_root_rel = f"dataset/{split}/unet_from_yolo/{variant}"

    manifest = build_unet_patch_manifest(
        variant=variant,
        split=split,
        grainseg_root=grainseg_root,
        yolo_manifest=yolo_doc,
        unet_root_rel=unet_root_rel,
    )
    out_path = Path(grainseg_root).resolve() / unet_root_rel / "manifest.json"
    write_dataset_manifest(out_path, manifest)
    return out_path


def default_unet_patch_root(split: ManifestSplit, variant: str) -> str:
    return f"dataset/{split}/unet_from_yolo/{variant}"
