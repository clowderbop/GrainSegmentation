"""Shared JSON manifest loading, validation, and staging helpers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

from common.variants import VariantSpec, get_variant

PathBase = Literal["grainseg_root", "work_root"]
GtOrigin = Literal["patch_stem", "whole_image"]
ManifestSplit = Literal["train", "test"]


@dataclass(frozen=True)
class ManifestSampleRow:
    """One manifest sample; use ``image`` *or* ``images``, not both."""

    sample_id: str
    image: str | None = None
    images: tuple[str, ...] | None = None
    mask: str | None = None
    gt_gpkg: str | None = None
    gt_origin: GtOrigin | None = None
    gt_txt: str | None = None
    pred_instances: str | None = None
    semantic: str | None = None

    def __post_init__(self) -> None:
        has_image = self.image is not None
        has_images = self.images is not None
        if has_image == has_images:
            raise ValueError(
                'Manifest sample requires exactly one of "image" or "images"'
            )

    @property
    def uses_multi_input(self) -> bool:
        return self.images is not None

    def anchor_image_path(self, *, suffix: str = "_PPL") -> str:
        if self.image is not None:
            return self.image
        assert self.images is not None
        for path in self.images:
            if Path(path).stem.endswith(suffix) or suffix in Path(path).stem:
                return path
        return self.images[0]


@dataclass(frozen=True)
class DatasetManifest:
    """Top-level manifest document (see ``docs/manifests.md``)."""

    schema_version: int
    variant: str
    unit: str
    grainseg_root: str
    path_base: PathBase
    samples: tuple[ManifestSampleRow, ...] = field(default_factory=tuple)
    source_path: Path | None = None

    @property
    def manifest_dir(self) -> Path:
        if self.source_path is not None:
            return self.source_path.parent.resolve()
        return Path(self.grainseg_root).resolve()


def whole_manifest_relpath(split: ManifestSplit, variant: str) -> Path:
    return Path("dataset") / split / "manifests" / f"{variant}.whole.json"


def default_whole_manifest_path(grainseg_root: str | Path, split: ManifestSplit, variant: str) -> Path:
    return Path(grainseg_root).resolve() / whole_manifest_relpath(split, variant)


def patch_dir_relpath(split: ManifestSplit, variant: str) -> Path:
    return Path("dataset") / split / "patches" / variant


def patch_manifest_relpath(split: ManifestSplit, variant: str) -> Path:
    return patch_dir_relpath(split, variant) / "manifest.json"


def default_patch_manifest_path(
    grainseg_root: str | Path, split: ManifestSplit, variant: str
) -> Path:
    return Path(grainseg_root).resolve() / patch_manifest_relpath(split, variant)


def resolve_manifest_path(raw: str, base_dir: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def manifest_path_base_dir(manifest: DatasetManifest) -> Path:
    if manifest.path_base == "grainseg_root":
        return Path(manifest.grainseg_root).resolve()
    if manifest.path_base == "work_root":
        return Path(manifest.grainseg_root).resolve()
    raise ValueError(f"Unsupported path_base: {manifest.path_base!r}")


def resolve_row_path(manifest: DatasetManifest, raw: str | None) -> Path | None:
    if raw is None:
        return None
    return resolve_manifest_path(raw, manifest_path_base_dir(manifest))


def load_manifest_json(path: Path) -> list[dict[str, Any]]:
    """Load raw sample dicts from ``{"samples": [...]}`` (legacy helper)."""
    manifest = load_dataset_manifest(path)
    return [sample_row_to_dict(row) for row in manifest.samples]


def load_dataset_manifest(path: Path) -> DatasetManifest:
    path = path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f'Manifest {path} must be a JSON object')
    manifest = _parse_dataset_manifest(payload, source_path=path)
    validate_dataset_manifest(manifest)
    return manifest


def validate_dataset_manifest(manifest: DatasetManifest) -> None:
    if manifest.schema_version != 1:
        raise ValueError(f"Unsupported schema_version: {manifest.schema_version}")

    spec = get_variant(manifest.variant)
    if manifest.path_base not in ("grainseg_root", "work_root"):
        raise ValueError(f"Unsupported path_base: {manifest.path_base!r}")

    if not manifest.samples:
        raise ValueError("Manifest samples must not be empty")

    stacked_train = spec.paths.train_mosaic_stacked
    stacked_test = spec.paths.test_mosaic_stacked

    for idx, row in enumerate(manifest.samples):
        _validate_sample_row(
            row,
            idx=idx,
            spec=spec,
            unit=manifest.unit,
            stacked_paths={stacked_train, stacked_test},
        )


def _validate_sample_row(
    row: ManifestSampleRow,
    *,
    idx: int,
    spec: VariantSpec,
    unit: str,
    stacked_paths: set[str],
) -> None:
    if not row.sample_id:
        raise ValueError(f"manifest samples[{idx}] requires non-empty sample_id")

    if row.images is not None:
        if len(row.images) != spec.unet.num_inputs:
            raise ValueError(
                f"manifest samples[{idx}] images length {len(row.images)} != "
                f"unet.num_inputs {spec.unet.num_inputs} for variant {spec.name!r}"
            )
        for path in row.images:
            if spec.unet.num_inputs > 1 and path in stacked_paths:
                raise ValueError(
                    f"manifest samples[{idx}] must not list stacked YOLO mosaic: {path}"
                )
        if unit == "whole" and row.image is not None:
            raise ValueError(f"manifest samples[{idx}] has both image and images")
    elif row.image is not None:
        pass
    else:
        raise ValueError(f'manifest samples[{idx}] requires "image" or "images"')

    if row.gt_origin is not None and row.gt_origin not in ("patch_stem", "whole_image"):
        raise ValueError(f"manifest samples[{idx}] invalid gt_origin: {row.gt_origin!r}")


def collect_manifest_image_paths(
    manifest_path: Path,
) -> list[tuple[Path, str]]:
    manifest = load_dataset_manifest(manifest_path)
    samples: list[tuple[Path, str]] = []
    for row in manifest.samples:
        if row.image is not None:
            image_path = resolve_row_path(manifest, row.image)
        else:
            image_path = resolve_row_path(manifest, row.anchor_image_path())
        assert image_path is not None
        samples.append((image_path, row.sample_id))
    if not samples:
        raise ValueError(f"Manifest contains no samples: {manifest_path}")
    return samples


def collect_manifest_unet_samples(
    manifest: Path | DatasetManifest,
    *,
    mask_dir: str | Path | None = None,
    mask_ext: str | None = None,
    mask_stem_suffix: str = "_labels",
) -> list[dict[str, Any]]:
    """Sample dicts for ``unet.predict`` / training (``images``, ``id``, optional ``mask``)."""
    doc = manifest if isinstance(manifest, DatasetManifest) else load_dataset_manifest(manifest)
    base = manifest_path_base_dir(doc)
    mask_root = Path(mask_dir).resolve() if mask_dir is not None else None
    mask_exts = [mask_ext] if mask_ext else [".tif", ".tiff"]

    spec = get_variant(doc.variant)
    samples: list[dict[str, Any]] = []
    for row in doc.samples:
        if row.images is not None:
            rel_paths = row.images
        elif row.image is not None and spec.unet.num_inputs == 1:
            rel_paths = (row.image,)
        else:
            raise ValueError(
                f"U-Net manifest row {row.sample_id!r} requires multi-input images"
            )
        image_paths = []
        for rel in rel_paths:
            path = resolve_manifest_path(rel, base)
            if not path.is_file():
                raise FileNotFoundError(f"Missing manifest image: {path}")
            image_paths.append(str(path))

        sample: dict[str, Any] = {"images": image_paths, "id": row.sample_id}

        mask_rel = row.mask
        if mask_rel is not None:
            mask_path = resolve_manifest_path(mask_rel, base)
            if not mask_path.is_file():
                raise FileNotFoundError(f"Missing manifest mask: {mask_path}")
            sample["mask"] = str(mask_path)
        elif mask_root is not None:
            found = None
            for ext in mask_exts:
                ext = ext if ext.startswith(".") else f".{ext}"
                candidate = mask_root / f"{row.sample_id}{mask_stem_suffix}{ext}"
                if candidate.is_file():
                    found = candidate
                    break
            if found is None:
                raise FileNotFoundError(
                    f"Missing raster mask for {row.sample_id} under {mask_root}"
                )
            sample["mask"] = str(found)

        samples.append(sample)
    return samples


def iter_manifest_asset_paths(manifest: DatasetManifest) -> Iterable[tuple[str, str]]:
    """Yield ``(relative_path, field_name)`` for each file referenced in samples."""
    for row in manifest.samples:
        if row.image is not None:
            yield row.image, "image"
        if row.images is not None:
            for index, path in enumerate(row.images):
                yield path, f"images[{index}]"
        if row.mask is not None:
            yield row.mask, "mask"
        if row.gt_gpkg is not None:
            yield row.gt_gpkg, "gt_gpkg"
        if row.gt_txt is not None:
            yield row.gt_txt, "gt_txt"
        if row.pred_instances is not None:
            yield row.pred_instances, "pred_instances"
        if row.semantic is not None:
            yield row.semantic, "semantic"


def build_eval_manifest(
    source: DatasetManifest,
    *,
    pred_instances_dir: Path,
    manifest_parent: Path,
    gt_gpkg: str | Path | None = None,
    anchor_suffix: str = "_PPL",
) -> DatasetManifest:
    """Single-image eval manifest with ``pred_instances`` filled in."""
    from common.instance_predictions import instance_map_filename

    manifest_parent = Path(manifest_parent).resolve()
    pred_instances_dir = pred_instances_dir.resolve()
    rows: list[ManifestSampleRow] = []
    for row in source.samples:
        anchor = row.anchor_image_path(suffix=anchor_suffix)
        anchor_resolved = resolve_row_path(source, anchor)
        assert anchor_resolved is not None
        try:
            rel_image = os.path.relpath(anchor_resolved, manifest_parent)
        except ValueError:
            rel_image = str(anchor_resolved)
        pred_path = pred_instances_dir / instance_map_filename(row.sample_id)
        try:
            rel_pred = os.path.relpath(pred_path, manifest_parent)
        except ValueError:
            rel_pred = str(pred_path)
        gt = row.gt_gpkg
        if gt is None and gt_gpkg is not None:
            try:
                gt = os.path.relpath(Path(gt_gpkg).resolve(), manifest_parent)
            except ValueError:
                gt = str(Path(gt_gpkg).resolve())
        rows.append(
            ManifestSampleRow(
                sample_id=row.sample_id,
                image=rel_image,
                gt_gpkg=gt,
                gt_origin=row.gt_origin or "whole_image",
                pred_instances=rel_pred,
            )
        )
    return DatasetManifest(
        schema_version=source.schema_version,
        variant=source.variant,
        unit=source.unit,
        grainseg_root=str(manifest_parent),
        path_base="work_root",
        samples=tuple(rows),
    )


def write_dataset_manifest(path: Path, manifest: DatasetManifest) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dataset_manifest_to_dict(manifest)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def dataset_manifest_to_dict(manifest: DatasetManifest) -> dict[str, Any]:
    return {
        "schema_version": manifest.schema_version,
        "variant": manifest.variant,
        "unit": manifest.unit,
        "grainseg_root": manifest.grainseg_root,
        "path_base": manifest.path_base,
        "samples": [sample_row_to_dict(row) for row in manifest.samples],
    }


def sample_row_to_dict(row: ManifestSampleRow) -> dict[str, Any]:
    payload: dict[str, Any] = {"sample_id": row.sample_id}
    if row.image is not None:
        payload["image"] = row.image
    if row.images is not None:
        payload["images"] = list(row.images)
    if row.mask is not None:
        payload["mask"] = row.mask
    if row.gt_gpkg is not None:
        payload["gt_gpkg"] = row.gt_gpkg
    if row.gt_origin is not None:
        payload["gt_origin"] = row.gt_origin
    if row.gt_txt is not None:
        payload["gt_txt"] = row.gt_txt
    if row.pred_instances is not None:
        payload["pred_instances"] = row.pred_instances
    if row.semantic is not None:
        payload["semantic"] = row.semantic
    return payload


def _parse_dataset_manifest(
    payload: dict[str, Any], *, source_path: Path | None
) -> DatasetManifest:
    try:
        schema_version = int(payload["schema_version"])
        variant = str(payload["variant"])
        unit = str(payload["unit"])
        grainseg_root = str(payload["grainseg_root"])
        path_base = str(payload["path_base"])
    except KeyError as exc:
        raise ValueError(f'Manifest missing required key: {exc}') from exc

    if path_base not in ("grainseg_root", "work_root"):
        raise ValueError(f"Invalid path_base: {path_base!r}")

    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list):
        raise ValueError('Manifest "samples" must be a list')

    samples = tuple(_parse_sample_row(raw, index) for index, raw in enumerate(raw_samples))
    return DatasetManifest(
        schema_version=schema_version,
        variant=variant,
        unit=unit,
        grainseg_root=grainseg_root,
        path_base=path_base,  # type: ignore[arg-type]
        samples=samples,
        source_path=source_path,
    )


def _parse_sample_row(raw: Any, index: int) -> ManifestSampleRow:
    if not isinstance(raw, dict):
        raise ValueError(f"manifest samples[{index}] must be an object")
    sample_id = str(raw.get("sample_id") or "")
    if not sample_id:
        stem_fallback = raw.get("image") or (raw.get("images") or [None])[0]
        if stem_fallback:
            sample_id = Path(str(stem_fallback)).stem
    image = raw.get("image")
    images_raw = raw.get("images")
    images: tuple[str, ...] | None = None
    if images_raw is not None:
        if not isinstance(images_raw, list) or not images_raw:
            raise ValueError(f"manifest samples[{index}].images must be a non-empty list")
        images = tuple(str(p) for p in images_raw)
    return ManifestSampleRow(
        sample_id=sample_id,
        image=str(image) if image is not None else None,
        images=images,
        mask=str(raw["mask"]) if raw.get("mask") is not None else None,
        gt_gpkg=str(raw["gt_gpkg"]) if raw.get("gt_gpkg") is not None else None,
        gt_origin=raw.get("gt_origin"),  # type: ignore[arg-type]
        gt_txt=str(raw["gt_txt"]) if raw.get("gt_txt") is not None else None,
        pred_instances=(
            str(raw["pred_instances"]) if raw.get("pred_instances") is not None else None
        ),
        semantic=str(raw["semantic"]) if raw.get("semantic") is not None else None,
    )


def build_yolo_whole_manifest(
    *,
    split: ManifestSplit,
    variant: str,
    grainseg_root: str | Path,
) -> DatasetManifest:
    """Construct a YOLO whole-section manifest (stacked mosaic ``image`` row)."""
    spec = get_variant(variant)
    root = Path(grainseg_root).resolve()
    if split == "train":
        image = spec.paths.train_mosaic_stacked
        labels_gpkg = spec.paths.train_labels_gpkg
    else:
        image = spec.paths.test_mosaic_stacked
        labels_gpkg = spec.paths.test_labels_gpkg

    mosaic_path = root / image
    if not mosaic_path.is_file():
        raise FileNotFoundError(f"Missing stacked mosaic for {variant} ({split}): {mosaic_path}")

    return DatasetManifest(
        schema_version=1,
        variant=variant,
        unit="whole",
        grainseg_root=str(root),
        path_base="grainseg_root",
        samples=(
            ManifestSampleRow(
                sample_id=split,
                image=image,
                gt_gpkg=labels_gpkg,
                gt_origin="whole_image",
            ),
        ),
    )


def whole_manifest_overlay_anchor(
    manifest: DatasetManifest,
    *,
    channel_suffix: str = "_PPL",
) -> tuple[str, str, str | None]:
    """Return ``(sample_id, image_rel, mask_rel)`` for overlay plotting."""
    if not manifest.samples:
        raise ValueError("Manifest has no samples")
    row = manifest.samples[0]
    if row.images is not None:
        image_rel = row.anchor_image_path(suffix=channel_suffix)
    elif row.image is not None:
        image_rel = row.image
    else:
        raise ValueError("Manifest row has no image paths")
    return row.sample_id, image_rel, row.mask


def build_unet_whole_manifest(
    *,
    split: ManifestSplit,
    variant: str,
    grainseg_root: str | Path,
) -> DatasetManifest:
    """Construct a U-Net whole-section manifest from the variant registry."""
    spec = get_variant(variant)
    root = Path(grainseg_root).resolve()
    sample_id = split
    if split == "train":
        labels_raster = spec.paths.train_labels_raster
        labels_gpkg = spec.paths.train_labels_gpkg
        channel_paths = [
            spec.paths.train_channel_template.format(suffix=suffix)
            for suffix in spec.unet.input_suffixes
        ]
    else:
        labels_raster = spec.paths.test_labels_raster
        labels_gpkg = spec.paths.test_labels_gpkg
        channel_paths = [
            spec.paths.test_channel_template.format(suffix=suffix)
            for suffix in spec.unet.input_suffixes
        ]

    missing = [p for p in channel_paths if not (root / p).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing channel TIFFs for {variant} ({split}): {', '.join(missing)}"
        )

    return DatasetManifest(
        schema_version=1,
        variant=variant,
        unit="whole",
        grainseg_root=str(root),
        path_base="grainseg_root",
        samples=(
            ManifestSampleRow(
                sample_id=sample_id,
                images=tuple(channel_paths),
                mask=labels_raster,
                gt_gpkg=labels_gpkg,
                gt_origin="whole_image",
            ),
        ),
    )
