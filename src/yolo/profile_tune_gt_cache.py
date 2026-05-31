"""Profile selection ground truth cache — rasterized train GT per variant (ADR 0005)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from common.evaluate_instances import (
    InstanceEvalSample,
    image_dimensions,
    load_gt_instance_map,
)
from common.manifest_io import collect_manifest_image_paths
from common.variants import get_variant, repo_root
from yolo.profile_tune_cli import parse_profile_tune_variants
from yolo.profile_tune_work import (
    default_grainseg_and_run_roots,
    ensure_staged_train_manifest,
)
from yolo.tiled_proposal_cache import file_sha256

GT_CACHE_SCHEMA_VERSION = 1
_INSTANCE_MAP_NAME = "instance_map.npz"
_FINGERPRINT_NAME = "fingerprint.json"


def gt_cache_dir(work_root: Path, variant: str) -> Path:
    return work_root / "gt_cache" / variant


def train_labels_gpkg_path(grainseg_root: Path) -> Path:
    return grainseg_root / get_variant("PPL").paths.train_labels_gpkg


def build_gt_fingerprint(
    *,
    variant: str,
    sample_id: str,
    labels_gpkg: Path,
) -> dict[str, Any]:
    if not labels_gpkg.is_file():
        raise FileNotFoundError(f"Train labels GeoPackage not found: {labels_gpkg}")
    return {
        "schema_version": GT_CACHE_SCHEMA_VERSION,
        "variant": variant,
        "sample_id": sample_id,
        "train_labels_gpkg_sha256": file_sha256(labels_gpkg),
    }


def validate_gt_cache_fingerprint(
    meta: dict[str, Any], expected: dict[str, Any]
) -> None:
    if meta.get("schema_version") != GT_CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"GT cache schema_version {meta.get('schema_version')!r} != "
            f"{GT_CACHE_SCHEMA_VERSION}"
        )
    for key, expected_value in expected.items():
        if meta.get(key) != expected_value:
            raise ValueError(
                f"GT cache fingerprint mismatch for {key!r}: "
                f"cache {meta.get(key)!r} != expected {expected_value!r}"
            )


def write_gt_instance_map_cache(
    cache_dir: Path,
    instance_map: np.ndarray,
    *,
    fingerprint: dict[str, Any],
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_dir / _INSTANCE_MAP_NAME, instance_map=instance_map)
    (cache_dir / _FINGERPRINT_NAME).write_text(
        json.dumps(fingerprint, indent=2),
        encoding="utf-8",
    )


def load_gt_instance_map_cache(
    cache_dir: Path, *, expected: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    meta_path = cache_dir / _FINGERPRINT_NAME
    map_path = cache_dir / _INSTANCE_MAP_NAME
    if not meta_path.is_file() or not map_path.is_file():
        raise FileNotFoundError(f"GT cache missing under {cache_dir}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    try:
        validate_gt_cache_fingerprint(meta, expected)
    except ValueError as exc:
        raise ValueError(f"GT cache fingerprint mismatch: {exc}") from exc
    with np.load(map_path) as data:
        instance_map = np.asarray(data["instance_map"], dtype=np.int32)
    return instance_map, meta


def rasterize_train_gt_instance_map(
    *,
    variant: str,
    grainseg_root: Path,
    staged_manifest: Path,
) -> tuple[np.ndarray, str]:
    pairs = collect_manifest_image_paths(staged_manifest)
    if len(pairs) != 1:
        raise ValueError(
            f"Profile tune GT cache expects one train whole sample, got {len(pairs)}"
        )
    image_path, sample_id = pairs[0]
    height, width = image_dimensions(image_path)
    labels_gpkg = train_labels_gpkg_path(grainseg_root)
    sample = InstanceEvalSample(
        sample_id=sample_id,
        image_path=image_path,
        instance_prediction_set=image_path,
        gt_gpkg=labels_gpkg,
        gt_origin="whole_image",
    )
    gt_map = load_gt_instance_map(sample, image_width=width, image_height=height)
    return np.asarray(gt_map, dtype=np.int32), sample_id


def write_train_gt_cache_for_variant(
    *,
    variant: str,
    work_root: Path,
    grainseg_root: Path,
    repo: Path,
) -> Path:
    staged_manifest = ensure_staged_train_manifest(
        grainseg_root=grainseg_root,
        variant=variant,
        work_root=work_root,
        repo=repo,
    )
    gt_map, sample_id = rasterize_train_gt_instance_map(
        variant=variant,
        grainseg_root=grainseg_root,
        staged_manifest=staged_manifest,
    )
    labels_gpkg = train_labels_gpkg_path(grainseg_root)
    fingerprint = build_gt_fingerprint(
        variant=variant,
        sample_id=sample_id,
        labels_gpkg=labels_gpkg,
    )
    cache_dir = gt_cache_dir(work_root, variant)
    write_gt_instance_map_cache(cache_dir, gt_map, fingerprint=fingerprint)
    return cache_dir


def _parse_args(argv: list[str] | None = None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grainseg-root", type=Path, default=None)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--variants", default=None)
    parser.add_argument("--work-root", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    variants = parse_profile_tune_variants(args.variants)
    grainseg_root, _run_root = default_grainseg_and_run_roots(
        args.grainseg_root, args.run_root
    )
    work_root = args.work_root or (args.output_dir / "_work")
    repo = repo_root()
    for variant in variants:
        cache_dir = write_train_gt_cache_for_variant(
            variant=variant,
            work_root=work_root,
            grainseg_root=grainseg_root,
            repo=repo,
        )
        print(f"Wrote GT cache for {variant} → {cache_dir}")


if __name__ == "__main__":
    main()
