"""Profile selection ground truth cache — rasterized train GT per variant (ADR 0005)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from common.evaluate_instances import (
    InstanceEvalSample,
    image_dimensions,
    load_gt_instance_map,
)
from common.manifest_io import collect_manifest_image_paths
from common.reporting import count_instances
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


def _log(*parts: object) -> None:
    print(*parts, flush=True)


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
    _log(f"  train image: {image_path} (sample_id={sample_id})")
    height, width = image_dimensions(image_path)
    _log(f"  image size: {width}×{height} px")
    labels_gpkg = train_labels_gpkg_path(grainseg_root)
    _log(
        f"  labels gpkg: {labels_gpkg} "
        f"(sha256={file_sha256(labels_gpkg)[:12]}…)"
    )
    sample = InstanceEvalSample(
        sample_id=sample_id,
        image_path=image_path,
        instance_prediction_set=image_path,
        gt_gpkg=labels_gpkg,
        gt_origin="whole_image",
    )
    _log("  rasterizing whole-image GT instance map …")
    t0 = time.monotonic()
    gt_map = load_gt_instance_map(sample, image_width=width, image_height=height)
    gt_map = np.asarray(gt_map, dtype=np.int32)
    elapsed = time.monotonic() - t0
    _log(
        f"  rasterized in {elapsed:.1f}s — "
        f"{count_instances(gt_map)} instances, dtype={gt_map.dtype}"
    )
    return gt_map, sample_id


def write_train_gt_cache_for_variant(
    *,
    variant: str,
    work_root: Path,
    grainseg_root: Path,
    repo: Path,
) -> Path:
    cache_dir = gt_cache_dir(work_root, variant)
    labels_gpkg = train_labels_gpkg_path(grainseg_root)
    staged_manifest = ensure_staged_train_manifest(
        grainseg_root=grainseg_root,
        variant=variant,
        work_root=work_root,
        repo=repo,
    )
    _log(f"  staged manifest: {staged_manifest}")
    pairs = collect_manifest_image_paths(staged_manifest)
    if len(pairs) != 1:
        raise ValueError(
            f"Profile tune GT cache expects one train whole sample, got {len(pairs)}"
        )
    _sample_image, sample_id = pairs[0]
    fingerprint = build_gt_fingerprint(
        variant=variant,
        sample_id=sample_id,
        labels_gpkg=labels_gpkg,
    )
    try:
        cached_map, _meta = load_gt_instance_map_cache(cache_dir, expected=fingerprint)
    except (FileNotFoundError, ValueError):
        pass
    else:
        _log(
            f"  skip: GT cache already valid → {cache_dir} "
            f"({count_instances(cached_map)} instances)"
        )
        return cache_dir

    gt_map, sample_id = rasterize_train_gt_instance_map(
        variant=variant,
        grainseg_root=grainseg_root,
        staged_manifest=staged_manifest,
    )
    assert sample_id == fingerprint["sample_id"]

    _log(f"  writing cache → {cache_dir}")
    t0 = time.monotonic()
    write_gt_instance_map_cache(cache_dir, gt_map, fingerprint=fingerprint)
    map_path = cache_dir / _INSTANCE_MAP_NAME
    map_bytes = map_path.stat().st_size if map_path.is_file() else 0
    _log(
        f"  wrote {map_path.name} ({map_bytes / 1e6:.1f} MB) "
        f"and {_FINGERPRINT_NAME} in {time.monotonic() - t0:.1f}s"
    )
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
    grainseg_root, run_root = default_grainseg_and_run_roots(
        args.grainseg_root, args.run_root
    )
    work_root = args.work_root or (args.output_dir / "_work")
    repo = repo_root()
    labels_gpkg = train_labels_gpkg_path(grainseg_root)
    _log("Profile selection ground truth cache")
    _log(f"  output_dir={args.output_dir}")
    _log(f"  work_root={work_root}")
    _log(f"  grainseg_root={grainseg_root}")
    _log(f"  run_root={run_root}")
    _log(f"  train_labels_gpkg={labels_gpkg}")
    _log(f"  variants ({len(variants)}): {', '.join(variants)}")
    t_run = time.monotonic()
    for index, variant in enumerate(variants, start=1):
        _log(f"[{index}/{len(variants)}] {variant}")
        t_variant = time.monotonic()
        cache_dir = write_train_gt_cache_for_variant(
            variant=variant,
            work_root=work_root,
            grainseg_root=grainseg_root,
            repo=repo,
        )
        _log(f"  done in {time.monotonic() - t_variant:.1f}s → {cache_dir}")
    _log(f"GT cache complete in {time.monotonic() - t_run:.1f}s")


if __name__ == "__main__":
    main()
