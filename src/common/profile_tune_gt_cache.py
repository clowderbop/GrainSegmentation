"""Profile selection ground truth cache — train merged instance view (ADR 0005)."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from common.profile_tune_paths import profile_tune_cache_root

import numpy as np

from common.evaluate_instances import image_dimensions
from common.file_hash import file_sha256
from common.gpkg_instance_map import gpkg_to_merged_instance_map
from common.reporting import count_instances
from common.variants import default_grainseg_root, get_variant

GT_CACHE_SCHEMA_VERSION = 2
_INSTANCE_MAP_NAME = "instance_map.npz"
_FINGERPRINT_NAME = "fingerprint.json"
_TRAIN_SAMPLE_ID = "train"


def _log(*parts: object) -> None:
    print(*parts, flush=True)


def gt_cache_dir(work_root: Path, *, sample_id: str = _TRAIN_SAMPLE_ID) -> Path:
    return work_root / "gt_cache" / sample_id


def train_labels_gpkg_path(grainseg_root: Path) -> Path:
    return grainseg_root / get_variant("PPL").paths.train_labels_gpkg


def train_anchor_image_path(grainseg_root: Path) -> Path:
    return grainseg_root / get_variant("PPL").paths.train_mosaic_stacked


def build_gt_fingerprint(
    *,
    sample_id: str,
    labels_gpkg: Path,
    width: int,
    height: int,
) -> dict[str, Any]:
    if not labels_gpkg.is_file():
        raise FileNotFoundError(f"Train labels GeoPackage not found: {labels_gpkg}")
    return {
        "schema_version": GT_CACHE_SCHEMA_VERSION,
        "sample_id": sample_id,
        "width": int(width),
        "height": int(height),
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


def copy_gpkg_to_tmpdir(gpkg_path: Path, *, tmp_dir: Path | None = None) -> tuple[Path, float]:
    """Copy GPKG to a local temp directory; return (local path, seconds)."""
    tmp_root = tmp_dir or Path(os.environ.get("TMPDIR", "/tmp"))
    tmp_root.mkdir(parents=True, exist_ok=True)
    local = tmp_root / f"profile_tune_gt_{gpkg_path.name}"
    t0 = time.monotonic()
    shutil.copy2(gpkg_path, local)
    return local, time.monotonic() - t0


def rasterize_train_gt_instance_map(
    gpkg_path: Path,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    _log(f"  rasterizing GPKG → merged instance view ({width}×{height}) …")
    t0 = time.monotonic()
    gt_map = gpkg_to_merged_instance_map(gpkg_path, height=height, width=width)
    gt_map = np.asarray(gt_map, dtype=np.int32)
    elapsed = time.monotonic() - t0
    _log(
        f"  rasterized in {elapsed:.1f}s — "
        f"{count_instances(gt_map)} instances, dtype={gt_map.dtype}"
    )
    return gt_map


def write_train_gt_cache(
    *,
    work_root: Path,
    grainseg_root: Path,
    sample_id: str = _TRAIN_SAMPLE_ID,
    tmp_dir: Path | None = None,
) -> Path:
    cache_dir = gt_cache_dir(work_root, sample_id=sample_id)
    labels_gpkg = train_labels_gpkg_path(grainseg_root)
    anchor_image = train_anchor_image_path(grainseg_root)
    if not anchor_image.is_file():
        raise FileNotFoundError(f"Train anchor image not found: {anchor_image}")
    height, width = image_dimensions(anchor_image)
    _log(f"  train anchor: {anchor_image} (sample_id={sample_id})")
    _log(f"  image size: {width}×{height} px")
    _log(
        f"  labels gpkg: {labels_gpkg} "
        f"(sha256={file_sha256(labels_gpkg)[:12]}…)"
    )
    fingerprint = build_gt_fingerprint(
        sample_id=sample_id,
        labels_gpkg=labels_gpkg,
        width=width,
        height=height,
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

    _log("  copying GPKG to local temp …")
    local_gpkg, copy_s = copy_gpkg_to_tmpdir(labels_gpkg, tmp_dir=tmp_dir)
    _log(f"  copied GPKG in {copy_s:.1f}s → {local_gpkg}")

    t_raster = time.monotonic()
    gt_map = rasterize_train_gt_instance_map(local_gpkg, height=height, width=width)
    raster_s = time.monotonic() - t_raster

    _log(f"  writing cache → {cache_dir}")
    t_write = time.monotonic()
    write_gt_instance_map_cache(cache_dir, gt_map, fingerprint=fingerprint)
    map_path = cache_dir / _INSTANCE_MAP_NAME
    map_bytes = map_path.stat().st_size if map_path.is_file() else 0
    write_s = time.monotonic() - t_write
    _log(
        f"  wrote {map_path.name} ({map_bytes / 1e6:.1f} MB) "
        f"and {_FINGERPRINT_NAME} in {write_s:.1f}s"
    )
    _log(
        f"  phase timings: copy={copy_s:.1f}s rasterize={raster_s:.1f}s "
        f"write={write_s:.1f}s"
    )
    return cache_dir


def _parse_args(argv: list[str] | None = None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grainseg-root", type=Path, default=None)
    parser.add_argument("--work-root", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    grainseg_root = args.grainseg_root or default_grainseg_root()
    work_root = args.work_root or profile_tune_cache_root(args.output_dir)
    labels_gpkg = train_labels_gpkg_path(grainseg_root)
    _log("Profile selection ground truth cache")
    _log(f"  output_dir={args.output_dir}")
    _log(f"  work_root={work_root}")
    _log(f"  grainseg_root={grainseg_root}")
    _log(f"  train_labels_gpkg={labels_gpkg}")
    t_run = time.monotonic()
    cache_dir = write_train_gt_cache(
        work_root=work_root,
        grainseg_root=grainseg_root,
    )
    _log(f"GT cache complete in {time.monotonic() - t_run:.1f}s → {cache_dir}")


if __name__ == "__main__":
    main()
