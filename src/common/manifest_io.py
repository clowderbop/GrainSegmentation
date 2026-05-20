"""Shared JSON manifest loading for evaluation and YOLO whole-image pipelines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def resolve_manifest_path(raw: str, base_dir: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def manifest_image_field(row: dict[str, Any]) -> str | None:
    raw = row.get("image") or row.get("test_tiff") or row.get("tiff")
    return str(raw) if raw else None


def manifest_gt_gpkg_field(row: dict[str, Any]) -> str | None:
    raw = row.get("gt_gpkg") or row.get("test_gpkg") or row.get("gpkg")
    return str(raw) if raw else None


def load_manifest_json(path: Path) -> list[dict[str, Any]]:
    """Load manifest rows from ``{"samples": [...]}`` or a bare JSON array."""
    path = path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        samples = payload.get("samples")
        if samples is None:
            raise ValueError(f'Manifest {path} object must include "samples" key')
        if not isinstance(samples, list):
            raise ValueError(f'Manifest {path} "samples" must be a list')
        rows = samples
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError(f"Manifest {path} must be a JSON object or array")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"manifest[{index}] must be an object")
    return rows


def collect_manifest_image_paths(
    manifest_path: Path,
) -> list[tuple[Path, str]]:
    manifest_path = manifest_path.resolve()
    manifest_dir = manifest_path.parent
    samples: list[tuple[Path, str]] = []
    for index, row in enumerate(load_manifest_json(manifest_path)):
        image_raw = manifest_image_field(row)
        if not image_raw:
            raise ValueError(
                f"Manifest row {index} requires image, test_tiff, or tiff field"
            )
        image_path = resolve_manifest_path(image_raw, manifest_dir)
        sample_id = str(row.get("sample_id") or image_path.stem)
        samples.append((image_path, sample_id))
    if not samples:
        raise ValueError(f"Manifest contains no samples: {manifest_path}")
    return samples
