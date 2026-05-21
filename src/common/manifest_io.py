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


def load_manifest_json(path: Path) -> list[dict[str, Any]]:
    """Load manifest rows from ``{"samples": [...]}``."""
    path = path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f'Manifest {path} must be a JSON object with a "samples" key')
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError(f'Manifest {path} "samples" must be a list')
    for index, row in enumerate(samples):
        if not isinstance(row, dict):
            raise ValueError(f"manifest samples[{index}] must be an object")
    return samples


def collect_manifest_image_paths(
    manifest_path: Path,
) -> list[tuple[Path, str]]:
    manifest_path = manifest_path.resolve()
    manifest_dir = manifest_path.parent
    samples: list[tuple[Path, str]] = []
    for index, row in enumerate(load_manifest_json(manifest_path)):
        image_raw = row.get("image")
        if not image_raw:
            raise ValueError(f'Manifest samples[{index}] requires "image" field')
        image_path = resolve_manifest_path(str(image_raw), manifest_dir)
        sample_id = str(row.get("sample_id") or image_path.stem)
        samples.append((image_path, sample_id))
    if not samples:
        raise ValueError(f"Manifest contains no samples: {manifest_path}")
    return samples
