"""Shared Ultralytics-style dataset YAML helpers (YOLO tooling)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_dataset_config(data_yaml: Path) -> tuple[Path, dict[str, Any]]:
    """Return resolved dataset root plus raw YAML mapping."""
    with data_yaml.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    dataset_root = Path(config.get("path", "."))
    if not dataset_root.is_absolute():
        dataset_root = (data_yaml.parent / dataset_root).resolve()
    return dataset_root, config


def label_map_from_yaml_names(config: dict[str, Any]) -> dict[int, str]:
    names = config.get("names", {})
    if isinstance(names, list):
        return {index: name for index, name in enumerate(names)}
    return {int(k): str(v) for k, v in names.items()}


def resolve_split_dir(dataset_root: Path, split_path: str | Path) -> Path:
    path = Path(split_path)
    if path.is_absolute():
        return path.resolve()
    return (dataset_root / path).resolve()


def default_labels_dir(dataset_root: Path, split_name: str, image_dir: Path) -> Path:
    """Prefer mirroring …/images/… segments into …/labels/… when rooted under dataset_root."""
    try:
        relative_parts = list(image_dir.relative_to(dataset_root).parts)
    except ValueError:
        relative_parts = []
    if "images" in relative_parts:
        relative_parts[relative_parts.index("images")] = "labels"
        return dataset_root.joinpath(*relative_parts)
    return dataset_root / "labels" / split_name
