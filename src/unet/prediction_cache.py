"""Semantic prediction TIFF cache validation and I/O."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

PREDICTION_CACHE_SCHEMA_VERSION = 2


def prediction_tiff_path(semantic_dir: Path, sample_id: str) -> Path:
    return semantic_dir / f"{sample_id}_pred.tif"


def prediction_meta_path(semantic_dir: Path, sample_id: str) -> Path:
    return semantic_dir / f"{sample_id}_pred.meta.json"


def prediction_cache_record(args: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": PREDICTION_CACHE_SCHEMA_VERSION,
        "model_path": str(Path(args.model_path).resolve()),
        "patch_size": args.patch_size,
        "stride": args.stride,
        "batch_size": args.batch_size,
        "num_inputs": args.num_inputs,
        "image_suffixes": list(args.image_suffixes),
        "unit": getattr(args, "unit", None),
        "variant": getattr(args, "variant", None),
    }
    return record


def validate_prediction_cache(meta: dict[str, Any], args: Any) -> None:
    expected = prediction_cache_record(args)
    if meta.get("schema_version") != PREDICTION_CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"Cache schema_version {meta.get('schema_version')!r} != "
            f"{PREDICTION_CACHE_SCHEMA_VERSION}"
        )
    for key, expected_value in expected.items():
        if meta.get(key) != expected_value:
            raise ValueError(
                f"Cache mismatch for {key!r}: cached {meta.get(key)!r} != "
                f"current {expected_value!r}"
            )


def load_cached_prediction_tiff(path: Path, expected_hw: tuple[int, int]) -> np.ndarray:
    arr = tifffile.imread(path)
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[2] == 1:
            arr = arr[:, :, 0]
        else:
            raise ValueError(f"Cached prediction must be single-channel TIFF: {path}")
    if arr.ndim != 2:
        raise ValueError(f"Cached prediction must be 2D: {path}")
    if arr.shape != expected_hw:
        raise ValueError(
            f"Cached prediction shape {arr.shape} does not match image shape "
            f"{expected_hw}: {path}"
        )
    return arr.astype(np.int32)


def write_prediction_cache(
    semantic_dir: Path,
    sample_id: str,
    pred_classes: np.ndarray,
    args: Any,
) -> None:
    cache_path = prediction_tiff_path(semantic_dir, sample_id)
    meta_path = prediction_meta_path(semantic_dir, sample_id)
    semantic_dir.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        cache_path,
        pred_classes.astype(np.uint8),
        compression="deflate",
    )
    with meta_path.open("w", encoding="utf-8") as meta_file:
        json.dump(prediction_cache_record(args), meta_file, indent=2, sort_keys=True)
        meta_file.write("\n")


def load_or_use_cached_prediction(
    *,
    args: Any,
    sample_id: str,
    semantic_dir: Path,
    expected_hw: tuple[int, int],
    predict_fn: Any,
) -> np.ndarray:
    cache_path = prediction_tiff_path(semantic_dir, sample_id)
    meta_path = prediction_meta_path(semantic_dir, sample_id)
    if cache_path.is_file():
        if not meta_path.is_file():
            print(
                "Cached prediction TIFF exists but metadata sidecar is missing "
                f"({meta_path}); recomputing.",
                file=sys.stderr,
            )
        else:
            try:
                with meta_path.open(encoding="utf-8") as meta_file:
                    meta = json.load(meta_file)
                validate_prediction_cache(meta, args)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                print(
                    f"Invalid or incompatible prediction cache metadata "
                    f"({meta_path}): {exc}; recomputing.",
                    file=sys.stderr,
                )
            else:
                print(f"Reusing cached prediction: {cache_path}")
                return load_cached_prediction_tiff(cache_path, expected_hw)

    pred_classes = predict_fn()
    write_prediction_cache(semantic_dir, sample_id, pred_classes, args)
    return pred_classes
