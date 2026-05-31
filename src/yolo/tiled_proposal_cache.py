"""Tiled detector proposal cache for YOLO profile tune (ADR 0005)."""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from common.test_inference import TestInferenceRecipe, sahi_overlap_ratio

TILED_PROPOSAL_CACHE_SCHEMA_VERSION = 1

_PROPOSALS_NAME = "proposals.pkl"
_META_NAME = "proposals.meta.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def weights_sha256(weights_path: Path) -> str:
    return file_sha256(weights_path)


def recipe_whole_window_fingerprint(recipe: TestInferenceRecipe) -> str:
    whole = recipe.whole
    overlap = sahi_overlap_ratio(window=whole.window, stride=whole.stride)
    payload = {
        "window": whole.window,
        "stride": whole.stride,
        "overlap_height_ratio": overlap,
        "overlap_width_ratio": overlap,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def proposal_cache_dir(
    variant_work_root: Path, *, conf: float, mask_threshold: float
) -> Path:
    return (
        variant_work_root
        / "tiled_proposals"
        / f"c{conf:g}_t{mask_threshold:g}"
    )


def proposal_cache_record(
    *,
    variant: str,
    weights_sha256: str,
    recipe_window_fingerprint: str,
    conf: float,
    mask_threshold: float,
    sample_id: str,
) -> dict[str, Any]:
    return {
        "schema_version": TILED_PROPOSAL_CACHE_SCHEMA_VERSION,
        "variant": variant,
        "weights_sha256": weights_sha256,
        "recipe_window_fingerprint": recipe_window_fingerprint,
        "conf": conf,
        "mask_threshold": mask_threshold,
        "sample_id": sample_id,
    }


def validate_tiled_proposal_cache(
    meta: dict[str, Any], expected: dict[str, Any]
) -> None:
    if meta.get("schema_version") != TILED_PROPOSAL_CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"Cache schema_version {meta.get('schema_version')!r} != "
            f"{TILED_PROPOSAL_CACHE_SCHEMA_VERSION}"
        )
    for key, expected_value in expected.items():
        if meta.get(key) != expected_value:
            raise ValueError(
                f"Cache mismatch for {key!r}: cached {meta.get(key)!r} != "
                f"current {expected_value!r}"
            )


def write_tiled_proposals(
    cache_dir: Path,
    proposals: list[Any],
    record: dict[str, Any],
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = cache_dir / _PROPOSALS_NAME
    meta_path = cache_dir / _META_NAME
    with pkl_path.open("wb") as handle:
        pickle.dump(proposals, handle, protocol=pickle.HIGHEST_PROTOCOL)
    meta_path.write_text(json.dumps(record, indent=2), encoding="utf-8")


def load_or_write_tiled_proposals(
    cache_dir: Path,
    *,
    expected: dict[str, Any],
    compute_fn: Callable[[], list[Any]],
) -> tuple[list[Any], bool]:
    """Return cached proposals when fingerprint-valid; otherwise compute and write.

    Returns ``(proposals, from_cache)`` where ``from_cache`` is True on reuse.
    """
    pkl_path = cache_dir / _PROPOSALS_NAME
    meta_path = cache_dir / _META_NAME
    if pkl_path.is_file():
        if not meta_path.is_file():
            print(
                "Cached tiled proposals exist but metadata sidecar is missing "
                f"({meta_path}); recomputing.",
                file=sys.stderr,
            )
        else:
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                validate_tiled_proposal_cache(meta, expected)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                print(
                    f"Invalid or incompatible tiled proposal cache ({cache_dir}): "
                    f"{exc}; recomputing.",
                    file=sys.stderr,
                )
            else:
                proposals, _meta = load_tiled_proposals(cache_dir, expected=expected)
                print(f"Reusing cached tiled proposals: {cache_dir}", flush=True)
                return proposals, True

    proposals = compute_fn()
    write_tiled_proposals(cache_dir, proposals, expected)
    return proposals, False


def detector_cache_expected_record(
    *,
    variant: str,
    weights_path: Path,
    conf: float,
    mask_threshold: float,
    sample_id: str,
    recipe: TestInferenceRecipe | None = None,
) -> dict[str, Any]:
    from common.test_inference import load_test_inference_recipe

    resolved_recipe = recipe or load_test_inference_recipe()
    return proposal_cache_record(
        variant=variant,
        weights_sha256=weights_sha256(weights_path),
        recipe_window_fingerprint=recipe_whole_window_fingerprint(resolved_recipe),
        conf=conf,
        mask_threshold=mask_threshold,
        sample_id=sample_id,
    )


def load_tiled_proposals(
    cache_dir: Path, *, expected: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    pkl_path = cache_dir / _PROPOSALS_NAME
    meta_path = cache_dir / _META_NAME
    if not pkl_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"Missing tiled proposal cache under {cache_dir}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    validate_tiled_proposal_cache(meta, expected)
    with pkl_path.open("rb") as handle:
        proposals = pickle.load(handle)
    if not isinstance(proposals, list):
        raise ValueError(f"Cached proposals must be a list: {pkl_path}")
    return proposals, meta
