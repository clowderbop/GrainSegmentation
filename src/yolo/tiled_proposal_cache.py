"""Tiled detector proposal cache for YOLO profile tune (ADR 0005, 0007)."""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

from common.prediction_set import binary_mask_to_segmentation, segmentation_to_binary_mask
from common.test_inference import TestInferenceRecipe, sahi_overlap_ratio

TILED_PROPOSAL_CACHE_SCHEMA_VERSION = 2

_PROPOSALS_NAME = "proposals.pkl"
_META_NAME = "proposals.meta.json"


class TiledProposalRecord(TypedDict):
    score: float
    bbox: list[float]
    segmentation: dict[str, Any]
    offset_y: int
    offset_x: int


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
    height: int,
    width: int,
) -> dict[str, Any]:
    return {
        **proposal_cache_fingerprint(
            variant=variant,
            weights_sha256=weights_sha256,
            recipe_window_fingerprint=recipe_window_fingerprint,
            conf=conf,
            mask_threshold=mask_threshold,
            sample_id=sample_id,
        ),
        "height": int(height),
        "width": int(width),
    }


def validate_tiled_proposal_cache(
    meta: dict[str, Any], expected: dict[str, Any]
) -> None:
    schema_version = meta.get("schema_version")
    if schema_version == 1:
        raise ValueError(
            "Unsupported tiled proposal cache schema_version 1; "
            "re-run detector jobs to produce schema_version 2"
        )
    if schema_version != TILED_PROPOSAL_CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"Cache schema_version {schema_version!r} != "
            f"{TILED_PROPOSAL_CACHE_SCHEMA_VERSION}"
        )
    for key, expected_value in expected.items():
        if meta.get(key) != expected_value:
            raise ValueError(
                f"Cache mismatch for {key!r}: cached {meta.get(key)!r} != "
                f"current {expected_value!r}"
            )


def _sahi_object_score(pred: Any) -> float:
    score_obj = getattr(pred, "score", None)
    if score_obj is None:
        raise ValueError("SAHI object prediction missing score")
    value = getattr(score_obj, "value", None)
    if value is None:
        raise ValueError("SAHI object prediction score has no value")
    return float(value)


def _sahi_binary_mask(pred: Any, *, height: int, width: int, mask_threshold: float) -> np.ndarray | None:
    from common.mask_ops import masks_hw_to_binary

    mask_obj = getattr(pred, "mask", None)
    if mask_obj is None:
        return None
    float_mask = getattr(mask_obj, "float_mask", None)
    if float_mask is not None:
        binary = masks_hw_to_binary(
            np.asarray(float_mask, dtype=np.float32)[None, ...],
            threshold=mask_threshold,
        )[0]
    else:
        mask = getattr(mask_obj, "bool_mask", None)
        if mask is None:
            return None
        binary = np.asarray(mask, dtype=bool)
    if binary.shape != (height, width):
        from common.mask_ops import resize_mask_nearest

        binary = resize_mask_nearest(binary.astype(np.uint8), height, width).astype(bool)
    return binary


def tiled_proposal_record_from_binary_mask(
    mask: np.ndarray, *, score: float
) -> TiledProposalRecord:
    """Build one v2 record with crop-local RLE from a full-image boolean mask."""
    binary = np.asarray(mask, dtype=bool)
    if not binary.any():
        raise ValueError("Cannot encode empty mask as tiled proposal record")
    ys, xs = np.where(binary)
    offset_y = int(ys.min())
    offset_x = int(xs.min())
    crop = binary[offset_y : int(ys.max()) + 1, offset_x : int(xs.max()) + 1]
    crop_h, crop_w = crop.shape
    return TiledProposalRecord(
        score=float(score),
        bbox=[
            float(offset_x),
            float(offset_y),
            float(offset_x + crop_w),
            float(offset_y + crop_h),
        ],
        segmentation=binary_mask_to_segmentation(crop, height=crop_h, width=crop_w),
        offset_y=offset_y,
        offset_x=offset_x,
    )


def tiled_proposal_records_from_sahi_predictions(
    predictions: Sequence[Any],
    *,
    height: int,
    width: int,
    mask_threshold: float,
) -> list[TiledProposalRecord]:
    """Convert shifted SAHI object predictions to v2 neutral records (crop-local RLE)."""
    records: list[TiledProposalRecord] = []
    for pred in predictions:
        binary = _sahi_binary_mask(
            pred, height=height, width=width, mask_threshold=mask_threshold
        )
        if binary is None or not binary.any():
            continue
        records.append(
            tiled_proposal_record_from_binary_mask(binary, score=_sahi_object_score(pred))
        )
    return records


def validate_tiled_proposal_records(
    records: list[Any], *, height: int, width: int
) -> list[TiledProposalRecord]:
    validated: list[TiledProposalRecord] = []
    for index, raw in enumerate(records):
        if not isinstance(raw, dict):
            raise ValueError(f"Proposal record {index} must be a dict")
        try:
            score = float(raw["score"])
            bbox = raw["bbox"]
            segmentation = raw["segmentation"]
            offset_y = int(raw["offset_y"])
            offset_x = int(raw["offset_x"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Proposal record {index} missing required fields") from exc
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"Proposal record {index} bbox must be [x0, y0, x1, y1]")
        crop_h, crop_w = segmentation["size"]
        if int(crop_h) <= 0 or int(crop_w) <= 0:
            raise ValueError(f"Proposal record {index} has invalid crop size")
        if offset_y < 0 or offset_x < 0:
            raise ValueError(f"Proposal record {index} has negative offset")
        if offset_y + int(crop_h) > height or offset_x + int(crop_w) > width:
            raise ValueError(
                f"Proposal record {index} crop exceeds cache extent "
                f"{height}x{width}"
            )
        validated.append(
            TiledProposalRecord(
                score=score,
                bbox=[float(v) for v in bbox],
                segmentation=segmentation,
                offset_y=offset_y,
                offset_x=offset_x,
            )
        )
    return validated


def full_binary_mask_from_tiled_proposal_record(
    record: TiledProposalRecord, *, height: int, width: int
) -> np.ndarray:
    """Reconstruct a full-section boolean mask from crop-local RLE + offsets."""
    crop = segmentation_to_binary_mask(record["segmentation"])
    plane = np.zeros((height, width), dtype=bool)
    offset_y = record["offset_y"]
    offset_x = record["offset_x"]
    crop_h, crop_w = crop.shape
    plane[offset_y : offset_y + crop_h, offset_x : offset_x + crop_w] = crop
    return plane


@dataclass(frozen=True)
class _AdaptedCategory:
    id: int


@dataclass(frozen=True)
class _AdaptedScore:
    value: float


@dataclass(frozen=True)
class _AdaptedMask:
    bool_mask: np.ndarray


@dataclass(frozen=True)
class _AdaptedBbox:
    minx: float
    miny: float
    maxx: float
    maxy: float

    def to_xyxy(self) -> list[float]:
        return [self.minx, self.miny, self.maxx, self.maxy]


@dataclass(frozen=True)
class _AdaptedSahiPrediction:
    mask: _AdaptedMask
    score: _AdaptedScore
    category: _AdaptedCategory
    bbox: _AdaptedBbox

    def tolist(self) -> _AdaptedSahiPrediction:
        return self


def sahi_predictions_from_tiled_proposal_records(
    records: Sequence[TiledProposalRecord],
    *,
    height: int,
    width: int,
) -> list[Any]:
    """Adapt v2 records to SAHI-shaped predictions for slice-merge (until ADR 0007 scoring)."""
    predictions: list[Any] = []
    for record in records:
        binary = full_binary_mask_from_tiled_proposal_record(
            record, height=height, width=width
        )
        x0, y0, x1, y1 = record["bbox"]
        predictions.append(
            _AdaptedSahiPrediction(
                mask=_AdaptedMask(bool_mask=binary),
                score=_AdaptedScore(value=record["score"]),
                category=_AdaptedCategory(id=0),
                bbox=_AdaptedBbox(
                    minx=float(x0),
                    miny=float(y0),
                    maxx=float(x1),
                    maxy=float(y1),
                ),
            )
        )
    return predictions


def write_tiled_proposals(
    cache_dir: Path,
    proposals: list[TiledProposalRecord],
    record: dict[str, Any],
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    height = int(record["height"])
    width = int(record["width"])
    validated = validate_tiled_proposal_records(proposals, height=height, width=width)
    pkl_path = cache_dir / _PROPOSALS_NAME
    meta_path = cache_dir / _META_NAME
    with pkl_path.open("wb") as handle:
        pickle.dump(validated, handle, protocol=pickle.HIGHEST_PROTOCOL)
    meta_path.write_text(json.dumps(record, indent=2), encoding="utf-8")


def load_or_write_tiled_proposals(
    cache_dir: Path,
    *,
    expected: dict[str, Any],
    compute_fn: Callable[[], list[TiledProposalRecord]],
    meta: dict[str, Any] | None = None,
) -> tuple[list[TiledProposalRecord], bool]:
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
    write_tiled_proposals(cache_dir, proposals, meta or expected)
    return proposals, False


def proposal_cache_fingerprint(
    *,
    variant: str,
    weights_sha256: str,
    recipe_window_fingerprint: str,
    conf: float,
    mask_threshold: float,
    sample_id: str,
) -> dict[str, Any]:
    """Fields validated on cache hit (excludes full-section height/width)."""
    return {
        "schema_version": TILED_PROPOSAL_CACHE_SCHEMA_VERSION,
        "variant": variant,
        "weights_sha256": weights_sha256,
        "recipe_window_fingerprint": recipe_window_fingerprint,
        "conf": conf,
        "mask_threshold": mask_threshold,
        "sample_id": sample_id,
    }


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
    return proposal_cache_fingerprint(
        variant=variant,
        weights_sha256=weights_sha256(weights_path),
        recipe_window_fingerprint=recipe_whole_window_fingerprint(resolved_recipe),
        conf=conf,
        mask_threshold=mask_threshold,
        sample_id=sample_id,
    )


def load_tiled_proposals(
    cache_dir: Path, *, expected: dict[str, Any]
) -> tuple[list[TiledProposalRecord], dict[str, Any]]:
    pkl_path = cache_dir / _PROPOSALS_NAME
    meta_path = cache_dir / _META_NAME
    if not pkl_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"Missing tiled proposal cache under {cache_dir}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    validate_tiled_proposal_cache(meta, expected)
    height = int(meta["height"])
    width = int(meta["width"])
    with pkl_path.open("rb") as handle:
        raw = pickle.load(handle)
    if not isinstance(raw, list):
        raise ValueError(f"Cached proposals must be a list: {pkl_path}")
    return validate_tiled_proposal_records(raw, height=height, width=width), meta
