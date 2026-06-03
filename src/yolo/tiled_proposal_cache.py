"""Tiled detector proposal cache for YOLO profile tune (ADR 0005, 0007)."""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

from common.file_hash import file_sha256
from common.prediction_set import (
    binary_mask_to_segmentation,
    segmentation_to_binary_mask as crop_segmentation_to_binary_mask,
)
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


def _tight_crop_mask(binary: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Return tight crop and its origin (row, col) within the source plane."""
    plane = np.asarray(binary, dtype=bool)
    if not plane.any():
        raise ValueError("Cannot tight-crop empty mask")
    ys, xs = np.where(plane)
    offset_y = int(ys.min())
    offset_x = int(xs.min())
    crop = plane[offset_y : int(ys.max()) + 1, offset_x : int(xs.max()) + 1]
    return crop, offset_y, offset_x


def _tile_binary_mask(pred: Any, *, mask_threshold: float) -> np.ndarray | None:
    """Decode a tile-local SAHI mask without upsampling to whole-section extent."""
    from common.mask_ops import masks_hw_to_binary

    mask_obj = getattr(pred, "mask", None)
    if mask_obj is None:
        return None
    float_mask = getattr(mask_obj, "float_mask", None)
    if float_mask is not None:
        return masks_hw_to_binary(
            np.asarray(float_mask, dtype=np.float32)[None, ...],
            threshold=mask_threshold,
        )[0]
    bool_mask = getattr(mask_obj, "bool_mask", None)
    if bool_mask is None:
        return None
    return np.asarray(bool_mask, dtype=bool)


def tiled_proposal_record_from_tile_mask(
    mask: np.ndarray, *, score: float, offset_y: int, offset_x: int
) -> TiledProposalRecord:
    """Build one v2 record from a tight tile-local crop and whole-image offsets."""
    binary = np.asarray(mask, dtype=bool)
    if not binary.any():
        raise ValueError("Cannot encode empty mask as tiled proposal record")
    crop_h, crop_w = binary.shape
    return TiledProposalRecord(
        score=float(score),
        bbox=[
            float(offset_x),
            float(offset_y),
            float(offset_x + crop_w),
            float(offset_y + crop_h),
        ],
        segmentation=binary_mask_to_segmentation(binary, height=crop_h, width=crop_w),
        offset_y=int(offset_y),
        offset_x=int(offset_x),
    )


def tiled_proposal_record_from_binary_mask(
    mask: np.ndarray, *, score: float
) -> TiledProposalRecord:
    """Build v2 record by tight-cropping any boolean plane (test/convenience helper)."""
    crop, offset_y, offset_x = _tight_crop_mask(mask)
    return tiled_proposal_record_from_tile_mask(
        crop, score=score, offset_y=offset_y, offset_x=offset_x
    )


def tiled_proposal_records_from_tile_predictions(
    predictions: Sequence[Any],
    *,
    tlx: int,
    tly: int,
    slice_height: int,
    slice_width: int,
    mask_threshold: float,
) -> list[TiledProposalRecord]:
    """Encode tile-local SAHI predictions to v2 records (crop-local RLE + offsets)."""
    records: list[TiledProposalRecord] = []
    for pred in predictions:
        if not pred:
            continue
        binary = _tile_binary_mask(pred, mask_threshold=mask_threshold)
        if binary is None or not binary.any():
            continue
        mask_h, mask_w = binary.shape
        if mask_h > slice_height or mask_w > slice_width:
            raise ValueError(
                f"tile mask shape ({mask_h}, {mask_w}) exceeds slice "
                f"({slice_height}, {slice_width})"
            )
        crop, crop_y0, crop_x0 = _tight_crop_mask(binary)
        records.append(
            tiled_proposal_record_from_tile_mask(
                crop,
                score=_sahi_object_score(pred),
                offset_y=tly + crop_y0,
                offset_x=tlx + crop_x0,
            )
        )
    return records


def collect_tiled_detector_proposals(
    image: np.ndarray,
    detection_model: Any,
    *,
    slice_height: int,
    slice_width: int,
    overlap_height_ratio: float,
    overlap_width_ratio: float,
    mask_threshold: float,
) -> list[TiledProposalRecord]:
    """Profile-tune detector path: slice loop + tile-local v2 encode (ADR 0005)."""
    from yolo.sliced_detection import iter_whole_slice_predictions

    records: list[TiledProposalRecord] = []
    slice_count = 0
    encode_s = 0.0
    for tlx, tly, brx, bry, predictions in iter_whole_slice_predictions(
        image,
        detection_model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        full_shape=None,
    ):
        slice_count += 1
        tile_h, tile_w = bry - tly, brx - tlx
        encode_start = time.perf_counter()
        records.extend(
            tiled_proposal_records_from_tile_predictions(
                predictions,
                tlx=tlx,
                tly=tly,
                slice_height=tile_h,
                slice_width=tile_w,
                mask_threshold=mask_threshold,
            )
        )
        encode_s += time.perf_counter() - encode_start
    print(
        f"Detector proposals: slices={slice_count} proposals={len(records)} "
        f"encode_s={encode_s:.1f}",
        file=sys.stderr,
        flush=True,
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


def _full_binary_mask_from_tiled_proposal_record(
    record: TiledProposalRecord, *, height: int, width: int
) -> np.ndarray:
    """Reconstruct one full-section mask from crop-local RLE + offsets.

    Not for profile selection scoring load (ADR 0005): materializing many planes
    OOMs candidate jobs. Use ``sahi_predictions_from_tiled_proposal_records`` instead.
    """
    crop = crop_segmentation_to_binary_mask(record["segmentation"])
    plane = np.zeros((height, width), dtype=bool)
    offset_y = record["offset_y"]
    offset_x = record["offset_x"]
    crop_h, crop_w = crop.shape
    plane[offset_y : offset_y + crop_h, offset_x : offset_x + crop_w] = crop
    return plane


@dataclass(frozen=True)
class _AdaptedCategory:
    id: int
    name: str = "grain"


@dataclass(frozen=True)
class _AdaptedScore:
    value: float


@dataclass(frozen=True)
class _AdaptedMask:
    bool_mask: np.ndarray
    segmentation: dict[str, Any]
    full_shape: tuple[int, int]
    shift_amount: tuple[int, int]


@dataclass(frozen=True)
class _AdaptedBbox:
    minx: float
    miny: float
    maxx: float
    maxy: float
    shift_amount: tuple[int, int] = (0, 0)

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
    """Adapt v2 records to SAHI-shaped predictions with crop-local masks (ADR 0005)."""
    from sahi.annotation import Mask as SahiMask

    section_shape = (int(height), int(width))
    predictions: list[Any] = []
    for record in records:
        crop = crop_segmentation_to_binary_mask(record["segmentation"])
        x0, y0, x1, y1 = record["bbox"]
        offset_x = int(record["offset_x"])
        offset_y = int(record["offset_y"])
        merge_mask = SahiMask.from_bool_mask(
            crop,
            full_shape=list(section_shape),
            shift_amount=[offset_x, offset_y],
        )
        predictions.append(
            _AdaptedSahiPrediction(
                mask=_AdaptedMask(
                    bool_mask=crop,
                    segmentation=merge_mask.segmentation,
                    full_shape=section_shape,
                    shift_amount=(offset_x, offset_y),
                ),
                score=_AdaptedScore(value=record["score"]),
                category=_AdaptedCategory(id=0),
                bbox=_AdaptedBbox(
                    minx=float(x0),
                    miny=float(y0),
                    maxx=float(x1),
                    maxy=float(y1),
                    shift_amount=(offset_x, offset_y),
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
