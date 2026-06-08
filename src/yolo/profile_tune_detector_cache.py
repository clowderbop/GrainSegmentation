"""Shared detector proposal cache read/write for profile selection (ADR 0005)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sahi import AutoDetectionModel

from common.test_inference import TestInferenceRecipe, load_test_inference_recipe
from yolo.predict import device_for_sahi, load_image_for_yolo
from yolo.profile_tune_cache_stage import (
    resolve_train_whole_image_path,
    stage_detector_train_image,
)
from yolo.profile_tune_work import sahi_window_kwargs, weights_path
from yolo.tiled_proposal_cache import (
    collect_tiled_detector_proposals,
    detector_cache_expected_record,
    load_or_write_tiled_proposals,
    load_tiled_proposals,
    proposal_cache_dir,
    proposal_cache_record,
    recipe_whole_window_fingerprint,
    weights_sha256,
)
from yolo.train import _parse_device


@dataclass(frozen=True)
class PreparedDetectorVariant:
    weights: Path
    image: np.ndarray
    sample_id: str
    recipe: TestInferenceRecipe
    window_kwargs: dict[str, int | float]
    sahi_device: str


def resolve_detector_train_image(
    *,
    grainseg_root: Path,
    variant: str,
    local_train_image: Path | None,
    train_image_staging_dir: Path | None,
) -> tuple[Path, str, str | None]:
    if local_train_image is not None and train_image_staging_dir is not None:
        raise ValueError(
            "Specify at most one of --local-train-image or --train-image-staging-dir"
        )
    if local_train_image is not None:
        _, sample_id = resolve_train_whole_image_path(
            grainseg_root=grainseg_root, variant=variant
        )
        return local_train_image.resolve(), sample_id, None
    if train_image_staging_dir is not None:
        staged = stage_detector_train_image(
            grainseg_root=grainseg_root,
            variant=variant,
            tmp_dir=train_image_staging_dir,
        )
        note = (
            f"Train whole image staged locally → {staged.image_path} "
            f"(copy_s={staged.copy_s:.1f})"
        )
        return staged.image_path, staged.sample_id, note
    image_path, sample_id = resolve_train_whole_image_path(
        grainseg_root=grainseg_root, variant=variant
    )
    return image_path, sample_id, None


def format_scratch_cache_label(cache_dir: Path, scratch_cache: Path) -> str:
    try:
        rel = cache_dir.relative_to(scratch_cache)
        return f"scratch .cache/{rel}"
    except ValueError:
        return str(cache_dir)


def prepare_detector_variant(
    *,
    variant: str,
    grainseg_root: Path,
    run_root: Path,
    device: str,
    local_train_image: Path | None = None,
    train_image_staging_dir: Path | None = None,
) -> tuple[PreparedDetectorVariant, str | None]:
    """Resolve weights, stage/load train whole TIFF, and build shared inference context."""
    weights = weights_path(grainseg_root, variant, run_root)
    if not weights.is_file():
        raise FileNotFoundError(f"Missing YOLO weights for {variant}: {weights}")

    image_path, sample_id, staging_note = resolve_detector_train_image(
        grainseg_root=grainseg_root,
        variant=variant,
        local_train_image=local_train_image,
        train_image_staging_dir=train_image_staging_dir,
    )
    image = load_image_for_yolo(image_path)
    recipe = load_test_inference_recipe()
    return (
        PreparedDetectorVariant(
            weights=weights,
            image=image,
            sample_id=sample_id,
            recipe=recipe,
            window_kwargs=sahi_window_kwargs(),
            sahi_device=device_for_sahi(_parse_device(device)),
        ),
        staging_note,
    )


def detector_cache_dir(work_root: Path, variant: str, *, conf: float) -> Path:
    return proposal_cache_dir(work_root / variant, conf=conf)


def expected_detector_cache(
    prepared: PreparedDetectorVariant,
    *,
    variant: str,
    conf: float,
    mask_threshold: float,
) -> dict[str, Any]:
    return detector_cache_expected_record(
        variant=variant,
        weights_path=prepared.weights,
        conf=conf,
        mask_threshold=mask_threshold,
        sample_id=prepared.sample_id,
        recipe=prepared.recipe,
    )


def skip_valid_detector_cache(
    cache_dir: Path,
    expected: dict[str, Any],
    *,
    conf: float,
    mask_threshold: float,
    log_skip: bool,
) -> bool:
    try:
        load_tiled_proposals(cache_dir, expected=expected)
    except (FileNotFoundError, ValueError):
        return False
    if log_skip:
        print(
            f"Skipping detector key conf={conf:g} mask_threshold={mask_threshold:g} "
            f"(cache valid → {cache_dir})",
            flush=True,
        )
    return True


def _proposal_cache_record(
    prepared: PreparedDetectorVariant,
    *,
    variant: str,
    conf: float,
    mask_threshold: float,
) -> dict[str, Any]:
    height, width = int(prepared.image.shape[0]), int(prepared.image.shape[1])
    return proposal_cache_record(
        variant=variant,
        weights_sha256=weights_sha256(prepared.weights),
        recipe_window_fingerprint=recipe_whole_window_fingerprint(prepared.recipe),
        conf=conf,
        mask_threshold=mask_threshold,
        sample_id=prepared.sample_id,
        height=height,
        width=width,
    )


def _ensure_detection_model(
    prepared: PreparedDetectorVariant,
    *,
    conf: float,
    mask_threshold: float,
    detection_model: Any | None,
) -> Any:
    if detection_model is None:
        return AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=str(prepared.weights.resolve()),
            confidence_threshold=conf,
            mask_threshold=mask_threshold,
            device=prepared.sahi_device,
            image_size=prepared.recipe.whole.window,
        )
    detection_model.confidence_threshold = conf
    detection_model.mask_threshold = mask_threshold
    return detection_model


def write_detector_key_proposals(
    prepared: PreparedDetectorVariant,
    *,
    variant: str,
    conf: float,
    mask_threshold: float,
    work_root: Path,
    detection_model: Any | None = None,
) -> tuple[Path, Any]:
    """Write one (conf, mask_threshold) proposal cache; reuse detection_model when provided."""
    cache_dir = detector_cache_dir(work_root, variant, conf=conf)
    expected = expected_detector_cache(
        prepared, variant=variant, conf=conf, mask_threshold=mask_threshold
    )
    record = _proposal_cache_record(
        prepared, variant=variant, conf=conf, mask_threshold=mask_threshold
    )
    model = _ensure_detection_model(
        prepared, conf=conf, mask_threshold=mask_threshold, detection_model=detection_model
    )
    window = prepared.window_kwargs

    def compute_proposals() -> list:
        return collect_tiled_detector_proposals(
            prepared.image,
            model,
            slice_height=int(window["slice_height"]),
            slice_width=int(window["slice_width"]),
            overlap_height_ratio=float(window["overlap_height_ratio"]),
            overlap_width_ratio=float(window["overlap_width_ratio"]),
            mask_threshold=mask_threshold,
        )

    load_or_write_tiled_proposals(
        cache_dir,
        expected=expected,
        meta=record,
        compute_fn=compute_proposals,
    )
    return cache_dir, model


def write_detector_key_proposals_if_needed(
    prepared: PreparedDetectorVariant,
    *,
    variant: str,
    conf: float,
    mask_threshold: float,
    work_root: Path,
    detection_model: Any | None = None,
    log_skip: bool,
) -> tuple[Path, Any | None, bool]:
    """Return (cache_dir, detection_model, wrote). Model is set only when inference ran."""
    cache_dir = detector_cache_dir(work_root, variant, conf=conf)
    expected = expected_detector_cache(
        prepared, variant=variant, conf=conf, mask_threshold=mask_threshold
    )
    if skip_valid_detector_cache(
        cache_dir,
        expected,
        conf=conf,
        mask_threshold=mask_threshold,
        log_skip=log_skip,
    ):
        return cache_dir, detection_model, False
    cache_dir, model = write_detector_key_proposals(
        prepared,
        variant=variant,
        conf=conf,
        mask_threshold=mask_threshold,
        work_root=work_root,
        detection_model=detection_model,
    )
    return cache_dir, model, True
