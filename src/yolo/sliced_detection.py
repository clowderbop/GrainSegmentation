"""SAHI whole-image sliced Ultralytics detection (shared by predict and profile tune)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SlicedDetectionResult:
    image: np.ndarray
    object_prediction_list: list[Any]


def perform_ultralytics_inference_preserve_channels(
    detection_model: Any, image: np.ndarray
) -> None:
    """Run Ultralytics on a numpy image slice without BGR channel reordering.

    Mirrors SAHI's Ultralytics backend internals (_original_predictions, mask
    tensors) so multi-channel TIFFs stay channel-ordered. Fragile across
    ultralytics/sahi upgrades; re-check when bumping those dependencies.
    """
    import torch
    from ultralytics.engine.results import Masks

    kwargs = {
        "cfg": detection_model.config_path,
        "verbose": False,
        "conf": detection_model.confidence_threshold,
        "device": detection_model.device,
    }
    if detection_model.image_size is not None:
        kwargs = {"imgsz": detection_model.image_size, **kwargs}

    prediction_result = detection_model.model(np.ascontiguousarray(image), **kwargs)
    if detection_model.has_mask:
        if not prediction_result[0].masks:
            device = getattr(detection_model.model, "device", "cpu")
            prediction_result[0].masks = Masks(
                torch.tensor([], device=device), prediction_result[0].boxes.orig_shape
            )
        prediction_result = [
            (result.boxes.data, result.masks.data) for result in prediction_result
        ]
    else:
        prediction_result = [result.boxes.data for result in prediction_result]

    detection_model._original_predictions = prediction_result
    detection_model._original_shape = image.shape


def iter_whole_slice_predictions(
    image: np.ndarray,
    detection_model: Any,
    *,
    slice_height: int,
    slice_width: int,
    overlap_height_ratio: float,
    overlap_width_ratio: float,
    full_shape: list[int] | None,
) -> Iterator[tuple[int, int, int, int, list[Any]]]:
    """Yield per-slice SAHI predictions (tile-local when ``full_shape`` is None)."""
    from sahi.predict import filter_predictions
    from sahi.slicing import get_slice_bboxes

    height, width = image.shape[:2]
    slice_bboxes = get_slice_bboxes(
        image_height=height,
        image_width=width,
        auto_slice_resolution=False,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
    )

    for tlx, tly, brx, bry in slice_bboxes:
        image_slice = image[tly:bry, tlx:brx]
        perform_ultralytics_inference_preserve_channels(detection_model, image_slice)
        detection_model.convert_original_predictions(
            shift_amount=[tlx, tly],
            full_shape=full_shape,
        )
        predictions = filter_predictions(
            detection_model.object_prediction_list,
            exclude_classes_by_name=None,
            exclude_classes_by_id=None,
        )
        yield tlx, tly, brx, bry, predictions


def run_whole_sliced_detection(
    image: np.ndarray,
    detection_model: Any,
    *,
    slice_height: int,
    slice_width: int,
    overlap_height_ratio: float,
    overlap_width_ratio: float,
) -> list[Any]:
    """Run per-slice detector inference; return shifted **tiled detector proposals** (pre slice-merge)."""
    height, width = image.shape[:2]
    object_prediction_list: list[Any] = []
    for _tlx, _tly, _brx, _bry, predictions in iter_whole_slice_predictions(
        image,
        detection_model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        full_shape=[height, width],
    ):
        for object_prediction in predictions:
            if object_prediction:
                object_prediction_list.append(
                    object_prediction.get_shifted_object_prediction()
                )
    return object_prediction_list


def merge_sliced_object_predictions(
    object_prediction_list: list[Any],
    *,
    postprocess_type: str,
    match_metric: str,
    match_threshold: float,
) -> list[Any]:
    from sahi.predict import POSTPROCESS_NAME_TO_CLASS

    if len(object_prediction_list) <= 1:
        return object_prediction_list
    postprocess_key = postprocess_type.upper()
    if postprocess_key not in POSTPROCESS_NAME_TO_CLASS:
        raise ValueError(
            f"Unsupported postprocess_type {postprocess_type!r}; "
            f"expected one of {sorted(POSTPROCESS_NAME_TO_CLASS)}"
        )
    postprocess = POSTPROCESS_NAME_TO_CLASS[postprocess_key](
        match_threshold=match_threshold,
        match_metric=match_metric,
        class_agnostic=False,
    )
    return postprocess(object_prediction_list)


def get_sliced_prediction_preserve_channels(
    image: np.ndarray,
    detection_model: Any,
    *,
    slice_height: int,
    slice_width: int,
    overlap_height_ratio: float,
    overlap_width_ratio: float,
    postprocess_type: str,
    match_metric: str,
    match_threshold: float,
) -> SlicedDetectionResult:
    object_prediction_list = run_whole_sliced_detection(
        image,
        detection_model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
    )
    merged = merge_sliced_object_predictions(
        object_prediction_list,
        postprocess_type=postprocess_type,
        match_metric=match_metric,
        match_threshold=match_threshold,
    )
    return SlicedDetectionResult(image=image, object_prediction_list=merged)
