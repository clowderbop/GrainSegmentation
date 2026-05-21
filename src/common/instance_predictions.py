"""Read/write canonical instance prediction artifacts (label-map TIFF, YOLO mask NPZ)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from pycocotools import mask as mask_utils
from skimage.transform import resize as sk_resize

from common.instance_maps import binary_masks_to_instance_map_by_confidence

INSTANCES_SUBDIR = "instances"
MASKS_SUBDIR = "masks"
INSTANCE_MAP_SUFFIX = "_instances.tif"


def instance_map_filename(sample_id: str) -> str:
    return f"{sample_id}{INSTANCE_MAP_SUFFIX}"


def instance_map_path(output_root: Path, sample_id: str) -> Path:
    return output_root / INSTANCES_SUBDIR / instance_map_filename(sample_id)


def yolo_mask_npz_path(output_root: Path, sample_id: str) -> Path:
    return output_root / MASKS_SUBDIR / f"{sample_id}.npz"


def write_instance_map_tiff(path: Path | str, label_map: np.ndarray) -> None:
    arr = np.asarray(label_map)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D instance label map, got shape {arr.shape}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, arr.astype(np.int32, copy=False))


def read_instance_map_tiff(path: Path | str) -> np.ndarray:
    arr = tifffile.imread(path)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D instance label map in {path}, got shape {arr.shape}")
    return arr.astype(np.int32, copy=False)


def resize_mask_nearest(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    if mask.shape == (height, width):
        return mask
    resized = sk_resize(
        mask.astype(np.float32),
        (height, width),
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    )
    return resized


def resize_masks_hw(masks_hw: np.ndarray, height: int, width: int) -> np.ndarray:
    if masks_hw.ndim != 3:
        raise ValueError(f"masks_hw must be (n, H, W), got {masks_hw.shape}")
    if masks_hw.shape[0] == 0:
        return masks_hw.reshape(0, height, width)
    return np.stack(
        [resize_mask_nearest(masks_hw[i], height, width) for i in range(masks_hw.shape[0])],
        axis=0,
    )


def masks_hw_to_binary(masks_hw: np.ndarray) -> np.ndarray:
    if masks_hw.dtype in (np.float32, np.float64):
        return masks_hw > 0.5
    return masks_hw.astype(bool)


def instance_map_from_masks(
    masks_hw: np.ndarray,
    confidences: np.ndarray,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    if masks_hw.ndim != 3:
        raise ValueError(f"masks_hw must be (n, H, W), got {masks_hw.shape}")
    if masks_hw.shape[0] == 0:
        return np.zeros((height, width), dtype=np.int32)
    resized = resize_masks_hw(masks_hw, height, width)
    binary = masks_hw_to_binary(resized)
    return binary_masks_to_instance_map_by_confidence(binary, confidences)


def save_yolo_mask_npz(
    path: Path | str,
    *,
    masks: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    orig_shape: tuple[int, ...] | np.ndarray,
    image_path: str = "",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        masks=np.asarray(masks),
        scores=np.asarray(scores),
        classes=np.asarray(classes),
        orig_shape=np.asarray(orig_shape),
        image_path=np.array(str(image_path)),
    )


def ultralytics_result_mask_arrays(
    result: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (masks, scores, classes) from an Ultralytics Results object."""
    if result.masks is None or len(result.masks) == 0:
        h, w = int(result.orig_shape[0]), int(result.orig_shape[1])
        return (
            np.zeros((0, h, w), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    return (
        result.masks.data.cpu().numpy(),
        result.boxes.conf.cpu().numpy(),
        result.boxes.cls.cpu().numpy(),
    )


def ultralytics_result_to_instance_map(
    result: Any, height: int, width: int
) -> np.ndarray:
    masks, scores, _classes = ultralytics_result_mask_arrays(result)
    if masks.shape[0] == 0:
        return np.zeros((height, width), dtype=np.int32)
    return instance_map_from_masks(masks, scores, height=height, width=width)


def ultralytics_result_prediction_arrays(
    result: Any, *, height: int, width: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (instance_map, masks, scores, classes) from one Results read."""
    masks, scores, classes = ultralytics_result_mask_arrays(result)
    if masks.shape[0] == 0:
        instance_map = np.zeros((height, width), dtype=np.int32)
    else:
        instance_map = instance_map_from_masks(
            masks, scores, height=height, width=width
        )
    return instance_map, masks, scores, classes


def save_ultralytics_result_npz(path: Path | str, result: Any) -> None:
    masks, scores, classes = ultralytics_result_mask_arrays(result)
    image_path = str(result.path) if getattr(result, "path", None) else ""
    save_yolo_mask_npz(
        path,
        masks=masks,
        scores=scores,
        classes=classes,
        orig_shape=np.array(result.orig_shape),
        image_path=image_path,
    )


def yolo_mask_npz_to_coco_dt(
    npz_path: Path | str,
    *,
    image_id: int,
    height: int,
    width: int,
) -> list[dict[str, object]]:
    data = np.load(npz_path)
    masks = np.asarray(data["masks"])
    scores = np.asarray(data["scores"])
    classes = np.asarray(data["classes"])
    if masks.ndim != 3:
        raise ValueError(f"NPZ masks must be (N, H, W), got {masks.shape}")
    n = masks.shape[0]
    detections: list[dict[str, object]] = []
    for i in range(n):
        mask = masks[i]
        if mask.shape != (height, width):
            mask = resize_mask_nearest(mask, height, width)
        binary = masks_hw_to_binary(np.asarray(mask)[None, ...])[0]
        if not binary.any():
            continue
        rle = mask_utils.encode(np.asfortranarray(binary.astype(np.uint8)))
        counts = rle["counts"]
        if isinstance(counts, bytes):
            counts = counts.decode("utf-8")
        segmentation = {"size": [int(height), int(width)], "counts": counts}
        ys, xs = np.where(binary)
        x0, x1 = float(xs.min()), float(xs.max())
        y0, y1 = float(ys.min()), float(ys.max())
        score = float(scores[i]) if i < len(scores) else 0.0
        category_id = int(classes[i]) if i < len(classes) else 0
        detections.append(
            {
                "image_id": int(image_id),
                "category_id": category_id,
                "segmentation": segmentation,
                "bbox": [x0, y0, x1 - x0 + 1.0, y1 - y0 + 1.0],
                "score": score,
            }
        )
    return detections


__all__ = [
    "INSTANCES_SUBDIR",
    "INSTANCE_MAP_SUFFIX",
    "MASKS_SUBDIR",
    "instance_map_filename",
    "instance_map_from_masks",
    "instance_map_path",
    "read_instance_map_tiff",
    "resize_masks_hw",
    "save_ultralytics_result_npz",
    "save_yolo_mask_npz",
    "ultralytics_result_mask_arrays",
    "ultralytics_result_prediction_arrays",
    "ultralytics_result_to_instance_map",
    "write_instance_map_tiff",
    "yolo_mask_npz_path",
    "yolo_mask_npz_to_coco_dt",
]
