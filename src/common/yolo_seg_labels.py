"""Read, write, and rasterize YOLO segmentation label files."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class YoloSegGtRow:
    class_id: int
    points: np.ndarray

    def __iter__(self) -> Iterator[object]:
        yield self.class_id
        yield self.points


@dataclass(frozen=True)
class YoloSegPredRow:
    class_id: int
    points: np.ndarray
    confidence: float

    def __iter__(self) -> Iterator[object]:
        yield self.class_id
        yield self.points
        yield self.confidence


def _scale_points(
    normalized_points: list[float], *, image_width: int, image_height: int
) -> np.ndarray:
    points = np.asarray(normalized_points, dtype=np.float32).reshape(-1, 2)
    points[:, 0] *= float(image_width)
    points[:, 1] *= float(image_height)
    return points


def _normalize_points(points: np.ndarray, *, image_width: int, image_height: int) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected points with shape (N, 2), got {arr.shape}")
    if len(arr) < 3:
        raise ValueError("YOLO segmentation polygons require at least three points")
    out = arr.copy()
    out[:, 0] /= float(image_width)
    out[:, 1] /= float(image_height)
    return out


def _format_float(value: float) -> str:
    return f"{float(value):.8g}"


def read_yolo_seg_gt_label_rows(
    label_path: Path | str, *, image_width: int, image_height: int
) -> list[YoloSegGtRow]:
    rows: list[YoloSegGtRow] = []
    path = Path(label_path)
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            values = [float(value) for value in line.split()]
            if len(values) < 7 or (len(values) - 1) % 2 != 0:
                raise ValueError(f"Invalid GT segmentation label row in {path}")
            class_id = int(values[0])
            rows.append(
                YoloSegGtRow(
                    class_id=class_id,
                    points=_scale_points(
                        values[1:], image_width=image_width, image_height=image_height
                    ),
                )
            )
    return rows


def read_yolo_seg_pred_label_rows(
    label_path: Path | str, *, image_width: int, image_height: int
) -> list[YoloSegPredRow]:
    rows: list[YoloSegPredRow] = []
    path = Path(label_path)
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            values = [float(value) for value in line.split()]
            if len(values) < 8 or (len(values) - 2) % 2 != 0:
                raise ValueError(f"Invalid prediction segmentation label row in {path}")
            class_id = int(values[0])
            rows.append(
                YoloSegPredRow(
                    class_id=class_id,
                    points=_scale_points(
                        values[1:-1], image_width=image_width, image_height=image_height
                    ),
                    confidence=float(values[-1]),
                )
            )
    return rows


def read_yolo_seg_label_rows(
    label_path: Path | str, *, image_width: int, image_height: int
) -> list[YoloSegGtRow]:
    """Read training/ground-truth rows without a confidence suffix."""
    return read_yolo_seg_gt_label_rows(
        label_path, image_width=image_width, image_height=image_height
    )


def _rows_to_polygons(rows: list[YoloSegGtRow] | list[YoloSegPredRow]) -> list[object]:
    from shapely.geometry import Polygon

    polygons: list[object] = []
    for row_index, row in enumerate(rows):
        if len(row.points) < 3:
            warnings.warn(
                f"Skipping label row {row_index}: fewer than three polygon points",
                stacklevel=2,
            )
            continue
        polygon = Polygon([(float(x), float(y)) for x, y in row.points])
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty or polygon.area <= 0:
            warnings.warn(
                f"Skipping label row {row_index}: empty or zero-area polygon",
                stacklevel=2,
            )
            continue
        polygons.append(polygon)
    return polygons


def yolo_seg_labels_to_polygons(
    label_path: Path | str,
    *,
    image_width: int,
    image_height: int,
    has_confidence: bool = False,
) -> list[object]:
    if has_confidence:
        rows = read_yolo_seg_pred_label_rows(
            label_path, image_width=image_width, image_height=image_height
        )
    else:
        rows = read_yolo_seg_gt_label_rows(
            label_path, image_width=image_width, image_height=image_height
        )
    return _rows_to_polygons(rows)


def yolo_seg_labels_to_instance_map(
    label_path: Path | str,
    *,
    image_width: int,
    image_height: int,
    has_confidence: bool = False,
) -> np.ndarray:
    from common.ground_truth import polygons_to_instance_map

    return polygons_to_instance_map(
        yolo_seg_labels_to_polygons(
            label_path,
            image_width=image_width,
            image_height=image_height,
            has_confidence=has_confidence,
        ),
        height=image_height,
        width=image_width,
    )


def write_yolo_seg_gt_label_file(
    label_path: Path | str,
    rows: list[YoloSegGtRow] | list[tuple[int, np.ndarray]],
    *,
    image_width: int,
    image_height: int,
) -> None:
    lines: list[str] = []
    for row in rows:
        class_id, points = row
        normalized = _normalize_points(
            np.asarray(points), image_width=image_width, image_height=image_height
        )
        tokens = [str(int(class_id))]
        tokens.extend(_format_float(value) for value in normalized.reshape(-1))
        lines.append(" ".join(tokens))
    path = Path(label_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_yolo_seg_pred_label_file(
    label_path: Path | str,
    rows: list[YoloSegPredRow] | list[tuple[int, np.ndarray]],
    *,
    image_width: int,
    image_height: int,
    confidences: list[float] | np.ndarray | None = None,
) -> None:
    lines: list[str] = []
    if confidences is not None and len(confidences) != len(rows):
        raise ValueError("confidences length must match rows length")
    for idx, row in enumerate(rows):
        if isinstance(row, YoloSegPredRow):
            class_id, points, confidence = row
        else:
            class_id, points = row
            if confidences is None:
                raise ValueError("confidences are required for tuple prediction rows")
            confidence = float(confidences[idx])
        normalized = _normalize_points(
            np.asarray(points), image_width=image_width, image_height=image_height
        )
        tokens = [str(int(class_id))]
        tokens.extend(_format_float(value) for value in normalized.reshape(-1))
        tokens.append(_format_float(float(confidence)))
        lines.append(" ".join(tokens))
    path = Path(label_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _instance_mask_to_polygons(mask: np.ndarray) -> list[object]:
    from shapely.geometry import Polygon
    from skimage.measure import find_contours

    from common.geometry import iter_polygon_parts

    binary = mask.astype(bool)
    if not binary.any():
        return []

    contours = find_contours(binary, 0.5)
    if not contours:
        return []

    polygons: list[object] = []
    for contour in contours:
        if len(contour) < 3:
            continue
        # find_contours returns (row, col) = (y, x)
        points = np.column_stack([contour[:, 1], contour[:, 0]])
        polygon = Polygon(points)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if not polygon.is_empty and polygon.area > 0:
            polygons.append(polygon)

    if not polygons:
        return []

    if len(polygons) == 1:
        return iter_polygon_parts(polygons[0], context="instance mask")

    largest = max(polygons, key=lambda poly: poly.area)
    return iter_polygon_parts(largest, context="instance mask")


def _polygon_exterior_points(polygon: object) -> np.ndarray:
    coords = np.asarray(polygon.exterior.coords[:-1], dtype=np.float32)
    if len(coords) < 3:
        raise ValueError("Polygon exterior has fewer than three points")
    return coords


def instance_label_map_to_yolo_seg_pred_label_file(
    label_map: np.ndarray,
    label_path: Path | str,
    *,
    default_confidence: float = 1.0,
    class_id: int = 0,
    min_area_px: int = 1,
) -> None:
    if label_map.ndim != 2:
        raise ValueError(f"Expected 2D instance label map, got shape {label_map.shape}")
    height, width = label_map.shape
    rows: list[YoloSegPredRow] = []
    for instance_id in sorted(int(v) for v in np.unique(label_map) if v != 0):
        mask = label_map == instance_id
        if int(mask.sum()) < min_area_px:
            continue
        for polygon in _instance_mask_to_polygons(mask):
            if polygon.area < float(min_area_px):
                continue
            rows.append(
                YoloSegPredRow(
                    class_id=class_id,
                    points=_polygon_exterior_points(polygon),
                    confidence=float(default_confidence),
                )
            )
    write_yolo_seg_pred_label_file(
        label_path,
        rows,
        image_width=width,
        image_height=height,
    )


def yolo_seg_pred_labels_to_coco_dt(
    label_path: Path | str, *, width: int, height: int, image_id: int
) -> list[dict[str, object]]:
    detections: list[dict[str, object]] = []
    for row in read_yolo_seg_pred_label_rows(
        label_path, image_width=width, image_height=height
    ):
        if len(row.points) < 3:
            continue
        segmentation = [row.points.astype(float).reshape(-1).tolist()]
        xs = row.points[:, 0]
        ys = row.points[:, 1]
        x0, x1 = float(xs.min()), float(xs.max())
        y0, y1 = float(ys.min()), float(ys.max())
        detections.append(
            {
                "image_id": int(image_id),
                "category_id": int(row.class_id),
                "segmentation": segmentation,
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "score": float(row.confidence),
            }
        )
    return detections


__all__ = [
    "YoloSegGtRow",
    "YoloSegPredRow",
    "instance_label_map_to_yolo_seg_pred_label_file",
    "read_yolo_seg_gt_label_rows",
    "read_yolo_seg_label_rows",
    "read_yolo_seg_pred_label_rows",
    "write_yolo_seg_gt_label_file",
    "write_yolo_seg_pred_label_file",
    "yolo_seg_labels_to_instance_map",
    "yolo_seg_labels_to_polygons",
    "yolo_seg_pred_labels_to_coco_dt",
]
