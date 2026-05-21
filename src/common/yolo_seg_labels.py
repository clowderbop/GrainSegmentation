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


__all__ = [
    "YoloSegGtRow",
    "YoloSegPredRow",
    "read_yolo_seg_gt_label_rows",
    "read_yolo_seg_label_rows",
    "read_yolo_seg_pred_label_rows",
    "yolo_seg_labels_to_instance_map",
    "yolo_seg_labels_to_polygons",
]
