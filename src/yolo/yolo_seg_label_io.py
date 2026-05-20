"""Read YOLO segmentation-format label rows (*.txt polygons in normalized coords)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from shapely.geometry import Polygon

__all__ = [
    "read_yolo_seg_label_rows",
    "yolo_seg_labels_to_instance_map",
    "yolo_seg_labels_to_polygons",
]


def read_yolo_seg_label_rows(
    label_path: Path, *, image_width: int, image_height: int
) -> list[tuple[int, np.ndarray]]:
    rows: list[tuple[int, np.ndarray]] = []
    with label_path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            values = [float(value) for value in line.split()]
            if len(values) < 7 or (len(values) - 1) % 2 != 0:
                raise ValueError(f"Invalid segmentation label row in {label_path}")
            class_id = int(values[0])
            points = np.asarray(values[1:], dtype=np.float32).reshape(-1, 2)
            points[:, 0] *= float(image_width)
            points[:, 1] *= float(image_height)
            rows.append((class_id, points))
    return rows


def yolo_seg_labels_to_polygons(
    label_path: Path, *, image_width: int, image_height: int
) -> list[Polygon]:
    from shapely.geometry import Polygon

    polygons: list[Polygon] = []
    for _class_id, points in read_yolo_seg_label_rows(
        label_path, image_width=image_width, image_height=image_height
    ):
        if len(points) < 3:
            continue
        polygon = Polygon([(float(x), float(y)) for x, y in points])
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if not polygon.is_empty and polygon.area > 0:
            polygons.append(polygon)
    return polygons


def yolo_seg_labels_to_instance_map(
    label_path: Path, *, image_width: int, image_height: int
) -> np.ndarray:
    from common.ground_truth import polygons_to_instance_map

    return polygons_to_instance_map(
        yolo_seg_labels_to_polygons(
            label_path, image_width=image_width, image_height=image_height
        ),
        height=image_height,
        width=image_width,
    )
