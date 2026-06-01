"""OpenCV GPKG → merged instance view (ADR 0006)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from shapely.geometry.polygon import orient

from common.coco_annotations import clip_polygon_to_hw
from common.geometry import iter_polygon_parts, load_image_space_polygons


def paint_polygons_merged_instance_view(
    polygons: list[Any],
    *,
    height: int,
    width: int,
) -> np.ndarray:
    """Paint exterior rings with ascending instance id (later grains win overlaps)."""
    paint_order: list[tuple[int, np.ndarray]] = []
    instance_id = 1
    for geom in polygons:
        for part in iter_polygon_parts(geom, context="paint_polygons_merged_instance_view"):
            for clipped in clip_polygon_to_hw(part, height, width):
                coords = list(orient(clipped, sign=1.0).exterior.coords[:-1])
                if len(coords) < 3:
                    continue
                pts = np.rint(np.asarray(coords, dtype=np.float64)).astype(np.int32)
                paint_order.append((instance_id, pts))
                instance_id += 1

    out = np.zeros((height, width), dtype=np.int32)
    for lid, pts in sorted(paint_order, key=lambda item: item[0]):
        cv2.fillPoly(out, [pts], int(lid))
    return out


def gpkg_to_merged_instance_map(
    gpkg_path: Path,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    polygons = load_image_space_polygons(Path(gpkg_path))
    return paint_polygons_merged_instance_view(
        polygons, height=height, width=width
    )
