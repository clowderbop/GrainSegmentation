from __future__ import annotations

from typing import Any

from shapely.geometry import Polygon, box
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.polygon import orient

from common.geometry import iter_polygon_parts


def clip_polygon_to_hw(polygon: Polygon, height: int, width: int) -> list[Polygon]:
    frame = box(0, 0, width, height)
    try:
        clipped = polygon.intersection(frame)
    except Exception:
        return []
    return [
        p
        for p in iter_polygon_parts(clipped, context="clip_polygon_to_hw")
        if not p.is_empty and p.area > 0
    ]


def polygon_to_coco_polygon(polygon: Polygon) -> list[float]:
    coords = list(orient(polygon, sign=1.0).exterior.coords[:-1])
    if len(coords) < 3:
        return []
    flat: list[float] = []
    for x, y in coords:
        flat.extend((float(x), float(y)))
    return flat


def build_gt_annotations(
    polygons: list[Polygon | MultiPolygon],
    *,
    image_id: int,
    height: int,
    width: int,
    category_id: int = 1,
) -> list[dict[str, Any]]:
    anns: list[dict[str, Any]] = []
    ann_id = 1
    for geom in polygons:
        for part in iter_polygon_parts(geom, context="build_gt_annotations"):
            for clipped in clip_polygon_to_hw(part, height, width):
                seg_flat = polygon_to_coco_polygon(clipped)
                if len(seg_flat) < 6:
                    continue
                xs = seg_flat[0::2]
                ys = seg_flat[1::2]
                x0, x1 = min(xs), max(xs)
                y0, y1 = min(ys), max(ys)
                w, h = x1 - x0, y1 - y0
                area = float(clipped.area)
                anns.append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "segmentation": [seg_flat],
                        "area": area,
                        "bbox": [float(x0), float(y0), float(w), float(h)],
                        "iscrowd": 0,
                    }
                )
                ann_id += 1
    return anns
