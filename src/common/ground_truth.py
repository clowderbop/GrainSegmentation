from __future__ import annotations

from typing import Any, Literal

from shapely.affinity import translate

from common.gpkg_instance_map import paint_polygons_merged_instance_view
from common.patching import sample_origin_xy, sample_origin_xy_or_whole_image

GtOriginMode = Literal["patch_stem", "whole_image"]


def polygons_to_instance_map(
    polygons: list[Any],
    *,
    height: int,
    width: int,
) -> np.ndarray:
    return paint_polygons_merged_instance_view(
        polygons, height=height, width=width
    )


def scene_polygons_to_patch_instance_map(
    polygons: list[Any],
    *,
    sample_id: str,
    height: int,
    width: int,
    gt_origin: GtOriginMode,
) -> np.ndarray:
    """Translate GIS polygons into patch image space, then paint merged instance view."""
    if gt_origin == "whole_image":
        origin_x, origin_y = sample_origin_xy_or_whole_image(sample_id)
    else:
        origin_x, origin_y = sample_origin_xy(sample_id)
    if origin_x or origin_y:
        polygons = [
            translate(p, xoff=-float(origin_x), yoff=-float(origin_y)) for p in polygons
        ]
    return paint_polygons_merged_instance_view(
        polygons, height=height, width=width
    )
