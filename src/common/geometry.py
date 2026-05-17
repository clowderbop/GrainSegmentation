from __future__ import annotations

from pathlib import Path
from typing import Any


def iter_polygon_parts(geometry: Any) -> list[Any]:
    from shapely.geometry import GeometryCollection, MultiPolygon, Polygon

    if geometry.is_empty:
        return []

    cleaned = geometry.buffer(0)
    if cleaned.is_empty:
        return []
    if isinstance(cleaned, Polygon):
        return [cleaned]
    if isinstance(cleaned, MultiPolygon):
        return [part for part in cleaned.geoms if not part.is_empty]
    if isinstance(cleaned, GeometryCollection):
        return [
            part
            for part in cleaned.geoms
            if isinstance(part, Polygon) and not part.is_empty
        ]
    return []


def load_polygons_from_vector(path: Path) -> list[Any]:
    import geopandas as gpd
    from shapely.geometry import GeometryCollection, MultiPolygon, Polygon

    geodata = gpd.read_file(path)
    polygons: list[Any] = []
    for geometry in geodata.geometry:
        if geometry is None or geometry.is_empty:
            continue
        if isinstance(geometry, (Polygon, MultiPolygon)):
            polygons.append(geometry)
            continue
        if isinstance(geometry, GeometryCollection):
            polygons.extend(
                part
                for part in geometry.geoms
                if isinstance(part, (Polygon, MultiPolygon)) and not part.is_empty
            )
    return polygons


def normalize_polygons_to_image_space(polygons: list[Any]) -> list[Any]:
    from shapely.affinity import scale as scale_geometry

    if not polygons:
        return polygons

    min_y = min(polygon.bounds[1] for polygon in polygons if not polygon.is_empty)
    max_y = max(polygon.bounds[3] for polygon in polygons if not polygon.is_empty)
    if max_y <= 0 and min_y < 0:
        return [
            scale_geometry(polygon, xfact=1.0, yfact=-1.0, origin=(0.0, 0.0))
            for polygon in polygons
        ]
    return polygons


def load_image_space_polygons(path: Path) -> list[Any]:
    return normalize_polygons_to_image_space(load_polygons_from_vector(path))
