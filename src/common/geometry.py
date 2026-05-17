from __future__ import annotations

from pathlib import Path
from typing import Any


def _polygonal_parts_from_cleaned(cleaned: Any, *, context: str) -> list[Any]:
    from shapely.geometry import GeometryCollection, MultiPolygon, Polygon

    if isinstance(cleaned, Polygon):
        return [cleaned] if not cleaned.is_empty else []
    if isinstance(cleaned, MultiPolygon):
        return [p for p in cleaned.geoms if not p.is_empty]
    if isinstance(cleaned, GeometryCollection):
        out: list[Any] = []
        for g in cleaned.geoms:
            if g.is_empty:
                continue
            if isinstance(g, Polygon):
                out.append(g)
            elif isinstance(g, MultiPolygon):
                out.extend(p for p in g.geoms if not p.is_empty)
            else:
                raise ValueError(
                    f"{context}: non-polygonal geometry {g.geom_type!r} in collection "
                    "(expected only Polygon or MultiPolygon parts)"
                )
        return out
    raise ValueError(
        f"{context}: unsupported geometry type {cleaned.geom_type!r} "
        "(expected Polygon, MultiPolygon, or GeometryCollection of polygons)"
    )


def iter_polygon_parts(geometry: Any, *, context: str = "geometry") -> list[Any]:
    if geometry.is_empty:
        return []

    cleaned = geometry.buffer(0)
    if cleaned.is_empty:
        return []

    return _polygonal_parts_from_cleaned(cleaned, context=context)


def load_polygons_from_vector(path: Path) -> list[Any]:
    import geopandas as gpd

    geodata = gpd.read_file(path)
    polygons: list[Any] = []
    for idx, geometry in enumerate(geodata.geometry):
        if geometry is None or geometry.is_empty:
            continue
        parts = iter_polygon_parts(geometry, context=f"{path} feature {idx}")
        polygons.extend(parts)
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
