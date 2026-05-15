import argparse
import numpy as np
import cv2
import geopandas as gpd
from PIL import Image
import tifffile
import sys
import os

Image.MAX_IMAGE_PIXELS = None


def process_polygons(gdf, height, width, boundary_width, flip_y):
    mask = np.zeros((height, width), dtype=np.uint8)


    def _get_rings(polygon):
        rings = [list(polygon.exterior.coords)]
        for interior in polygon.interiors:
            rings.append(list(interior.coords))
        return rings

    def _draw_poly(m, poly, val):
        if poly.geom_type == "Polygon":
            rings = _get_rings(poly)
            pts = [np.array(r, dtype=np.int32) for r in rings]
            if flip_y:
                for p in pts:
                    p[:, 1] = -p[:, 1]
            cv2.fillPoly(m, [pts[0]], val)
            for hole in pts[1:]:
                cv2.fillPoly(m, [hole], 0)
        elif poly.geom_type == "MultiPolygon":
            for p in poly.geoms:
                _draw_poly(m, p, val)


    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        _draw_poly(mask, geom, 2)


    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue

        inner_geom = geom.buffer(-boundary_width)
        if not inner_geom.is_empty:
            _draw_poly(mask, inner_geom, 1)

    return mask


def main():
    parser = argparse.ArgumentParser(
        )
    parser.add_argument("-i", "--input", required=True, )
    parser.add_argument(
        "-r", "--reference", required=True, )
    parser.add_argument(
        "-o", "--output", required=True, )
    parser.add_argument(
        "--boundary-width",
        type=float,
        default=3.0,
        )
    parser.add_argument(
        "--no-flip-y",
        action="store_true",
        )

    args = parser.parse_args()


    try:
        with Image.open(args.reference) as img:
            width, height = img.size
    except Exception as e:
        print(f"Error reading reference image: {e}")
        sys.exit(1)


    try:
        gdf = gpd.read_file(args.input)
    except Exception as e:
        print(f"Error reading GPKG file: {e}")
        sys.exit(1)


    mask = process_polygons(gdf, height, width, args.boundary_width, not args.no_flip_y)


    output_path = args.output
    if not (
        output_path.lower().endswith(".tif") or output_path.lower().endswith(".tiff")
    ):
        output_path = os.path.splitext(output_path)[0] + ".tif"

    try:
        tifffile.imwrite(output_path, mask, compression="deflate")
        print(f"Saved mask to {output_path}")
    except Exception as e:
        print(f"Error saving output image: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
