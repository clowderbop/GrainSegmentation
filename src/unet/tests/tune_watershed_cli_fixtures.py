"""Shared fixtures for tune_watershed CLI integration tests."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
import numpy as np
import tifffile
import yaml

from unet.extraction_tune_scoring import WatershedParamSet
from unet.watershed_tune_fixtures import cached_semantic_pred_speckle_prone
from unet.watershed_tune_grid_shard import WatershedTuneShard

REPO_SRC = Path(__file__).resolve().parents[2]
MICRO_GPKG = (
    REPO_SRC
    / "common"
    / "tests"
    / "fixtures"
    / "gpkg_merged_instance_map"
    / "micro_labels.gpkg"
)
TUNE_CLI_HEIGHT = 48
TUNE_CLI_WIDTH = 64

MINI_TUNE_GRID: dict[str, object] = {
    "min_distance": [5, 9],
    "boundary_dilate_iter": [0],
    "watershed_connectivity": [1],
    "min_area_px": [0, 64],
    "exclude_border": [0],
    "ridge_level": [None],
}


def write_mini_tune_grid(
    path: Path,
    *,
    grid: dict[str, object] | None = None,
) -> None:
    path.write_text(
        yaml.safe_dump({"grid": grid or MINI_TUNE_GRID}),
        encoding="utf-8",
    )


def make_tune_collect_args(
    tmp_path: Path,
    *,
    sample_id: str = "train",
    pred_shape: tuple[int, int] = (TUNE_CLI_HEIGHT, TUNE_CLI_WIDTH),
    image_shape: tuple[int, int] = (TUNE_CLI_HEIGHT, TUNE_CLI_WIDTH),
    paint_semantic_region: bool = False,
    semantic: np.ndarray | None = None,
    metadata_only_manifest: bool = False,
) -> Namespace:
    """Minimal manifest + cached pred for tune_watershed CLI or ``_collect_samples``."""
    pred_height, pred_width = pred_shape
    image_height, image_width = image_shape

    image_path = tmp_path / f"{sample_id}_PPL.tif"
    if not metadata_only_manifest:
        rgb = np.zeros((image_height, image_width, 3), dtype=np.uint8)
        tifffile.imwrite(image_path, rgb, photometric="rgb")
    manifest_image = "missing_PPL.tif" if metadata_only_manifest else str(image_path)

    pred_path = tmp_path / "preds" / f"{sample_id}_pred.tif"
    pred_path.parent.mkdir(parents=True)
    if semantic is not None:
        pred_arr = semantic
    else:
        pred_arr = np.zeros((pred_height, pred_width), dtype=np.uint8)
        if paint_semantic_region:
            pred_arr[8:20, 8:24] = 1
    tifffile.imwrite(pred_path, pred_arr)

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "variant": "PPL",
                "unit": "whole",
                "grainseg_root": str(tmp_path),
                "path_base": "work_root",
                "samples": [
                    {
                        "sample_id": sample_id,
                        "image": manifest_image,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    return Namespace(
        manifest=manifest_path,
        gt_gpkg=MICRO_GPKG,
        preds_dir=tmp_path / "preds",
        max_samples=None,
        num_inputs=None,
    )


def watershed_param_set_from_csv_row(row: dict[str, str]) -> WatershedParamSet:
    return WatershedParamSet(
        min_distance=int(row["min_distance"]),
        boundary_dilate_iter=int(row["boundary_dilate_iter"]),
        watershed_connectivity=int(row["watershed_connectivity"]),
        min_area_px=int(row["min_area_px"]),
        exclude_border=bool(int(row["exclude_border"])),
        ridge_level=None if row["ridge_level"] == "" else float(row["ridge_level"]),
    )


def watershed_param_sets_from_csv_rows(
    rows: list[dict[str, str]],
) -> list[WatershedParamSet]:
    return [watershed_param_set_from_csv_row(row) for row in rows]


def tune_watershed_argv(
    args: Namespace,
    *,
    output_csv: Path,
    grid_config: Path,
    output_json: Path | None = None,
    shard: WatershedTuneShard | None = None,
) -> list[str]:
    argv = [
        "tune_watershed",
        "--preds-dir",
        str(args.preds_dir),
        "--manifest",
        str(args.manifest),
        "--gt-gpkg",
        str(args.gt_gpkg),
        "--output-csv",
        str(output_csv),
        "--grid-config",
        str(grid_config),
    ]
    if output_json is not None:
        argv.extend(["--output-json", str(output_json)])
    if shard is not None:
        argv.extend(
            [
                "--shard-index",
                str(shard.index),
                "--shard-min-distance",
                str(shard.min_distance),
                "--shard-boundary-dilate-iter",
                str(shard.boundary_dilate_iter),
            ]
        )
    return argv


def speckle_prone_tune_collect_args(
    tmp_path: Path,
    *,
    height: int = TUNE_CLI_HEIGHT,
    width: int = TUNE_CLI_WIDTH,
) -> Namespace:
    return make_tune_collect_args(
        tmp_path,
        semantic=cached_semantic_pred_speckle_prone(height, width),
        metadata_only_manifest=True,
    )

