"""Test helpers for watershed_best_*.json artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from unet.extraction_tune_scoring import WatershedParamSet


def write_watershed_best_json(path: Path, params: WatershedParamSet) -> None:
    payload = {
        "selection_objective": "pq",
        "best_params": {
            "min_distance": params.min_distance,
            "boundary_dilate_iter": params.boundary_dilate_iter,
            "h_maxima": params.h_maxima,
            "watershed_connectivity": params.watershed_connectivity,
            "min_area_px": params.min_area_px,
            "exclude_border": params.exclude_border,
            "ridge_level": params.ridge_level,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
