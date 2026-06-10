"""Load validated watershed best_params from watershed_best_*.json artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from unet.extraction_tune_scoring import WatershedParamSet


def watershed_best_params_from_payload(payload: dict[str, Any]) -> WatershedParamSet:
    best_params = payload.get("best_params")
    if not isinstance(best_params, dict):
        raise ValueError("JSON missing best_params object")

    ridge_level = best_params.get("ridge_level")
    return WatershedParamSet(
        min_distance=int(best_params["min_distance"]),
        boundary_dilate_iter=int(best_params["boundary_dilate_iter"]),
        watershed_connectivity=int(best_params["watershed_connectivity"]),
        min_area_px=int(best_params["min_area_px"]),
        exclude_border=bool(best_params["exclude_border"]),
        ridge_level=None if ridge_level is None else float(ridge_level),
    )


def load_watershed_best_params(path: Path | str) -> WatershedParamSet:
    json_path = Path(path)
    with json_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return watershed_best_params_from_payload(payload)
