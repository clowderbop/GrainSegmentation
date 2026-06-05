"""Default production watershed tune grid contract (scale-watershed-tuning issue 02)."""

from __future__ import annotations

import re

from unet.slurm_watershed_tune import run_watershed_tuning_script_path
from unet.tune_watershed import _build_arg_parser
from unet.watershed_tune_grid import (
    DEFAULT_WATERSHED_TUNE_CANDIDATE_COUNT,
    default_watershed_tune_grid_axes,
)


def _bash_int_array(script_text: str, name: str) -> tuple[int, ...]:
    match = re.search(rf"^{name}=\(([^)]*)\)", script_text, flags=re.MULTILINE)
    assert match is not None, f"{name}=() not found in run_watershed_tuning.sh"
    inner = match.group(1).strip()
    if not inner:
        return ()
    return tuple(int(part) for part in inner.split())


def test_default_watershed_tune_grid_excludes_pixel_scale_min_distance() -> None:
    axes = default_watershed_tune_grid_axes()
    assert 1 not in axes["min_distance"]
    assert axes["min_distance"] == (3, 5, 9)


def test_default_watershed_tune_grid_includes_min_area_speckle_axis() -> None:
    axes = default_watershed_tune_grid_axes()
    assert axes["min_area_px"] == (0, 64, 256)
    assert any(v > 0 for v in axes["min_area_px"])


def test_default_watershed_tune_grid_preserves_other_extraction_axes() -> None:
    axes = default_watershed_tune_grid_axes()
    assert axes["boundary_dilate_iter"] == (0, 1)
    assert axes["watershed_connectivity"] == (1, 2)
    assert axes["exclude_border"] == (0, 1)
    assert axes["ridge_level"] == (None,)


def test_default_watershed_tune_candidate_count_matches_documented_grid() -> None:
    assert DEFAULT_WATERSHED_TUNE_CANDIDATE_COUNT == 72


def _cli_default(dest: str) -> object:
    for action in _build_arg_parser()._actions:
        if action.dest == dest:
            return action.default
    raise KeyError(dest)


def test_tune_watershed_cli_defaults_match_production_grid() -> None:
    axes = default_watershed_tune_grid_axes()
    assert _cli_default("min_distance") == list(axes["min_distance"])
    assert _cli_default("boundary_dilate_iter") == list(axes["boundary_dilate_iter"])
    assert _cli_default("watershed_connectivity") == list(axes["watershed_connectivity"])
    assert _cli_default("min_area_px") == list(axes["min_area_px"])
    assert _cli_default("exclude_border") == list(axes["exclude_border"])
    assert _cli_default("ridge_level") is None


def test_run_watershed_tuning_shell_default_grid_matches_python_contract() -> None:
    text = run_watershed_tuning_script_path().read_text(encoding="utf-8")
    axes = default_watershed_tune_grid_axes()
    assert _bash_int_array(text, "MIN_DISTANCE") == axes["min_distance"]
    assert _bash_int_array(text, "BOUNDARY_DILATE_ITER") == axes["boundary_dilate_iter"]
    assert _bash_int_array(text, "WATERSHED_CONNECTIVITY") == axes["watershed_connectivity"]
    assert _bash_int_array(text, "MIN_AREA_PX") == axes["min_area_px"]
    assert _bash_int_array(text, "EXCLUDE_BORDER") == axes["exclude_border"]
