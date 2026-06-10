"""Load tuned watershed best_params from watershed_best_*.json artifacts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from common.variants import repo_root
from unet.extraction_tune_scoring import WatershedParamSet
from unet.tests.watershed_best_json_fixtures import write_watershed_best_json
from unet.watershed_best_params import load_watershed_best_params


def test_load_watershed_best_params_reads_min_area_px(tmp_path: Path) -> None:
    """INTENT: shared loader returns tuned min_area_px from watershed_best JSON."""
    params = WatershedParamSet(5, 1, 2, 256, True, 0.25)
    json_path = tmp_path / "watershed_best_12345.json"
    write_watershed_best_json(json_path, params)

    loaded = load_watershed_best_params(json_path)

    assert loaded == params


def test_load_watershed_best_params_rejects_missing_best_params(
    tmp_path: Path,
) -> None:
    """INTENT: shared loader rejects watershed JSON without a best_params object."""
    json_path = tmp_path / "watershed_best_bad.json"
    json_path.write_text(json.dumps({"selection_objective": "pq"}), encoding="utf-8")

    with pytest.raises(ValueError, match="best_params"):
        load_watershed_best_params(json_path)


def test_watershed_json_min_area_px_cli_prints_tuned_value(tmp_path: Path) -> None:
    """INTENT: min_area_px query CLI reuses the shared watershed best-params loader."""
    params = WatershedParamSet(5, 0, 1, 64, False, None)
    json_path = tmp_path / "watershed_best_99.json"
    write_watershed_best_json(json_path, params)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "unet.watershed_json_min_area_px",
            str(json_path),
        ],
        cwd=repo_root() / "src" / "unet",
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "64"
