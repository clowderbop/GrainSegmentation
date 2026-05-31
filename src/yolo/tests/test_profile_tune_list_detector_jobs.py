"""Tests for profile_tune_list_detector_jobs CLI."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from yolo.profile_tune_list_detector_jobs import main


def _write_minimal_grid(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "grid": {
                    "postprocess_type": ["GREEDYNMM"],
                    "match_metric": ["IOS"],
                    "match_threshold": [0.5],
                    "conf": [0.2, 0.3],
                    "mask_threshold": [0.45, 0.55],
                },
            }
        ),
        encoding="utf-8",
    )


def test_list_detector_jobs_prints_tsv_rows(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    grid_path = tmp_path / "grid.yaml"
    _write_minimal_grid(grid_path)
    main(["--grid-config", str(grid_path), "--variants", "PPL"])
    lines = [line for line in capsys.readouterr().out.strip().splitlines() if line]
    assert lines == [
        "PPL\t0.2\t0.45",
        "PPL\t0.2\t0.55",
        "PPL\t0.3\t0.45",
        "PPL\t0.3\t0.55",
    ]
