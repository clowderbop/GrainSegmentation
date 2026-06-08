"""CLI contract for preds-only watershed tuning (issue 14)."""

from __future__ import annotations

from pathlib import Path

import pytest

from unet.tune_watershed import _build_arg_parser, _parse_args


def test_tune_watershed_requires_preds_dir() -> None:
    """INTENT: tune_watershed CLI rejects invocations that omit the required preds-dir."""
    parser = _build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--manifest",
                "m.json",
                "--gt-gpkg",
                "gt.gpkg",
                "--output-csv",
                "out.csv",
            ]
        )


def test_tune_watershed_rejects_model_path() -> None:
    """INTENT: tune_watershed CLI rejects model-path because tuning is preds-only."""
    parser = _build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--model-path",
                "model.keras",
                "--manifest",
                "m.json",
                "--gt-gpkg",
                "gt.gpkg",
                "--output-csv",
                "out.csv",
            ]
        )


def test_tune_watershed_invalid_preds_dir_exits_with_clear_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """INTENT: tune_watershed exits with a clear error when preds-dir is not a directory."""
    gt_gpkg = (
        Path(__file__).resolve().parents[2]
        / "common"
        / "tests"
        / "fixtures"
        / "gpkg_merged_instance_map"
        / "micro_labels.gpkg"
    )
    missing = tmp_path / "no_such_preds"
    with pytest.raises(SystemExit):
        _parse_args(
            [
                "--preds-dir",
                str(missing),
                "--manifest",
                "m.json",
                "--gt-gpkg",
                str(gt_gpkg),
                "--output-csv",
                "out.csv",
            ]
        )
    assert "preds-dir is not a directory" in capsys.readouterr().err


