"""CLI contract for preds-only watershed tuning (issue 14)."""

from __future__ import annotations

from pathlib import Path

import pytest

from unet.tune_watershed import _build_arg_parser, _parse_args


def test_tune_watershed_requires_preds_dir() -> None:
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


def test_tune_watershed_cli_log_extraction_cache_defaults_off() -> None:
    args = _build_arg_parser().parse_args(
        [
            "--preds-dir",
            "/tmp/preds",
            "--manifest",
            "m.json",
            "--gt-gpkg",
            "gt.gpkg",
            "--output-csv",
            "out.csv",
        ]
    )
    assert args.log_extraction_cache is False


def test_tune_watershed_cli_accepts_log_extraction_cache() -> None:
    args = _build_arg_parser().parse_args(
        [
            "--preds-dir",
            "/tmp/preds",
            "--manifest",
            "m.json",
            "--gt-gpkg",
            "gt.gpkg",
            "--output-csv",
            "out.csv",
            "--log-extraction-cache",
        ]
    )
    assert args.log_extraction_cache is True
