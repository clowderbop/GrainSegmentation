"""Figure-facing labels for producer families and PQ headline policy."""

from pathlib import Path

import pandas as pd
import pytest

from analysis.derived_tables import whole_section_pq_matrix_table
from analysis.figures import (
    HeadlineFigureError,
    figure_headline_heatmap,
    figure_pq_decomposition_grouped_bars,
    model_display_name,
    require_headline_pq_table,
)
from analysis.tests.test_derived_tables import _four_combo_instance_df


def test_model_display_name_maps_producers() -> None:
    """INTENT: thesis-facing producer labels stay stable for report figures."""
    assert model_display_name("yolo") == "YOLO"
    assert model_display_name("unet") == "U-Net"


def test_figure_headline_heatmap_pq_panel_uses_derived_matrix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """INTENT: headline heatmap plots whole-section PQ from the derived matrix table."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")

    calls: list[pd.DataFrame] = []

    def tracking(df: pd.DataFrame) -> pd.DataFrame:
        result = whole_section_pq_matrix_table(df)
        calls.append(result)
        return result

    monkeypatch.setattr("analysis.figures.whole_section_pq_matrix_table", tracking)
    figure_headline_heatmap(_four_combo_instance_df(), tmp_path / "headline_heatmap.png")

    assert len(calls) == 1
    assert calls[0].loc["YOLO", "PPL"] == pytest.approx(0.30)
    assert (tmp_path / "headline_heatmap.png").is_file()


def test_require_headline_pq_table_rejects_missing_pq_columns() -> None:
    """INTENT: headline figures fail fast when PQ columns are absent."""
    df = pd.DataFrame(
        [
            {
                "unit": "whole",
                "source": "instance",
                "producer": "yolo",
                "display_name": "PPL",
                "aji_plus": 0.1,
            }
        ]
    )
    with pytest.raises(HeadlineFigureError, match="missing"):
        require_headline_pq_table(df)


def test_pq_decomposition_grouped_bars_writes_figure(tmp_path: Path) -> None:
    """INTENT: PQ decomposition grouped bars render from instance metrics input."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")

    out = tmp_path / "pq_decomposition_grouped_bars.png"
    figure_pq_decomposition_grouped_bars(_four_combo_instance_df(), out)
    assert out.is_file()
