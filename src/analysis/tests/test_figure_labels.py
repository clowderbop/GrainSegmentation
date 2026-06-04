"""Figure-facing labels for producer families and PQ headline policy."""

from pathlib import Path

import pandas as pd
import pytest

from analysis.build_report import SCOPE_NOTE
from analysis.derived_tables import whole_section_pq_matrix_table
from analysis.figures import (
    COUNT_ERROR_BAR_CHART_TITLE,
    HEADLINE_PQ_TITLE,
    HeadlineFigureError,
    MODEL_AXIS_LABEL,
    MODEL_VARIANT_BARS_TITLE,
    PARETO_PLOT_TITLE,
    PATCH_TO_WHOLE_DIAGNOSTIC_HEATMAP_TITLE,
    PQ_DECOMPOSITION_GROUPED_BARS_TITLE,
    PPL_DELTA_PQ_TITLE,
    PRECISION_RECALL_IOU75_TITLE,
    STRICTNESS_DROP_PLOT_TITLE,
    ULTRALYTICS_VAL_PANEL_TITLE,
    figure_headline_heatmap,
    figure_pq_decomposition_grouped_bars,
    model_display_name,
    require_headline_pq_table,
)
from analysis.tests.test_derived_tables import _four_combo_instance_df


def test_model_display_name_maps_producers() -> None:
    assert model_display_name("yolo") == "YOLO"
    assert model_display_name("unet") == "U-Net"


def test_model_axis_label() -> None:
    assert MODEL_AXIS_LABEL == "Model"


def test_headline_figure_copy_uses_whole_section_pq() -> None:
    assert "PQ" in HEADLINE_PQ_TITLE
    assert "whole-section" in HEADLINE_PQ_TITLE.lower()
    assert "AJI" not in HEADLINE_PQ_TITLE
    assert "PQ" in MODEL_VARIANT_BARS_TITLE
    assert "PQ" in PPL_DELTA_PQ_TITLE
    assert "AJI" not in MODEL_VARIANT_BARS_TITLE


def test_threshold_diagnostic_figure_titles_are_not_headline() -> None:
    for title in (
        PATCH_TO_WHOLE_DIAGNOSTIC_HEATMAP_TITLE,
        COUNT_ERROR_BAR_CHART_TITLE,
        STRICTNESS_DROP_PLOT_TITLE,
        PRECISION_RECALL_IOU75_TITLE,
        PARETO_PLOT_TITLE,
    ):
        assert "Diagnostic" in title
        assert "whole-section" in title.lower() or "patch-to-whole" in title.lower()
        assert "Headline" not in title


def test_ultralytics_panel_is_supporting_not_headline() -> None:
    assert "Supporting" in ULTRALYTICS_VAL_PANEL_TITLE
    assert "mAP" in ULTRALYTICS_VAL_PANEL_TITLE


def test_scope_note_headlines_whole_section_pq() -> None:
    assert "Headline whole-section PQ" in SCOPE_NOTE
    assert "Headline AJI" not in SCOPE_NOTE
    assert "F1@IoU50" not in SCOPE_NOTE
    assert "AP/mAP" in SCOPE_NOTE


def test_figure_headline_heatmap_pq_panel_uses_derived_matrix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
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


def test_pq_decomposition_grouped_bars_title_and_file(
    tmp_path: Path,
) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")

    assert "PQ" in PQ_DECOMPOSITION_GROUPED_BARS_TITLE
    assert "DQ" in PQ_DECOMPOSITION_GROUPED_BARS_TITLE
    assert "SQ" in PQ_DECOMPOSITION_GROUPED_BARS_TITLE
    assert "AJI" not in PQ_DECOMPOSITION_GROUPED_BARS_TITLE

    out = tmp_path / "pq_decomposition_grouped_bars.png"
    figure_pq_decomposition_grouped_bars(_four_combo_instance_df(), out)
    assert out.is_file()

