"""Figure-facing labels for producer families and PQ headline policy."""

import pandas as pd
import pytest

from analysis.build_report import SCOPE_NOTE
from analysis.figures import (
    HEADLINE_PQ_TITLE,
    HeadlineFigureError,
    MODEL_AXIS_LABEL,
    MODEL_VARIANT_BARS_TITLE,
    PPL_DELTA_PQ_TITLE,
    ULTRALYTICS_VAL_PANEL_TITLE,
    model_display_name,
    require_headline_pq_table,
)


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


def test_ultralytics_panel_is_supporting_not_headline() -> None:
    assert "Supporting" in ULTRALYTICS_VAL_PANEL_TITLE
    assert "mAP" in ULTRALYTICS_VAL_PANEL_TITLE


def test_scope_note_headlines_whole_section_pq() -> None:
    assert "Headline whole-section PQ" in SCOPE_NOTE
    assert "Headline AJI" not in SCOPE_NOTE
    assert "F1@IoU50" not in SCOPE_NOTE
    assert "AP/mAP" in SCOPE_NOTE


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
