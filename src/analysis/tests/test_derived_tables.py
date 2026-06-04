"""Thesis-core derived tables from normalized instance metrics."""

from __future__ import annotations

import pandas as pd
import pytest

from analysis.derived_tables import (
    INPUT_CONFIGURATION_COL,
    MODEL_COL,
    WHOLE_SECTION_PQ_COL,
    headline_ranking_table,
    thesis_ready_results_table,
    whole_section_pq_matrix_table,
)
from analysis.tests.test_load_metrics import PQ_SAMPLE_ROW
from common.instance_metric_bundle import INSTANCE_METRIC_BUNDLE_KEYS
from common.variants import variant_display_names_in_thesis_order


def _metric_bundle(**overrides: float) -> dict[str, float]:
    base = {key: PQ_SAMPLE_ROW[key] for key in INSTANCE_METRIC_BUNDLE_KEYS}
    base.update(overrides)
    return base


def _whole_row(
    *,
    producer: str,
    variant: str,
    display_name: str,
    pq: float,
) -> dict[str, object]:
    return {
        "producer": producer,
        "variant": variant,
        "display_name": display_name,
        "unit": "whole",
        "source": "instance",
        **_metric_bundle(pq=pq),
    }


def _four_combo_instance_df() -> pd.DataFrame:
    """Two producers × two variants with distinct whole-section PQ for ranking tests."""
    return pd.DataFrame(
        [
            _whole_row(
                producer="yolo",
                variant="PPL",
                display_name="PPL",
                pq=0.30,
            ),
            _whole_row(
                producer="yolo",
                variant="PPL+AllPPX",
                display_name="FullStack",
                pq=0.50,
            ),
            _whole_row(
                producer="unet",
                variant="PPL",
                display_name="PPL",
                pq=0.40,
            ),
            _whole_row(
                producer="unet",
                variant="PPL+AllPPX",
                display_name="FullStack",
                pq=0.45,
            ),
        ]
    )


def test_headline_ranking_table_ranks_by_whole_section_pq_descending() -> None:
    table = headline_ranking_table(_four_combo_instance_df())

    assert list(table.columns[:4]) == [
        "Rank",
        MODEL_COL,
        INPUT_CONFIGURATION_COL,
        WHOLE_SECTION_PQ_COL,
    ]
    assert table["Rank"].tolist() == [1, 2, 3, 4]
    assert table[WHOLE_SECTION_PQ_COL].tolist() == pytest.approx([0.50, 0.45, 0.40, 0.30])
    assert table.iloc[0][MODEL_COL] == "YOLO"
    assert table.iloc[0][INPUT_CONFIGURATION_COL] == "FullStack"


def test_whole_section_pq_matrix_uses_model_rows_and_thesis_input_columns() -> None:
    table = whole_section_pq_matrix_table(_four_combo_instance_df())

    assert table.index.name == MODEL_COL
    assert list(table.columns) == ["PPL", "FullStack"]
    assert table.loc["YOLO", "PPL"] == pytest.approx(0.30)
    assert table.loc["U-Net", "FullStack"] == pytest.approx(0.45)


def test_thesis_ready_results_table_has_approved_columns_and_thesis_order() -> None:
    table = thesis_ready_results_table(_four_combo_instance_df())

    assert table.columns[0] == MODEL_COL
    assert table.columns[1] == INPUT_CONFIGURATION_COL
    assert WHOLE_SECTION_PQ_COL in table.columns
    assert "DQ" in table.columns
    assert "SQ" in table.columns
    assert "Mean precision @ IoU 0.50:0.95" in table.columns
    assert "Mean recall @ IoU 0.50:0.95" in table.columns
    assert "Mean F1 @ IoU 0.50:0.95" in table.columns
    assert "AJI+" in table.columns
    assert not any("map" in col.lower() or col.lower().startswith("ap") for col in table.columns)

    yolo_rows = table[table[MODEL_COL] == "YOLO"]
    assert yolo_rows[INPUT_CONFIGURATION_COL].tolist() == ["PPL", "FullStack"]
    assert yolo_rows[WHOLE_SECTION_PQ_COL].tolist() == pytest.approx([0.30, 0.50])


def test_thesis_ready_results_orders_models_yolo_before_unet() -> None:
    table = thesis_ready_results_table(_four_combo_instance_df())
    assert table[MODEL_COL].tolist() == ["YOLO", "YOLO", "U-Net", "U-Net"]
