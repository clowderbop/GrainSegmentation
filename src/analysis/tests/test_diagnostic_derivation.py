"""PQ decomposition and failure-mode diagnostic derivation."""

from __future__ import annotations

import pandas as pd
import pytest

from analysis.diagnostic_derivation import (
    FAILURE_MODE_LABEL_COL,
    failure_mode_classification_table,
    failure_mode_metrics_available,
    failure_mode_rules_markdown,
    pq_decomposition_long_table,
    pq_decomposition_metrics_available,
)
from analysis.derived_tables import INPUT_CONFIGURATION_COL, MODEL_COL
from analysis.tests.test_derived_tables import _four_combo_instance_df, _whole_row


def test_pq_decomposition_long_table_orders_model_input_and_metric() -> None:
    table = pq_decomposition_long_table(_four_combo_instance_df())

    assert list(table.columns) == [MODEL_COL, INPUT_CONFIGURATION_COL, "Metric", "Value"]
    assert table[MODEL_COL].tolist() == ["YOLO"] * 6 + ["U-Net"] * 6
    assert table[INPUT_CONFIGURATION_COL].tolist() == (
        ["PPL", "PPL", "PPL", "FullStack", "FullStack", "FullStack"] * 2
    )
    assert table["Metric"].tolist() == ["PQ", "DQ", "SQ"] * 4
    yolo_ppl = table[
        (table[MODEL_COL] == "YOLO") & (table[INPUT_CONFIGURATION_COL] == "PPL")
    ]
    assert yolo_ppl.set_index("Metric")["Value"].to_dict() == pytest.approx(
        {"PQ": 0.30, "DQ": 0.5, "SQ": 0.84}
    )


def test_failure_mode_classification_labels_pq_and_count_cases() -> None:
    df = pd.DataFrame(
        [
            _whole_row(
                producer="yolo",
                variant="PPL",
                display_name="PPL",
                pq=0.20,
            )
            | {"dq": 0.30, "sq": 0.80, "pred_gt_instance_ratio": 1.15},
            _whole_row(
                producer="unet",
                variant="PPL",
                display_name="PPL",
                pq=0.50,
            )
            | {"dq": 0.85, "sq": 0.40, "pred_gt_instance_ratio": 0.70},
        ]
    )
    table = failure_mode_classification_table(df)

    assert list(table.columns) == [
        MODEL_COL,
        INPUT_CONFIGURATION_COL,
        FAILURE_MODE_LABEL_COL,
    ]
    yolo = table[table[MODEL_COL] == "YOLO"].iloc[0]
    unet = table[table[MODEL_COL] == "U-Net"].iloc[0]
    assert "detection-limited" in yolo[FAILURE_MODE_LABEL_COL]
    assert "overpredicting" in yolo[FAILURE_MODE_LABEL_COL]
    assert "mask-quality-limited" in unet[FAILURE_MODE_LABEL_COL]
    assert "underpredicting" in unet[FAILURE_MODE_LABEL_COL]


def test_failure_mode_rules_markdown_documents_thresholds() -> None:
    text = failure_mode_rules_markdown()

    assert "detection-limited" in text
    assert "mask-quality-limited" in text
    assert "overpredicting" in text
    assert "underpredicting" in text
    assert "0.05" in text


def test_pq_decomposition_metrics_available_requires_finite_pq_dq_sq() -> None:
    df = _four_combo_instance_df()
    assert pq_decomposition_metrics_available(df)

    missing_sq = df.drop(columns=["sq"])
    assert not pq_decomposition_metrics_available(missing_sq)

    nan_dq = df.copy()
    nan_dq.loc[0, "dq"] = float("nan")
    assert not pq_decomposition_metrics_available(nan_dq)

    inf_dq = df.copy()
    inf_dq.loc[0, "dq"] = float("inf")
    assert not pq_decomposition_metrics_available(inf_dq)


def test_failure_mode_metrics_available_requires_finite_dq_sq() -> None:
    df = _four_combo_instance_df()
    assert failure_mode_metrics_available(df)

    missing_dq = df.drop(columns=["dq"])
    assert not failure_mode_metrics_available(missing_dq)

    inf_sq = df.copy()
    inf_sq.loc[0, "sq"] = float("inf")
    assert not failure_mode_metrics_available(inf_sq)
