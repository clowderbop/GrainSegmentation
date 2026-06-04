"""PQ decomposition and failure-mode diagnostic derivation."""

from __future__ import annotations

import pandas as pd
import pytest

from analysis.diagnostic_derivation import (
    ABSOLUTE_GAP_COL,
    DIAGNOSTIC_ONLY_LABEL,
    FAILURE_MODE_LABEL_COL,
    RELATIVE_GAP_COL,
    failure_mode_classification_table,
    failure_mode_metrics_available,
    failure_mode_rules_markdown,
    patch_to_whole_gap_metrics_available,
    patch_to_whole_gap_table,
    patch_to_whole_relative_gap_matrix_table,
    pq_decomposition_long_table,
    pq_decomposition_metrics_available,
    precision_recall_iou75_informative,
    strictness_drop,
    strictness_drop_metrics_available,
    strictness_drop_matrix_table,
)
from analysis.derived_tables import INPUT_CONFIGURATION_COL, MODEL_COL
from analysis.tests.test_derived_tables import _four_combo_instance_df, _whole_row
from common.instance_metric_bundle import INSTANCE_METRIC_BUNDLE_KEYS
from common.reporting import patch_aggregate_weighted_key


def _patch_row(
    *,
    producer: str,
    variant: str,
    display_name: str,
    weighted_overrides: dict[str, float],
) -> dict[str, object]:
    base = {key: 0.0 for key in INSTANCE_METRIC_BUNDLE_KEYS}
    extras = {
        "n_patches": 2,
        "n_empty_gt": 0,
    }
    for key, value in weighted_overrides.items():
        extras[patch_aggregate_weighted_key(key)] = value
    return {
        "producer": producer,
        "variant": variant,
        "display_name": display_name,
        "unit": "patch",
        "source": "instance",
        **base,
        **extras,
    }


def _paired_whole_patch_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _whole_row(
                producer="yolo",
                variant="PPL",
                display_name="PPL",
                pq=0.50,
            )
            | {
                "f1_iou50": 0.80,
                "f1_iou75": 0.50,
            },
            _patch_row(
                producer="yolo",
                variant="PPL",
                display_name="PPL",
                weighted_overrides={"pq": 0.40, "pred_gt_instance_ratio": 1.10},
            ),
        ]
    )


def test_patch_to_whole_helpers_tolerate_schema_less_empty_dataframe() -> None:
    empty = pd.DataFrame()

    assert not patch_to_whole_gap_metrics_available(empty)
    assert not strictness_drop_metrics_available(empty)
    assert not precision_recall_iou75_informative(empty)
    assert patch_to_whole_gap_table(empty).empty


def test_patch_to_whole_gap_table_absolute_and_relative_gaps() -> None:
    table = patch_to_whole_gap_table(_paired_whole_patch_df())

    pq = table[table["Metric"] == "Whole-section PQ"].iloc[0]
    assert pq[MODEL_COL] == "YOLO"
    assert pq[INPUT_CONFIGURATION_COL] == "PPL"
    assert pq["Whole-section value"] == pytest.approx(0.50)
    assert pq["Patch aggregate (grain-weighted)"] == pytest.approx(0.40)
    assert pq[ABSOLUTE_GAP_COL] == pytest.approx(0.10)
    assert pq[RELATIVE_GAP_COL] == pytest.approx(0.25)

    count = table[table["Metric"] == "Signed count bias"].iloc[0]
    assert count["Whole-section value"] == pytest.approx(-0.20)
    assert count["Patch aggregate (grain-weighted)"] == pytest.approx(0.10)
    assert count[ABSOLUTE_GAP_COL] == pytest.approx(-0.30)


def test_patch_to_whole_gap_table_includes_diagnostic_only_label() -> None:
    table = patch_to_whole_gap_table(_paired_whole_patch_df())

    assert table["Scope"].iloc[0] == DIAGNOSTIC_ONLY_LABEL


def test_strictness_drop_is_f1_iou50_minus_f1_iou75() -> None:
    row = _whole_row(producer="yolo", variant="PPL", display_name="PPL", pq=0.3)
    row["f1_iou50"] = 0.80
    row["f1_iou75"] = 0.50
    assert strictness_drop(pd.Series(row)) == pytest.approx(0.30)


def test_strictness_drop_matrix_orders_model_and_input() -> None:
    matrix = strictness_drop_matrix_table(_paired_whole_patch_df())

    assert matrix.index.name == MODEL_COL
    assert matrix.loc["YOLO", "PPL"] == pytest.approx(0.30)


def test_patch_to_whole_gap_metrics_available_requires_patch_aggregates() -> None:
    assert patch_to_whole_gap_metrics_available(_paired_whole_patch_df())

    whole_only = pd.DataFrame(
        [
            _whole_row(
                producer="yolo",
                variant="PPL",
                display_name="PPL",
                pq=0.50,
            )
        ]
    )
    assert not patch_to_whole_gap_metrics_available(whole_only)


def test_patch_to_whole_relative_gap_matrix_for_pq() -> None:
    matrix = patch_to_whole_relative_gap_matrix_table(_paired_whole_patch_df(), "pq")

    assert matrix.loc["YOLO", "PPL"] == pytest.approx(0.25)


def test_precision_recall_iou75_informative_requires_spread() -> None:
    varied = pd.DataFrame(
        [
            _whole_row(
                producer="yolo",
                variant="PPL",
                display_name="PPL",
                pq=0.3,
            )
            | {"precision_iou75": 0.2, "recall_iou75": 0.3},
            _whole_row(
                producer="yolo",
                variant="PPL+AllPPX",
                display_name="FullStack",
                pq=0.5,
            )
            | {"precision_iou75": 0.8, "recall_iou75": 0.7},
        ]
    )
    assert precision_recall_iou75_informative(varied)

    flat = pd.DataFrame(
        [
            _whole_row(
                producer="yolo",
                variant="PPL",
                display_name="PPL",
                pq=0.3,
            )
            | {"precision_iou75": 0.5, "recall_iou75": 0.5},
        ]
    )
    assert not precision_recall_iou75_informative(flat)


def test_strictness_drop_metrics_available_requires_finite_f1_columns() -> None:
    assert strictness_drop_metrics_available(_paired_whole_patch_df())

    missing = _paired_whole_patch_df().drop(columns=["f1_iou75"])
    assert not strictness_drop_metrics_available(missing)


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
