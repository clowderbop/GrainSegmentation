"""Thesis-core derived tables from normalized instance metrics."""

from __future__ import annotations

import pandas as pd
import pytest

from analysis.derived_tables import (
    INPUT_CONFIGURATION_COL,
    MODEL_COL,
    PPL_RELATIVE_GAIN_COL,
    PQ_MARGIN_COL,
    WHOLE_SECTION_PQ_COL,
    WINNER_COL,
    available_ppl_relative_diagnostic_metrics,
    can_compare_yolo_and_unet_on_shared_inputs,
    headline_ranking_table,
    model_family_comparison_matrix_table,
    per_variant_winner_table,
    ppl_baseline_gain_table,
    ppl_relative_gain_matrix_table,
    thesis_ready_results_table,
    whole_section_pq_matrix_table,
    yolo_unet_paired_input_configurations,
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


def test_thesis_ready_results_table_omits_missing_optional_columns() -> None:
    minimal_keys = ("pq", "dq", "sq")
    rows = [
        {
            "producer": "yolo",
            "variant": "PPL",
            "display_name": "PPL",
            "unit": "whole",
            "source": "instance",
            **{key: PQ_SAMPLE_ROW[key] for key in minimal_keys},
        },
        {
            "producer": "unet",
            "variant": "PPL",
            "display_name": "PPL",
            "unit": "whole",
            "source": "instance",
            **{key: PQ_SAMPLE_ROW[key] for key in minimal_keys},
        },
    ]
    table = thesis_ready_results_table(pd.DataFrame(rows))

    assert WHOLE_SECTION_PQ_COL in table.columns
    assert "DQ" in table.columns
    assert "Mean precision @ IoU 0.50:0.95" not in table.columns
    assert "AJI+" not in table.columns
    assert "GT instances" not in table.columns


def test_per_variant_winner_table_picks_higher_pq_and_margin() -> None:
    table = per_variant_winner_table(_four_combo_instance_df())

    assert list(table.columns) == [
        INPUT_CONFIGURATION_COL,
        "YOLO whole-section PQ",
        "U-Net whole-section PQ",
        WINNER_COL,
        PQ_MARGIN_COL,
    ]
    assert table[INPUT_CONFIGURATION_COL].tolist() == ["PPL", "FullStack"]

    ppl = table[table[INPUT_CONFIGURATION_COL] == "PPL"].iloc[0]
    assert ppl["YOLO whole-section PQ"] == pytest.approx(0.30)
    assert ppl["U-Net whole-section PQ"] == pytest.approx(0.40)
    assert ppl[WINNER_COL] == "U-Net"
    assert ppl[PQ_MARGIN_COL] == pytest.approx(0.10)

    full = table[table[INPUT_CONFIGURATION_COL] == "FullStack"].iloc[0]
    assert full[WINNER_COL] == "YOLO"
    assert full[PQ_MARGIN_COL] == pytest.approx(0.05)


def test_ppl_baseline_gain_table_uses_same_producer_ppl_baseline() -> None:
    table = ppl_baseline_gain_table(_four_combo_instance_df())

    assert table.columns.tolist() == [
        MODEL_COL,
        INPUT_CONFIGURATION_COL,
        WHOLE_SECTION_PQ_COL,
        PPL_RELATIVE_GAIN_COL,
    ]
    yolo = table[table[MODEL_COL] == "YOLO"].set_index(INPUT_CONFIGURATION_COL)
    unet = table[table[MODEL_COL] == "U-Net"].set_index(INPUT_CONFIGURATION_COL)

    assert yolo.loc["PPL", PPL_RELATIVE_GAIN_COL] == pytest.approx(0.0)
    assert yolo.loc["FullStack", PPL_RELATIVE_GAIN_COL] == pytest.approx(0.20)
    assert unet.loc["PPL", PPL_RELATIVE_GAIN_COL] == pytest.approx(0.0)
    assert unet.loc["FullStack", PPL_RELATIVE_GAIN_COL] == pytest.approx(0.05)


def test_model_family_comparison_matrix_yolo_minus_unet_deltas() -> None:
    df = _four_combo_instance_df()
    df = df.copy()
    df.loc[df["producer"] == "yolo", "dq"] = 0.6
    df.loc[df["producer"] == "unet", "dq"] = 0.5
    df.loc[df["producer"] == "yolo", "pred_gt_instance_ratio"] = 1.1
    df.loc[df["producer"] == "unet", "pred_gt_instance_ratio"] = 0.9

    table = model_family_comparison_matrix_table(df)

    assert table.index.name == "Metric (YOLO − U-Net)"
    assert list(table.columns) == ["PPL", "FullStack"]
    assert table.loc["Whole-section PQ", "PPL"] == pytest.approx(-0.10)
    assert table.loc["Whole-section PQ", "FullStack"] == pytest.approx(0.05)
    assert table.loc["DQ", "PPL"] == pytest.approx(0.10)
    assert table.loc["Signed count bias", "PPL"] == pytest.approx(0.20)


def test_ppl_relative_gain_matrix_uses_same_producer_baseline() -> None:
    table = ppl_relative_gain_matrix_table(_four_combo_instance_df(), "pq")

    assert list(table.index) == ["YOLO", "U-Net"]
    assert list(table.columns) == ["FullStack"]
    assert table.loc["YOLO", "FullStack"] == pytest.approx(0.20)
    assert table.loc["U-Net", "FullStack"] == pytest.approx(0.05)


def _mosaic_producer_instance_df() -> pd.DataFrame:
    """YOLO on PPL only, U-Net on FullStack only — both producers, no shared input."""
    return pd.DataFrame(
        [
            _whole_row(
                producer="yolo",
                variant="PPL",
                display_name="PPL",
                pq=0.30,
            ),
            _whole_row(
                producer="unet",
                variant="PPL+AllPPX",
                display_name="FullStack",
                pq=0.45,
            ),
        ]
    )


def test_yolo_unet_paired_input_configurations_requires_shared_finite_pq() -> None:
    assert yolo_unet_paired_input_configurations(_four_combo_instance_df()) == [
        "PPL",
        "FullStack",
    ]
    assert yolo_unet_paired_input_configurations(_mosaic_producer_instance_df()) == []
    assert not can_compare_yolo_and_unet_on_shared_inputs(_mosaic_producer_instance_df())


def test_per_variant_winner_omits_unpaired_input_configurations() -> None:
    df = pd.DataFrame(
        [
            *_four_combo_instance_df().to_dict("records"),
            _whole_row(
                producer="yolo",
                variant="PPL+XPLComp",
                display_name="PPL+XPLComp",
                pq=0.55,
            ),
        ]
    )
    table = per_variant_winner_table(df)
    assert table[INPUT_CONFIGURATION_COL].tolist() == ["PPL", "FullStack"]
    assert "PPL+XPLComp" not in table[INPUT_CONFIGURATION_COL].tolist()


def test_model_family_comparison_omits_unpaired_columns() -> None:
    comparison = model_family_comparison_matrix_table(_mosaic_producer_instance_df())
    assert comparison.empty


def test_available_ppl_relative_skips_signed_count_bias_without_ratio_column() -> None:
    df = _four_combo_instance_df().drop(columns=["pred_gt_instance_ratio"])
    metrics = available_ppl_relative_diagnostic_metrics(df)
    assert "signed_count_bias" not in metrics
    assert "pq" in metrics
