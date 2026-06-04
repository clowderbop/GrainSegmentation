"""Count bias and input-complexity (Pareto) diagnostics for post-eval reporting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from analysis.build_report import build_reporting_bundle
from analysis.diagnostic_derivation import (
    DIAGNOSTIC_ONLY_LABEL,
    count_error_bar_points,
    pareto_frontier_membership as pareto_membership,
    pareto_frontier_table,
    signed_count_bias_for_row,
)
from analysis.tests.test_derived_tables import _whole_row
from analysis.tests.test_load_metrics import PQ_SAMPLE_ROW, instance_metric_row
from common.variants import variant_input_image_count


def test_variant_input_image_count_from_registry() -> None:
    assert variant_input_image_count("PPL") == 1
    assert variant_input_image_count("PPL+PPXblend") == 2
    assert variant_input_image_count("PPL+AllPPX") == 7
    assert variant_input_image_count("PPLPPXblend") == 1


def test_signed_count_bias_positive_negative_and_calibrated() -> None:
    over = pd.Series({"pred_gt_instance_ratio": 1.15})
    under = pd.Series({"pred_gt_instance_ratio": 0.70})
    calibrated = pd.Series({"pred_gt_instance_ratio": 1.02})

    assert signed_count_bias_for_row(over) == pytest.approx(0.15)
    assert signed_count_bias_for_row(under) == pytest.approx(-0.30)
    assert signed_count_bias_for_row(calibrated) == pytest.approx(0.02)


def test_count_error_bar_points_omit_non_finite_ratio() -> None:
    df = pd.DataFrame(
        [
            _whole_row(
                producer="yolo",
                variant="PPL",
                display_name="PPL",
                pq=0.30,
            )
            | {"pred_gt_instance_ratio": 1.10},
            _whole_row(
                producer="yolo",
                variant="PPL+AllPPX",
                display_name="FullStack",
                pq=0.50,
            )
            | {"pred_gt_instance_ratio": float("inf")},
        ]
    )
    points = count_error_bar_points(df)

    assert len(points) == 1
    assert points.iloc[0]["Signed count bias"] == pytest.approx(0.10)
    assert points.iloc[0]["Input image count"] == 1


def test_pareto_frontier_membership_maximizes_pq_at_min_cost() -> None:
    costs = [1, 1, 2, 7]
    pqs = [0.40, 0.35, 0.45, 0.50]
    membership = pareto_membership(list(zip(costs, pqs, strict=True)))

    assert membership == [True, False, True, True]


def test_pareto_frontier_table_labels_diagnostic_only() -> None:
    df = pd.DataFrame(
        [
            _whole_row(
                producer="yolo",
                variant="PPL",
                display_name="PPL",
                pq=0.40,
            ),
            _whole_row(
                producer="yolo",
                variant="PPL+PPXblend",
                display_name="PPL+XPLComp",
                pq=0.38,
            ),
            _whole_row(
                producer="yolo",
                variant="PPL+AllPPX",
                display_name="FullStack",
                pq=0.50,
            ),
        ]
    )
    table = pareto_frontier_table(df)

    assert (table["Scope"] == DIAGNOSTIC_ONLY_LABEL).all()
    assert table.loc[table["Input configuration"] == "PPL", "Input image count"].iloc[
        0
    ] == 1
    assert table.loc[table["Input configuration"] == "FullStack", "Input image count"].iloc[
        0
    ] == 7
    frontier = table[table["On Pareto frontier"]]
    assert frontier["Input configuration"].tolist() == ["PPL", "FullStack"]


def test_instance_metric_row_includes_input_image_count() -> None:
    row = instance_metric_row(
        producer="yolo",
        variant="PPL+AllPPX",
        unit="whole",
        metrics={"pq": 0.5},
    )
    assert row["input_image_count"] == 7


def test_build_reporting_bundle_writes_count_and_pareto_outputs(
    tmp_path: Path,
) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")

    from analysis.tests.test_discover import MINIMAL_INSTANCE_METRICS, _write_json

    root = tmp_path / "GrainSeg"
    for variant, pq, ratio in (
        ("PPL", 0.30, 1.10),
        ("PPL+PPXblend", 0.45, 0.95),
        ("PPL+AllPPX", 0.50, 1.20),
    ):
        _write_json(
            root / f"eval/yolo_{variant}/instance_metrics.json",
            {
                **MINIMAL_INSTANCE_METRICS,
                "schema_version": 2,
                "variant": variant,
                "samples": [{**PQ_SAMPLE_ROW, "pq": pq, "pred_gt_instance_ratio": ratio}],
            },
        )

    out = tmp_path / "reporting"
    summary = build_reporting_bundle(root, out, render_figures=True)

    derived = summary["written"]["derived_tables"]
    assert "pareto_frontier.csv" in derived
    figures = summary["written"]["figures"]
    assert "count_error_bar_chart.png" in figures

    pareto = pd.read_csv(out / "derived" / "pareto_frontier.csv")
    assert "On Pareto frontier" in pareto.columns
    assert pareto["Scope"].iloc[0] == DIAGNOSTIC_ONLY_LABEL

    skipped_required = {item["id"] for item in summary["skipped"]}
    assert "count_error_bar_chart" not in skipped_required
    assert "pareto_frontier_table" not in skipped_required

    skipped_optional_by_id = {
        item["id"]: item["reason"] for item in summary["skipped_optional"]
    }
    if "pareto_plot" in skipped_optional_by_id:
        assert skipped_optional_by_id["pareto_plot"]
    else:
        assert "pareto_plot.png" in figures
