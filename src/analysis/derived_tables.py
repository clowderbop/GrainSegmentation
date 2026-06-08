"""Thesis-core derived tables from normalized instance metric rows."""

from __future__ import annotations

import pandas as pd

from analysis.reporting_labels import MODEL_LEGEND_ORDER, model_display_name
from analysis.variant_order import thesis_ordered_display_names

MODEL_COL = "Model"
INPUT_CONFIGURATION_COL = "Input configuration"
WHOLE_SECTION_PQ_COL = "Whole-section PQ"
RANK_COL = "Rank"
WINNER_COL = "Winner"
PQ_MARGIN_COL = "PQ margin"
YOLO_PQ_COL = "YOLO whole-section PQ"
UNET_PQ_COL = "U-Net whole-section PQ"
PPL_BASELINE_DISPLAY_NAME = "PPL"
PPL_RELATIVE_GAIN_COL = "PPL-relative whole-section PQ gain"
MODEL_FAMILY_DELTA_INDEX = "Metric (YOLO − U-Net)"

PPL_RELATIVE_DIAGNOSTIC_METRICS: tuple[tuple[str, str], ...] = (
    ("Whole-section PQ", "pq"),
    ("DQ", "dq"),
    ("SQ", "sq"),
    ("Signed count bias", "signed_count_bias"),
    ("AJI+", "aji_plus"),
)

MODEL_FAMILY_COMPARISON_METRICS: tuple[tuple[str, str], ...] = (
    ("Whole-section PQ", "pq"),
    ("DQ", "dq"),
    ("SQ", "sq"),
    ("Signed count bias", "signed_count_bias"),
    ("AJI+", "aji_plus"),
)

THESIS_READY_METRIC_COLUMNS: tuple[tuple[str, str], ...] = (
    (WHOLE_SECTION_PQ_COL, "pq"),
    ("DQ", "dq"),
    ("SQ", "sq"),
    ("F1 @ IoU 0.50", "f1_iou50"),
    ("F1 @ IoU 0.75", "f1_iou75"),
    ("Precision @ IoU 0.50", "precision_iou50"),
    ("Recall @ IoU 0.50", "recall_iou50"),
    ("Precision @ IoU 0.75", "precision_iou75"),
    ("Recall @ IoU 0.75", "recall_iou75"),
    ("Mean precision @ IoU 0.50:0.95", "mP_iou50_95"),
    ("Mean recall @ IoU 0.50:0.95", "mR_iou50_95"),
    ("Mean F1 @ IoU 0.50:0.95", "mF1_iou50_95"),
    ("GT instances", "gt_instance_count"),
    ("Predicted instances", "pred_instance_count"),
    ("Predicted/GT ratio", "pred_gt_instance_ratio"),
    ("AJI+", "aji_plus"),
)


INSTANCE_ROW_INDEX_COLUMNS: tuple[str, ...] = ("unit", "source")


def has_instance_metric_row_schema(df: pd.DataFrame) -> bool:
    """True when normalized instance metric rows can be filtered by unit and source."""
    return all(column in df.columns for column in INSTANCE_ROW_INDEX_COLUMNS)


def whole_section_instance_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Held-out whole-section instance metric rows only."""
    if not has_instance_metric_row_schema(df):
        return df.iloc[0:0].copy()
    return df[(df["unit"] == "whole") & (df["source"] == "instance")].copy()


def yolo_unet_paired_input_configurations(df: pd.DataFrame) -> list[str]:
    """Input configurations with finite whole-section PQ for both YOLO and U-Net."""
    whole = whole_section_instance_rows(df)
    paired: list[str] = []
    for display_name in thesis_ordered_display_names(whole["display_name"].unique()):
        subset = whole[whole["display_name"] == display_name]
        yolo_rows = subset[subset["producer"] == "yolo"]["pq"]
        unet_rows = subset[subset["producer"] == "unet"]["pq"]
        if len(yolo_rows) != 1 or len(unet_rows) != 1:
            continue
        yolo_pq = float(yolo_rows.iloc[0])
        unet_pq = float(unet_rows.iloc[0])
        if pd.isna(yolo_pq) or pd.isna(unet_pq):
            continue
        paired.append(display_name)
    return paired


def can_compare_yolo_and_unet_on_shared_inputs(df: pd.DataFrame) -> bool:
    """True when at least one input configuration has both producers' whole-section PQ."""
    return bool(yolo_unet_paired_input_configurations(df))


def headline_ranking_table(df: pd.DataFrame) -> pd.DataFrame:
    """All producer × input-configuration pairs ranked by whole-section PQ (descending)."""
    whole = whole_section_instance_rows(df)
    ranked = whole.sort_values("pq", ascending=False, kind="mergesort")
    return pd.DataFrame(
        {
            RANK_COL: range(1, len(ranked) + 1),
            MODEL_COL: ranked["producer"].map(model_display_name).tolist(),
            INPUT_CONFIGURATION_COL: ranked["display_name"].tolist(),
            WHOLE_SECTION_PQ_COL: ranked["pq"].astype(float).tolist(),
        }
    )


def whole_section_metric_matrix_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Pivot one whole-section metric: rows are Model, columns are Input configuration."""
    whole = whole_section_instance_rows(df)
    pivot = whole.pivot(index="producer", columns="display_name", values=metric)
    column_order = thesis_ordered_display_names(whole["display_name"])
    pivot = pivot.reindex(columns=column_order)
    pivot.index = pivot.index.map(model_display_name)
    row_order = [m for m in MODEL_LEGEND_ORDER if m in set(pivot.index)]
    pivot = pivot.reindex(index=row_order)
    pivot.index.name = MODEL_COL
    return pivot.astype(float)


def whole_section_pq_matrix_table(df: pd.DataFrame) -> pd.DataFrame:
    """Whole-section PQ pivot: rows are Model, columns are Input configuration (thesis order)."""
    return whole_section_metric_matrix_table(df, "pq")


def signed_count_bias(series: pd.Series) -> pd.Series:
    """Signed over/under-prediction: pred_gt_instance_ratio − 1."""
    return series.astype(float) - 1.0


def whole_section_metric_available(df: pd.DataFrame, metric: str) -> bool:
    """True when a whole-section metric column is present with finite values."""
    whole = whole_section_instance_rows(df)
    if metric == "signed_count_bias":
        if "pred_gt_instance_ratio" not in whole.columns:
            return False
        return bool(whole["pred_gt_instance_ratio"].notna().all())
    if metric not in whole.columns:
        return False
    return bool(whole[metric].notna().all())


def _whole_section_metric_for_matrix(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric == "signed_count_bias":
        whole = whole_section_instance_rows(df)
        whole = whole.copy()
        whole["signed_count_bias"] = signed_count_bias(whole["pred_gt_instance_ratio"])
        pivot = whole.pivot(
            index="producer", columns="display_name", values="signed_count_bias"
        )
        column_order = thesis_ordered_display_names(whole["display_name"])
        pivot = pivot.reindex(columns=column_order)
        pivot.index = pivot.index.map(model_display_name)
        row_order = [m for m in MODEL_LEGEND_ORDER if m in set(pivot.index)]
        pivot = pivot.reindex(index=row_order)
        pivot.index.name = MODEL_COL
        return pivot.astype(float)
    return whole_section_metric_matrix_table(df, metric)


def ppl_relative_gain_matrix_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per Model: input-configuration gain vs that producer's PPL baseline for one metric."""
    matrix = _whole_section_metric_for_matrix(df, metric)
    if PPL_BASELINE_DISPLAY_NAME not in matrix.columns:
        msg = f"PPL baseline column missing for PPL-relative {metric!r} heatmap"
        raise ValueError(msg)
    baseline = matrix[PPL_BASELINE_DISPLAY_NAME]
    delta = matrix.drop(columns=[PPL_BASELINE_DISPLAY_NAME]).sub(baseline, axis=0)
    column_order = [c for c in matrix.columns if c != PPL_BASELINE_DISPLAY_NAME]
    row_order = [m for m in MODEL_LEGEND_ORDER if m in delta.index]
    return delta.reindex(index=row_order, columns=column_order)


def available_ppl_relative_diagnostic_metrics(df: pd.DataFrame) -> list[str]:
    """Metric keys present with finite whole-section values for PPL-relative heatmaps."""
    return [
        metric
        for _label, metric in PPL_RELATIVE_DIAGNOSTIC_METRICS
        if whole_section_metric_available(df, metric)
    ]


def model_family_comparison_matrix_table(df: pd.DataFrame) -> pd.DataFrame:
    """YOLO minus U-Net deltas per paired input configuration for discussion diagnostics."""
    paired = yolo_unet_paired_input_configurations(df)
    if not paired:
        empty = pd.DataFrame()
        empty.index.name = MODEL_FAMILY_DELTA_INDEX
        return empty
    deltas: dict[str, pd.Series] = {}
    for label, metric in MODEL_FAMILY_COMPARISON_METRICS:
        if not whole_section_metric_available(df, metric):
            continue
        matrix = _whole_section_metric_for_matrix(df, metric)
        deltas[label] = matrix.loc["YOLO", paired] - matrix.loc["U-Net", paired]
    result = pd.DataFrame(deltas).T
    result.index.name = MODEL_FAMILY_DELTA_INDEX
    return result.astype(float)


def ppl_baseline_gain_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per producer: whole-section PQ gain vs that producer's PPL baseline."""
    whole = whole_section_instance_rows(df)
    rows: list[dict[str, object]] = []
    for producer in sorted(whole["producer"].unique()):
        producer_rows = whole[whole["producer"] == producer]
        baseline_rows = producer_rows[
            producer_rows["display_name"] == PPL_BASELINE_DISPLAY_NAME
        ]
        if baseline_rows.empty:
            continue
        baseline_pq = float(baseline_rows.iloc[0]["pq"])
        model = model_display_name(producer)
        for _, row in producer_rows.sort_values(
            "display_name", kind="mergesort"
        ).iterrows():
            pq = float(row["pq"])
            rows.append(
                {
                    MODEL_COL: model,
                    INPUT_CONFIGURATION_COL: row["display_name"],
                    WHOLE_SECTION_PQ_COL: pq,
                    PPL_RELATIVE_GAIN_COL: pq - baseline_pq,
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result[MODEL_COL] = pd.Categorical(
        result[MODEL_COL],
        categories=[m for m in MODEL_LEGEND_ORDER if m in set(result[MODEL_COL])],
        ordered=True,
    )
    result[INPUT_CONFIGURATION_COL] = pd.Categorical(
        result[INPUT_CONFIGURATION_COL],
        categories=thesis_ordered_display_names(whole["display_name"]),
        ordered=True,
    )
    return result.sort_values([MODEL_COL, INPUT_CONFIGURATION_COL], kind="mergesort")


def _winner_from_pq_pair(yolo_pq: float, unet_pq: float) -> tuple[str, float]:
    if yolo_pq > unet_pq:
        return "YOLO", yolo_pq - unet_pq
    if unet_pq > yolo_pq:
        return "U-Net", unet_pq - yolo_pq
    return "Tie", 0.0


def per_variant_winner_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per paired input configuration: YOLO vs U-Net by whole-section PQ and margin."""
    matrix = whole_section_pq_matrix_table(df)
    rows: list[dict[str, object]] = []
    for display_name in yolo_unet_paired_input_configurations(df):
        yolo_pq = float(matrix.loc["YOLO", display_name])
        unet_pq = float(matrix.loc["U-Net", display_name])
        winner, margin = _winner_from_pq_pair(yolo_pq, unet_pq)
        rows.append(
            {
                INPUT_CONFIGURATION_COL: display_name,
                YOLO_PQ_COL: yolo_pq,
                UNET_PQ_COL: unet_pq,
                WINNER_COL: winner,
                PQ_MARGIN_COL: margin,
            }
        )
    return pd.DataFrame(rows)


def thesis_ready_results_table(df: pd.DataFrame) -> pd.DataFrame:
    """Thesis-facing results with headline PQ, PQ diagnostics, and AJI+ as supporting only."""
    whole = whole_section_instance_rows(df)
    whole = whole.copy()
    whole[MODEL_COL] = whole["producer"].map(model_display_name)
    whole[INPUT_CONFIGURATION_COL] = whole["display_name"]
    whole[MODEL_COL] = pd.Categorical(
        whole[MODEL_COL],
        categories=[m for m in MODEL_LEGEND_ORDER if m in set(whole[MODEL_COL])],
        ordered=True,
    )
    whole[INPUT_CONFIGURATION_COL] = pd.Categorical(
        whole[INPUT_CONFIGURATION_COL],
        categories=thesis_ordered_display_names(whole["display_name"]),
        ordered=True,
    )
    whole = whole.sort_values([MODEL_COL, INPUT_CONFIGURATION_COL], kind="mergesort")
    columns: dict[str, object] = {
        MODEL_COL: whole[MODEL_COL].astype(str).tolist(),
        INPUT_CONFIGURATION_COL: whole[INPUT_CONFIGURATION_COL].astype(str).tolist(),
    }
    for label, key in THESIS_READY_METRIC_COLUMNS:
        if key not in whole.columns:
            continue
        columns[label] = pd.to_numeric(whole[key], errors="coerce").tolist()
    return pd.DataFrame(columns)
