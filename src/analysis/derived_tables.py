"""Thesis-core derived tables from normalized instance metric rows."""

from __future__ import annotations

import pandas as pd

from analysis.reporting_labels import MODEL_LEGEND_ORDER, model_display_name
from common.variants import variant_display_names_in_thesis_order

MODEL_COL = "Model"
INPUT_CONFIGURATION_COL = "Input configuration"
WHOLE_SECTION_PQ_COL = "Whole-section PQ"
RANK_COL = "Rank"

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


def whole_section_instance_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Held-out whole-section instance metric rows only."""
    return df[(df["unit"] == "whole") & (df["source"] == "instance")].copy()


def _thesis_input_order(df: pd.DataFrame) -> list[str]:
    thesis_order = list(variant_display_names_in_thesis_order())
    present = [name for name in thesis_order if name in set(df["display_name"])]
    extra = sorted(set(df["display_name"]) - set(present))
    return present + extra


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
    column_order = _thesis_input_order(whole)
    pivot = pivot.reindex(columns=column_order)
    pivot.index = pivot.index.map(model_display_name)
    pivot.index.name = MODEL_COL
    return pivot.astype(float)


def whole_section_pq_matrix_table(df: pd.DataFrame) -> pd.DataFrame:
    """Whole-section PQ pivot: rows are Model, columns are Input configuration (thesis order)."""
    return whole_section_metric_matrix_table(df, "pq")


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
        categories=_thesis_input_order(whole),
        ordered=True,
    )
    whole = whole.sort_values([MODEL_COL, INPUT_CONFIGURATION_COL], kind="mergesort")
    columns: dict[str, object] = {
        MODEL_COL: whole[MODEL_COL].astype(str).tolist(),
        INPUT_CONFIGURATION_COL: whole[INPUT_CONFIGURATION_COL].astype(str).tolist(),
    }
    for label, key in THESIS_READY_METRIC_COLUMNS:
        columns[label] = whole[key].astype(float).tolist()
    return pd.DataFrame(columns)
