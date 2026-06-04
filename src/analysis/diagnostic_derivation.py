"""Discussion diagnostics: PQ decomposition data and failure-mode labels."""

from __future__ import annotations

import math

import pandas as pd

from analysis.derived_tables import INPUT_CONFIGURATION_COL, MODEL_COL, whole_section_instance_rows
from analysis.reporting_labels import MODEL_LEGEND_ORDER, model_display_name
from analysis.variant_order import thesis_ordered_display_names

PQ_DECOMPOSITION_METRICS: tuple[tuple[str, str], ...] = (
    ("PQ", "pq"),
    ("DQ", "dq"),
    ("SQ", "sq"),
)

FAILURE_MODE_LABEL_COL = "Failure mode labels"
PQ_COMPONENT_GAP_THRESHOLD = 0.05
COUNT_OVERPREDICT_THRESHOLD = 0.05
COUNT_UNDERPREDICT_THRESHOLD = -0.05

FAILURE_MODE_RULES: tuple[tuple[str, str], ...] = (
    (
        "detection-limited",
        f"DQ + {PQ_COMPONENT_GAP_THRESHOLD} < SQ (detection quality is the weaker PQ component)",
    ),
    (
        "mask-quality-limited",
        f"SQ + {PQ_COMPONENT_GAP_THRESHOLD} < DQ (matched-mask quality is the weaker PQ component)",
    ),
    (
        "balanced PQ components",
        f"|DQ − SQ| ≤ {PQ_COMPONENT_GAP_THRESHOLD} (neither PQ component clearly dominates)",
    ),
    (
        "overpredicting",
        f"pred_gt_instance_ratio − 1 > {COUNT_OVERPREDICT_THRESHOLD}",
    ),
    (
        "underpredicting",
        f"pred_gt_instance_ratio − 1 < {COUNT_UNDERPREDICT_THRESHOLD}",
    ),
    (
        "calibrated count",
        f"{COUNT_UNDERPREDICT_THRESHOLD} ≤ pred_gt_instance_ratio − 1 ≤ {COUNT_OVERPREDICT_THRESHOLD}",
    ),
)


def _finite_whole_section_values(whole: pd.DataFrame, keys: tuple[str, ...]) -> bool:
    """True when every key is present with finite values on all whole-section rows."""
    for key in keys:
        if key not in whole.columns:
            return False
        numeric = pd.to_numeric(whole[key], errors="coerce")
        if not numeric.notna().all():
            return False
        if not numeric.map(math.isfinite).all():
            return False
    return True


def failure_mode_metrics_available(df: pd.DataFrame) -> bool:
    """True when whole-section DQ and SQ are present with finite values."""
    whole = whole_section_instance_rows(df)
    if whole.empty:
        return False
    return _finite_whole_section_values(whole, ("dq", "sq"))


def _pq_component_labels(dq: float, sq: float) -> list[str]:
    gap = float(dq) - float(sq)
    if abs(gap) <= PQ_COMPONENT_GAP_THRESHOLD:
        return ["balanced PQ components"]
    if gap < 0:
        return ["detection-limited"]
    return ["mask-quality-limited"]


def _count_calibration_label(ratio: float | None) -> list[str]:
    if ratio is None or pd.isna(ratio):
        return []
    bias = float(ratio) - 1.0
    if bias > COUNT_OVERPREDICT_THRESHOLD:
        return ["overpredicting"]
    if bias < COUNT_UNDERPREDICT_THRESHOLD:
        return ["underpredicting"]
    return ["calibrated count"]


def failure_mode_labels_for_row(row: pd.Series) -> str:
    """Semicolon-separated failure-mode labels for one whole-section metric row."""
    labels = _pq_component_labels(float(row["dq"]), float(row["sq"]))
    labels.extend(_count_calibration_label(row.get("pred_gt_instance_ratio")))
    return "; ".join(labels)


def failure_mode_classification_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per Model × input configuration failure-mode labels from bundle fields."""
    whole = whole_section_instance_rows(df)
    whole = whole.copy()
    whole[MODEL_COL] = whole["producer"].map(model_display_name)
    whole[INPUT_CONFIGURATION_COL] = whole["display_name"]
    model_order = [m for m in MODEL_LEGEND_ORDER if m in set(whole[MODEL_COL])]
    input_order = thesis_ordered_display_names(whole[INPUT_CONFIGURATION_COL])
    rows: list[dict[str, object]] = []
    for model in model_order:
        for display_name in input_order:
            subset = whole[
                (whole[MODEL_COL] == model)
                & (whole[INPUT_CONFIGURATION_COL] == display_name)
            ]
            if subset.empty:
                continue
            row = subset.iloc[0]
            rows.append(
                {
                    MODEL_COL: model,
                    INPUT_CONFIGURATION_COL: display_name,
                    FAILURE_MODE_LABEL_COL: failure_mode_labels_for_row(row),
                }
            )
    return pd.DataFrame(rows)


def failure_mode_rules_markdown() -> str:
    """Documented predicates for failure-mode classification."""
    lines = [
        "# Failure-mode classification rules",
        "",
        "Labels are assigned from whole-section **instance metric bundle** fields only.",
        "Multiple labels may apply to one row; they are joined with `; `.",
        "",
        "## Predicates",
        "",
    ]
    for label, predicate in FAILURE_MODE_RULES:
        lines.append(f"- **{label}**: {predicate}")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- PQ component rules compare DQ and SQ on the same row.",
            "- Count calibration requires a finite `pred_gt_instance_ratio`; "
            "when missing, count labels are omitted.",
            "- These labels are report-only diagnostics, not ranking criteria.",
        ]
    )
    return "\n".join(lines) + "\n"


def pq_decomposition_metrics_available(df: pd.DataFrame) -> bool:
    """True when whole-section PQ, DQ, and SQ are present with finite values."""
    whole = whole_section_instance_rows(df)
    if whole.empty:
        return False
    keys = tuple(key for _label, key in PQ_DECOMPOSITION_METRICS)
    return _finite_whole_section_values(whole, keys)


def pq_decomposition_long_table(df: pd.DataFrame) -> pd.DataFrame:
    """Long-format PQ/DQ/SQ rows ordered by Model, Input configuration, Metric."""
    whole = whole_section_instance_rows(df)
    whole = whole.copy()
    whole[MODEL_COL] = whole["producer"].map(model_display_name)
    whole[INPUT_CONFIGURATION_COL] = whole["display_name"]
    model_order = [m for m in MODEL_LEGEND_ORDER if m in set(whole[MODEL_COL])]
    input_order = thesis_ordered_display_names(whole[INPUT_CONFIGURATION_COL])
    rows: list[dict[str, object]] = []
    for model in model_order:
        for display_name in input_order:
            subset = whole[
                (whole[MODEL_COL] == model)
                & (whole[INPUT_CONFIGURATION_COL] == display_name)
            ]
            if subset.empty:
                continue
            row = subset.iloc[0]
            for metric_label, key in PQ_DECOMPOSITION_METRICS:
                rows.append(
                    {
                        MODEL_COL: model,
                        INPUT_CONFIGURATION_COL: display_name,
                        "Metric": metric_label,
                        "Value": float(row[key]),
                    }
                )
    return pd.DataFrame(rows)
