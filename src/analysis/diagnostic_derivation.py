"""Discussion diagnostics: PQ decomposition data and failure-mode labels."""

from __future__ import annotations

import math

import pandas as pd

from analysis.derived_tables import (
    INPUT_CONFIGURATION_COL,
    MODEL_COL,
    has_instance_metric_row_schema,
    whole_section_instance_rows,
)
from analysis.reporting_labels import MODEL_LEGEND_ORDER, model_display_name
from analysis.variant_order import thesis_ordered_display_names
from common.reporting import patch_aggregate_weighted_key
from common.variants import variant_input_image_count

PQ_DECOMPOSITION_METRICS: tuple[tuple[str, str], ...] = (
    ("PQ", "pq"),
    ("DQ", "dq"),
    ("SQ", "sq"),
)

DIAGNOSTIC_ONLY_LABEL = "Diagnostic only (not for ranking)"
SCOPE_COL = "Scope"
WHOLE_VALUE_COL = "Whole-section value"
PATCH_AGGREGATE_COL = "Patch aggregate (grain-weighted)"
ABSOLUTE_GAP_COL = "Absolute gap (whole − patch)"
RELATIVE_GAP_COL = "Relative gap"
METRIC_COL = "Metric"
STRICTNESS_DROP_COL = "F1 strictness drop (F1@IoU0.50 − F1@IoU0.75)"

PATCH_TO_WHOLE_GAP_METRICS: tuple[tuple[str, str], ...] = (
    ("Whole-section PQ", "pq"),
    ("DQ", "dq"),
    ("SQ", "sq"),
    ("F1 @ IoU 0.50", "f1_iou50"),
    ("F1 @ IoU 0.75", "f1_iou75"),
    ("Signed count bias", "signed_count_bias"),
    ("AJI+", "aji_plus"),
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


def patch_instance_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Held-out patch instance metric rows (with patch aggregates when present)."""
    if not has_instance_metric_row_schema(df):
        return df.iloc[0:0].copy()
    return df[(df["unit"] == "patch") & (df["source"] == "instance")].copy()


def _finite_value(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _whole_metric_value(row: pd.Series, metric_key: str) -> float:
    if metric_key == "signed_count_bias":
        return float(row["pred_gt_instance_ratio"]) - 1.0
    return float(row[metric_key])


def _patch_weighted_metric_value(patch_row: pd.Series, metric_key: str) -> float | None:
    source_key = (
        "pred_gt_instance_ratio" if metric_key == "signed_count_bias" else metric_key
    )
    column = patch_aggregate_weighted_key(source_key)
    if column not in patch_row.index:
        return None
    raw = patch_row[column]
    if not _finite_value(raw):
        return None
    value = float(raw)
    if metric_key == "signed_count_bias":
        return value - 1.0
    return value


def _relative_gap(whole_value: float, patch_value: float) -> float | None:
    if patch_value == 0.0:
        return None
    return (whole_value - patch_value) / patch_value


def paired_patch_whole_rows(df: pd.DataFrame) -> list[tuple[pd.Series, pd.Series]]:
    """Whole-section rows paired with patch rows for the same producer and input."""
    whole = whole_section_instance_rows(df)
    patch = patch_instance_rows(df)
    pairs: list[tuple[pd.Series, pd.Series]] = []
    for _, whole_row in whole.iterrows():
        matches = patch[
            (patch["producer"] == whole_row["producer"])
            & (patch["display_name"] == whole_row["display_name"])
        ]
        if len(matches) != 1:
            continue
        pairs.append((whole_row, matches.iloc[0]))
    return pairs


def patch_to_whole_gap_metrics_available(df: pd.DataFrame) -> bool:
    """True when at least one producer/input pair has a finite grain-weighted PQ aggregate."""
    for _whole_row, patch_row in paired_patch_whole_rows(df):
        if _patch_weighted_metric_value(patch_row, "pq") is not None:
            return True
    return False


def patch_to_whole_gap_table(df: pd.DataFrame) -> pd.DataFrame:
    """Patch-to-whole gaps per Model × input configuration for discussion diagnostics."""
    whole = whole_section_instance_rows(df)
    input_order = (
        thesis_ordered_display_names(whole["display_name"].unique())
        if not whole.empty
        else []
    )
    rows: list[dict[str, object]] = []
    for whole_row, patch_row in paired_patch_whole_rows(df):
        model = model_display_name(str(whole_row["producer"]))
        display_name = str(whole_row["display_name"])
        for metric_label, metric_key in PATCH_TO_WHOLE_GAP_METRICS:
            whole_value = _whole_metric_value(whole_row, metric_key)
            patch_value = _patch_weighted_metric_value(patch_row, metric_key)
            if patch_value is None or not _finite_value(whole_value):
                continue
            absolute_gap = whole_value - patch_value
            relative_gap = _relative_gap(whole_value, patch_value)
            rows.append(
                {
                    SCOPE_COL: DIAGNOSTIC_ONLY_LABEL,
                    MODEL_COL: model,
                    INPUT_CONFIGURATION_COL: display_name,
                    METRIC_COL: metric_label,
                    WHOLE_VALUE_COL: whole_value,
                    PATCH_AGGREGATE_COL: patch_value,
                    ABSOLUTE_GAP_COL: absolute_gap,
                    RELATIVE_GAP_COL: relative_gap,
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                SCOPE_COL,
                MODEL_COL,
                INPUT_CONFIGURATION_COL,
                METRIC_COL,
                WHOLE_VALUE_COL,
                PATCH_AGGREGATE_COL,
                ABSOLUTE_GAP_COL,
                RELATIVE_GAP_COL,
            ]
        )
    result = pd.DataFrame(rows)
    model_order = [m for m in MODEL_LEGEND_ORDER if m in set(result[MODEL_COL])]
    result[MODEL_COL] = pd.Categorical(
        result[MODEL_COL], categories=model_order, ordered=True
    )
    result[INPUT_CONFIGURATION_COL] = pd.Categorical(
        result[INPUT_CONFIGURATION_COL],
        categories=input_order,
        ordered=True,
    )
    return result.sort_values(
        [MODEL_COL, INPUT_CONFIGURATION_COL, METRIC_COL], kind="mergesort"
    )


def patch_to_whole_relative_gap_matrix_table(
    df: pd.DataFrame, metric_key: str
) -> pd.DataFrame:
    """Relative patch-to-whole gap pivot for one metric (diagnostic heatmap input)."""
    label_by_key = {key: label for label, key in PATCH_TO_WHOLE_GAP_METRICS}
    metric_label = label_by_key[metric_key]
    gap_rows = patch_to_whole_gap_table(df)
    subset = gap_rows[gap_rows[METRIC_COL] == metric_label].copy()
    if subset.empty or subset[RELATIVE_GAP_COL].isna().all():
        empty = pd.DataFrame()
        empty.index.name = MODEL_COL
        return empty
    pivot = subset.pivot(
        index=MODEL_COL, columns=INPUT_CONFIGURATION_COL, values=RELATIVE_GAP_COL
    )
    column_order = thesis_ordered_display_names(subset[INPUT_CONFIGURATION_COL])
    pivot = pivot.reindex(columns=column_order)
    row_order = [m for m in MODEL_LEGEND_ORDER if m in set(pivot.index)]
    pivot = pivot.reindex(index=row_order)
    pivot.index.name = MODEL_COL
    return pivot.astype(float)


def strictness_drop(row: pd.Series) -> float:
    """Whole-section F1@IoU0.50 minus F1@IoU0.75 (strict-vs-loose overlap sensitivity)."""
    return float(row["f1_iou50"]) - float(row["f1_iou75"])


def strictness_drop_metrics_available(df: pd.DataFrame) -> bool:
    """True when whole-section F1@50 and F1@75 are present with finite values."""
    whole = whole_section_instance_rows(df)
    if whole.empty:
        return False
    return _finite_whole_section_values(whole, ("f1_iou50", "f1_iou75"))


def strictness_drop_matrix_table(df: pd.DataFrame) -> pd.DataFrame:
    """Whole-section strictness drop pivot: rows are Model, columns are Input configuration."""
    whole = whole_section_instance_rows(df)
    whole = whole.copy()
    whole[MODEL_COL] = whole["producer"].map(model_display_name)
    whole[STRICTNESS_DROP_COL] = whole.apply(strictness_drop, axis=1)
    pivot = whole.pivot(
        index=MODEL_COL, columns="display_name", values=STRICTNESS_DROP_COL
    )
    column_order = thesis_ordered_display_names(whole["display_name"])
    pivot = pivot.reindex(columns=column_order)
    row_order = [m for m in MODEL_LEGEND_ORDER if m in set(pivot.index)]
    pivot = pivot.reindex(index=row_order)
    pivot.index.name = MODEL_COL
    return pivot.astype(float)


def precision_recall_iou75_points(df: pd.DataFrame) -> pd.DataFrame:
    """Whole-section precision and recall at IoU 0.75 for optional diagnostic map."""
    columns = [
        MODEL_COL,
        INPUT_CONFIGURATION_COL,
        "precision_iou75",
        "recall_iou75",
    ]
    whole = whole_section_instance_rows(df)
    if whole.empty or not {
        "producer",
        "display_name",
        "precision_iou75",
        "recall_iou75",
    } <= set(whole.columns):
        return pd.DataFrame(columns=columns)
    whole = whole.copy()
    whole[MODEL_COL] = whole["producer"].map(model_display_name)
    whole[INPUT_CONFIGURATION_COL] = whole["display_name"]
    return whole[columns].astype(
        {
            "precision_iou75": float,
            "recall_iou75": float,
        }
    )


def precision_recall_iou75_informative(df: pd.DataFrame) -> bool:
    """True when IoU75 precision/recall vary enough for a diagnostic scatter."""
    points = precision_recall_iou75_points(df)
    if points.empty:
        return False
    if not _finite_whole_section_values(points, ("precision_iou75", "recall_iou75")):
        return False
    precision = points["precision_iou75"].astype(float)
    recall = points["recall_iou75"].astype(float)
    if len(points) < 2:
        return False
    spread = max(precision.max() - precision.min(), recall.max() - recall.min())
    return spread > 1e-6


INPUT_IMAGE_COUNT_COL = "Input image count"
WHOLE_SECTION_PQ_LABEL = "Whole-section PQ"
PRED_GT_RATIO_COL = "Predicted/GT ratio"
SIGNED_COUNT_BIAS_COL = "Signed count bias"
ON_PARETO_FRONTIER_COL = "On Pareto frontier"


def signed_count_bias_for_row(row: pd.Series) -> float:
    """Signed count calibration: pred_gt_instance_ratio − 1."""
    return float(row["pred_gt_instance_ratio"]) - 1.0


def _finite_count_ratio(row: pd.Series) -> bool:
    ratio = row.get("pred_gt_instance_ratio")
    if ratio is None or pd.isna(ratio):
        return False
    try:
        value = float(ratio)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value)


def count_error_metrics_available(df: pd.DataFrame) -> bool:
    """True when at least one whole-section row has a finite predicted/GT ratio."""
    whole = whole_section_instance_rows(df)
    if whole.empty or "pred_gt_instance_ratio" not in whole.columns:
        return False
    return any(_finite_count_ratio(row) for _, row in whole.iterrows())


def _input_image_count_for_row(row: pd.Series) -> int:
    if "input_image_count" in row.index and pd.notna(row["input_image_count"]):
        return int(row["input_image_count"])
    return variant_input_image_count(str(row["variant"]))


def count_error_bar_points(df: pd.DataFrame) -> pd.DataFrame:
    """Whole-section rows with finite count ratio for the signed count error bar chart."""
    whole = whole_section_instance_rows(df)
    rows: list[dict[str, object]] = []
    for _, row in whole.iterrows():
        if not _finite_count_ratio(row):
            continue
        model = model_display_name(str(row["producer"]))
        display_name = str(row["display_name"])
        rows.append(
            {
                MODEL_COL: model,
                INPUT_CONFIGURATION_COL: display_name,
                INPUT_IMAGE_COUNT_COL: _input_image_count_for_row(row),
                PRED_GT_RATIO_COL: float(row["pred_gt_instance_ratio"]),
                SIGNED_COUNT_BIAS_COL: signed_count_bias_for_row(row),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                MODEL_COL,
                INPUT_CONFIGURATION_COL,
                INPUT_IMAGE_COUNT_COL,
                PRED_GT_RATIO_COL,
                SIGNED_COUNT_BIAS_COL,
            ]
        )
    result = pd.DataFrame(rows)
    model_order = [m for m in MODEL_LEGEND_ORDER if m in set(result[MODEL_COL])]
    input_order = thesis_ordered_display_names(result[INPUT_CONFIGURATION_COL])
    result[MODEL_COL] = pd.Categorical(
        result[MODEL_COL], categories=model_order, ordered=True
    )
    result[INPUT_CONFIGURATION_COL] = pd.Categorical(
        result[INPUT_CONFIGURATION_COL], categories=input_order, ordered=True
    )
    return result.sort_values([MODEL_COL, INPUT_CONFIGURATION_COL], kind="mergesort")


def _point_dominates(
    pq_a: float,
    cost_a: int,
    pq_b: float,
    cost_b: int,
) -> bool:
    """Higher PQ and lower input image count is better; strict improvement on at least one axis."""
    return pq_a >= pq_b and cost_a <= cost_b and (pq_a > pq_b or cost_a < cost_b)


def pareto_frontier_membership(
    points: list[tuple[int, float]],
) -> list[bool]:
    """Whether each (input_image_count, pq) point lies on the cost/benefit Pareto frontier."""
    membership: list[bool] = []
    for i, (cost_i, pq_i) in enumerate(points):
        dominated = False
        for j, (cost_j, pq_j) in enumerate(points):
            if i == j:
                continue
            if _point_dominates(pq_j, cost_j, pq_i, cost_i):
                dominated = True
                break
        membership.append(not dominated)
    return membership


def pareto_metrics_available(df: pd.DataFrame) -> bool:
    """True when whole-section PQ and input image count are available for Pareto views."""
    whole = whole_section_instance_rows(df)
    if whole.empty or "pq" not in whole.columns:
        return False
    if not _finite_whole_section_values(whole, ("pq",)):
        return False
    for _, row in whole.iterrows():
        try:
            _input_image_count_for_row(row)
        except (TypeError, ValueError):
            return False
    return True


def pareto_frontier_table(df: pd.DataFrame) -> pd.DataFrame:
    """Whole-section PQ versus input image count with Pareto frontier membership."""
    whole = whole_section_instance_rows(df)
    rows: list[dict[str, object]] = []
    for _, row in whole.iterrows():
        pq = float(row["pq"])
        cost = _input_image_count_for_row(row)
        rows.append(
            {
                SCOPE_COL: DIAGNOSTIC_ONLY_LABEL,
                MODEL_COL: model_display_name(str(row["producer"])),
                INPUT_CONFIGURATION_COL: str(row["display_name"]),
                INPUT_IMAGE_COUNT_COL: cost,
                WHOLE_SECTION_PQ_LABEL: pq,
                PRED_GT_RATIO_COL: (
                    float(row["pred_gt_instance_ratio"])
                    if _finite_count_ratio(row)
                    else None
                ),
                SIGNED_COUNT_BIAS_COL: (
                    signed_count_bias_for_row(row) if _finite_count_ratio(row) else None
                ),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                SCOPE_COL,
                MODEL_COL,
                INPUT_CONFIGURATION_COL,
                INPUT_IMAGE_COUNT_COL,
                WHOLE_SECTION_PQ_LABEL,
                PRED_GT_RATIO_COL,
                SIGNED_COUNT_BIAS_COL,
                ON_PARETO_FRONTIER_COL,
            ]
        )
    result = pd.DataFrame(rows)
    costs_pqs = list(
        zip(
            result[INPUT_IMAGE_COUNT_COL].astype(int).tolist(),
            result[WHOLE_SECTION_PQ_LABEL].astype(float).tolist(),
            strict=True,
        )
    )
    on_frontier = pareto_frontier_membership(costs_pqs)
    result[ON_PARETO_FRONTIER_COL] = on_frontier
    model_order = [m for m in MODEL_LEGEND_ORDER if m in set(result[MODEL_COL])]
    input_order = thesis_ordered_display_names(result[INPUT_CONFIGURATION_COL])
    result[MODEL_COL] = pd.Categorical(
        result[MODEL_COL], categories=model_order, ordered=True
    )
    result[INPUT_CONFIGURATION_COL] = pd.Categorical(
        result[INPUT_CONFIGURATION_COL], categories=input_order, ordered=True
    )
    return result.sort_values([MODEL_COL, INPUT_CONFIGURATION_COL], kind="mergesort")


def pareto_plot_informative(df: pd.DataFrame) -> bool:
    """True when a Pareto scatter adds visual value beyond the frontier table alone."""
    table = pareto_frontier_table(df)
    if len(table) < 3:
        return False
    costs = table[INPUT_IMAGE_COUNT_COL].astype(int)
    pqs = table[WHOLE_SECTION_PQ_LABEL].astype(float)
    if costs.nunique() < 2:
        return False
    if pqs.max() - pqs.min() <= 1e-6:
        return False
    return True


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
