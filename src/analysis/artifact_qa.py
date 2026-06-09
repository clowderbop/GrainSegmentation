"""Artifact completeness, anomaly detection, and narrative summary (QA / writing-aid tier)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis.derived_tables import (
    INPUT_CONFIGURATION_COL,
    MODEL_COL,
    PPL_RELATIVE_GAIN_COL,
    WHOLE_SECTION_PQ_COL,
    headline_ranking_table,
    ppl_baseline_gain_table,
    whole_section_instance_rows,
)
from analysis.diagnostic_derivation import (
    ABSOLUTE_GAP_COL,
    METRIC_COL,
    PQ_COMPONENT_GAP_THRESHOLD,
    patch_to_whole_gap_table,
    signed_count_bias_for_row,
)
from analysis.discover import (
    EvalRunRef,
    UltralyticsValRef,
    discover_eval_runs,
    discover_ultralytics_val,
    ultralytics_val_metrics_path,
    unet_patch_variant_dir,
    unet_whole_eval_dir,
    yolo_patch_variant_dir,
    yolo_whole_eval_dir,
)
from analysis.reporting_labels import model_display_name
from common.variants import all_variant_names

VARIANT_COL = "Variant"
PRODUCER_COL = "Producer"
ARTIFACT_COL = "Artifact"
EXPECTED_COL = "Expected"
STATUS_COL = "Status"
PATH_COL = "Path"
ANOMALY_COL = "Anomaly"
DETAIL_COL = "Detail"

WHOLE_SECTION_ARTIFACT = "whole-section instance metrics"
PATCH_ARTIFACT = "patch instance metrics"
ULTRALYTICS_VAL_ARTIFACT = "YOLO patch Ultralytics val metrics (optional)"

PATCH_GOOD_WHOLE_BAD_PQ_GAP = 0.05
STRONG_COUNT_BIAS_THRESHOLD = 0.15


def _run_key(run: EvalRunRef) -> tuple[str, str, str]:
    return (run.producer, run.variant, run.unit)


def _whole_section_metrics_path(
    grainseg_root: Path, producer: str, variant: str
) -> Path:
    if producer == "yolo":
        return yolo_whole_eval_dir(grainseg_root, variant) / "instance_metrics.json"
    return unet_whole_eval_dir(grainseg_root, variant) / "instance_metrics.json"


def _patch_variant_dir(grainseg_root: Path, producer: str, variant: str) -> Path:
    if producer == "yolo":
        return yolo_patch_variant_dir(grainseg_root, variant)
    return unet_patch_variant_dir(grainseg_root, variant)


def completeness_artifact_audit_table(
    grainseg_root: Path,
    *,
    runs: list[EvalRunRef] | None = None,
    val_refs: list[UltralyticsValRef] | None = None,
    variants: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Expected versus found eval artifacts for whole, patch, optional diagnostic, and val."""
    root = grainseg_root.resolve()
    variant_names = variants or all_variant_names()
    if runs is None:
        runs = discover_eval_runs(root, variants=variant_names)
    if val_refs is None:
        val_refs = discover_ultralytics_val(root, variants=variant_names)

    found_runs = {_run_key(run): run.instance_metrics_path for run in runs}
    found_val = {ref.variant: ref.metrics_path for ref in val_refs}

    rows: list[dict[str, object]] = []
    for variant in variant_names:
        for producer in ("yolo", "unet"):
            whole_path = _whole_section_metrics_path(root, producer, variant)
            whole_key = (producer, variant, "whole")
            rows.append(
                {
                    VARIANT_COL: variant,
                    PRODUCER_COL: producer,
                    ARTIFACT_COL: WHOLE_SECTION_ARTIFACT,
                    EXPECTED_COL: "required",
                    STATUS_COL: "found" if whole_key in found_runs else "missing",
                    PATH_COL: str(
                        found_runs[whole_key] if whole_key in found_runs else whole_path
                    ),
                }
            )
            patch_key = (producer, variant, "patch")
            if patch_key in found_runs:
                patch_status = "found"
                patch_path = found_runs[patch_key]
            else:
                patch_status = "missing"
                patch_path = _patch_variant_dir(root, producer, variant)
            rows.append(
                {
                    VARIANT_COL: variant,
                    PRODUCER_COL: producer,
                    ARTIFACT_COL: PATCH_ARTIFACT,
                    EXPECTED_COL: "required",
                    STATUS_COL: patch_status,
                    PATH_COL: str(patch_path),
                }
            )

        val_path = ultralytics_val_metrics_path(root, variant)
        rows.append(
            {
                VARIANT_COL: variant,
                PRODUCER_COL: "",
                ARTIFACT_COL: ULTRALYTICS_VAL_ARTIFACT,
                EXPECTED_COL: "optional",
                STATUS_COL: "found" if variant in found_val else "missing",
                PATH_COL: str(found_val.get(variant, val_path)),
            }
        )

    return pd.DataFrame(rows)


def _high_sq_low_dq_anomalies(df: pd.DataFrame) -> list[dict[str, object]]:
    whole = whole_section_instance_rows(df)
    rows: list[dict[str, object]] = []
    for _, row in whole.iterrows():
        dq = float(row["dq"])
        sq = float(row["sq"])
        if sq > dq + PQ_COMPONENT_GAP_THRESHOLD:
            model = model_display_name(str(row["producer"]))
            display = str(row["display_name"])
            rows.append(
                {
                    MODEL_COL: model,
                    INPUT_CONFIGURATION_COL: display,
                    ANOMALY_COL: "high SQ with low DQ",
                    DETAIL_COL: f"DQ={dq:.3f}, SQ={sq:.3f} (SQ exceeds DQ by >{PQ_COMPONENT_GAP_THRESHOLD})",
                }
            )
    return rows


def _patch_good_whole_bad_anomalies(df: pd.DataFrame) -> list[dict[str, object]]:
    gap = patch_to_whole_gap_table(df)
    if gap.empty:
        return []
    pq_gaps = gap[gap[METRIC_COL] == "Whole-section PQ"]
    rows: list[dict[str, object]] = []
    for _, row in pq_gaps.iterrows():
        absolute_gap = float(row[ABSOLUTE_GAP_COL])
        if absolute_gap > -PATCH_GOOD_WHOLE_BAD_PQ_GAP:
            continue
        patch_pq = float(row["Patch aggregate (grain-weighted)"])
        whole_pq = float(row["Whole-section value"])
        rows.append(
            {
                MODEL_COL: row[MODEL_COL],
                INPUT_CONFIGURATION_COL: row[INPUT_CONFIGURATION_COL],
                ANOMALY_COL: "patch-good / whole-bad (PQ)",
                DETAIL_COL: (
                    f"patch PQ aggregate={patch_pq:.3f}, whole-section PQ={whole_pq:.3f} "
                    f"(patch exceeds whole by ≥{PATCH_GOOD_WHOLE_BAD_PQ_GAP})"
                ),
            }
        )
    return rows


def _strong_count_bias_anomalies(df: pd.DataFrame) -> list[dict[str, object]]:
    whole = whole_section_instance_rows(df)
    rows: list[dict[str, object]] = []
    for _, row in whole.iterrows():
        ratio = row.get("pred_gt_instance_ratio")
        if ratio is None or pd.isna(ratio):
            continue
        bias = signed_count_bias_for_row(row)
        if abs(bias) < STRONG_COUNT_BIAS_THRESHOLD:
            continue
        model = model_display_name(str(row["producer"]))
        display = str(row["display_name"])
        direction = "over-prediction" if bias > 0 else "under-prediction"
        rows.append(
            {
                MODEL_COL: model,
                INPUT_CONFIGURATION_COL: display,
                ANOMALY_COL: "strong signed count bias",
                DETAIL_COL: (
                    f"signed bias={bias:+.3f} ({direction}); "
                    f"|bias| ≥ {STRONG_COUNT_BIAS_THRESHOLD}"
                ),
            }
        )
    return rows


def anomaly_report_table(df: pd.DataFrame) -> pd.DataFrame:
    """Suspicious whole-section and patch-vs-whole patterns for manual review."""
    if df.empty:
        return pd.DataFrame(
            columns=pd.Index(
                [MODEL_COL, INPUT_CONFIGURATION_COL, ANOMALY_COL, DETAIL_COL]
            )
        )
    rows: list[dict[str, object]] = []
    if "dq" in df.columns and "sq" in df.columns:
        rows.extend(_high_sq_low_dq_anomalies(df))
    rows.extend(_patch_good_whole_bad_anomalies(df))
    rows.extend(_strong_count_bias_anomalies(df))
    if not rows:
        return pd.DataFrame(
            columns=pd.Index(
                [MODEL_COL, INPUT_CONFIGURATION_COL, ANOMALY_COL, DETAIL_COL]
            )
        )
    return pd.DataFrame(rows)


def _largest_ppl_gain_line(df: pd.DataFrame) -> str | None:
    gain = ppl_baseline_gain_table(df)
    if gain.empty or PPL_RELATIVE_GAIN_COL not in gain.columns:
        return None
    best = gain.loc[gain[PPL_RELATIVE_GAIN_COL].idxmax()]
    if float(best[PPL_RELATIVE_GAIN_COL]) <= 0:
        return None
    return (
        f"- **Largest PPL-relative whole-section PQ gain:** {best[MODEL_COL]} on "
        f"{best[INPUT_CONFIGURATION_COL]} ({float(best[PPL_RELATIVE_GAIN_COL]):+.3f} vs "
        f"that producer's PPL baseline)."
    )


def _biggest_patch_to_whole_drop_line(df: pd.DataFrame) -> str | None:
    gap = patch_to_whole_gap_table(df)
    if gap.empty:
        return None
    pq_gaps = gap[gap[METRIC_COL] == "Whole-section PQ"]
    if pq_gaps.empty or pq_gaps[ABSOLUTE_GAP_COL].isna().all():
        return None
    worst = pq_gaps.loc[pq_gaps[ABSOLUTE_GAP_COL].idxmin()]
    drop = float(worst[ABSOLUTE_GAP_COL])
    if drop >= 0:
        return None
    return (
        f"- **Biggest patch-to-whole PQ drop:** {worst[MODEL_COL]} on "
        f"{worst[INPUT_CONFIGURATION_COL]} (whole − patch = {drop:+.3f}; patch aggregate "
        f"exceeds whole-section PQ)."
    )


def _strongest_count_bias_line(df: pd.DataFrame) -> str | None:
    whole = whole_section_instance_rows(df)
    best_row = None
    best_abs = -1.0
    for _, row in whole.iterrows():
        ratio = row.get("pred_gt_instance_ratio")
        if ratio is None or pd.isna(ratio):
            continue
        bias = signed_count_bias_for_row(row)
        if abs(bias) > best_abs:
            best_abs = abs(bias)
            best_row = row
    if best_row is None:
        return None
    model = model_display_name(str(best_row["producer"]))
    display = str(best_row["display_name"])
    bias = signed_count_bias_for_row(best_row)
    return (
        f"- **Strongest signed count bias:** {model} on {display} "
        f"(pred/GT ratio − 1 = {bias:+.3f})."
    )


def narrative_summary_markdown(df: pd.DataFrame) -> str:
    """First-draft result bullets from derived whole-section metrics and diagnostics."""
    lines = [
        "# Narrative summary (draft)",
        "",
        "Headline metric: whole-section PQ on held-out test eval (sliding-window). "
        "Bullets below are generated for writing aid only; verify against tables and figures.",
        "",
    ]
    whole = whole_section_instance_rows(df)
    if whole.empty:
        lines.append("_No whole-section instance metric rows were loaded._")
        return "\n".join(lines) + "\n"

    ranking = headline_ranking_table(df)
    best = ranking.iloc[0]
    lines.append(
        f"- **Best overall whole-section PQ:** {best[MODEL_COL]} on "
        f"{best[INPUT_CONFIGURATION_COL]} ({float(best[WHOLE_SECTION_PQ_COL]):.3f})."
    )

    for producer, label in (("yolo", "YOLO"), ("unet", "U-Net")):
        subset = whole[whole["producer"] == producer]
        if subset.empty:
            continue
        top = subset.loc[subset["pq"].idxmax()]
        lines.append(
            f"- **Best {label} input configuration:** {top['display_name']} "
            f"(whole-section PQ = {float(top['pq']):.3f})."
        )

    optional = [
        _largest_ppl_gain_line(df),
        _biggest_patch_to_whole_drop_line(df),
        _strongest_count_bias_line(df),
    ]
    for bullet in optional:
        if bullet:
            lines.append(bullet)

    lines.append("")
    return "\n".join(lines) + "\n"
