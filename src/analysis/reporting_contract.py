"""Wave 1 post-eval reporting contract (approved, deferred, cut, headline policy)."""

from __future__ import annotations

from typing import Any, TypedDict


class ReportingOutput(TypedDict, total=False):
    id: str
    label: str
    tier: str
    optional: bool
    filename_patterns: list[str]


HEADLINE_POLICY = (
    "Whole-section PQ is the headline ranking and model-comparison metric on held-out "
    "test eval (sliding-window). AJI+ may appear only as a supporting diagnostic column "
    "or delta; no output may rank by AJI+, patch metrics, AP/mAP, count ratio, or input "
    "efficiency."
)

WAVE1_APPROVED_OUTPUTS: list[ReportingOutput] = [
    # Thesis-core tier
    {
        "id": "headline_ranking_table",
        "label": "Headline Ranking Table",
        "tier": "thesis-core",
        "filename_patterns": ["headline_pq_ranking.csv"],
    },
    {
        "id": "whole_section_pq_matrix_heatmap",
        "label": "Whole-section PQ Matrix / Heatmap",
        "tier": "thesis-core",
        "filename_patterns": [
            "whole_section_pq_matrix.csv",
            "headline_heatmap.png",
        ],
    },
    {
        "id": "thesis_ready_results_table",
        "label": "Thesis-Ready Results Table",
        "tier": "thesis-core",
        "filename_patterns": ["thesis_ready_results.csv"],
    },
    {
        "id": "per_variant_winner_table",
        "label": "Per-Variant Winner Table",
        "tier": "thesis-core",
        "filename_patterns": ["per_variant_winner.csv"],
    },
    {
        "id": "ppl_baseline_gain_table",
        "label": "PPL Baseline Gain Table",
        "tier": "thesis-core",
        "filename_patterns": ["ppl_baseline_gain.csv"],
    },
    {
        "id": "pq_decomposition_grouped_bars",
        "label": "PQ Decomposition Grouped Bars",
        "tier": "thesis-core",
        "filename_patterns": ["pq_decomposition_grouped_bars.png"],
    },
    # Discussion-diagnostic tier
    {
        "id": "ppl_relative_diagnostic_heatmap",
        "label": "PPL-Relative Diagnostic Heatmap",
        "tier": "discussion-diagnostic",
        "filename_patterns": [
            "ppl_relative_diagnostic_heatmaps.png",
            "ppl_delta_heatmap.png",
        ],
    },
    {
        "id": "model_family_comparison_matrix",
        "label": "Model Family Comparison Matrix",
        "tier": "discussion-diagnostic",
        "filename_patterns": ["model_family_comparison_matrix.csv"],
    },
    {
        "id": "patch_to_whole_gap_table",
        "label": "Patch-to-Whole Gap Table",
        "tier": "discussion-diagnostic",
        "filename_patterns": ["patch_to_whole_gap.csv"],
    },
    {
        "id": "patch_to_whole_diagnostic_heatmap",
        "label": "Patch-to-Whole Diagnostic Heatmap",
        "tier": "discussion-diagnostic",
        "filename_patterns": ["patch_to_whole_diagnostic_heatmap.png"],
    },
    {
        "id": "count_error_bar_chart",
        "label": "Count Error Bar Chart",
        "tier": "discussion-diagnostic",
        "filename_patterns": ["count_error_bar_chart.png"],
    },
    {
        "id": "strictness_drop_plot",
        "label": "Strictness Drop Plot",
        "tier": "discussion-diagnostic",
        "filename_patterns": ["strictness_drop_plot.png"],
    },
    {
        "id": "precision_recall_diagnostic_map_iou75",
        "label": "Precision-Recall Diagnostic Map at IoU75",
        "tier": "discussion-diagnostic",
        "optional": True,
        "filename_patterns": ["precision_recall_diagnostic_map_iou75.png"],
    },
    {
        "id": "pareto_frontier_table",
        "label": "Pareto Frontier Table",
        "tier": "discussion-diagnostic",
        "filename_patterns": ["pareto_frontier.csv"],
    },
    {
        "id": "pareto_plot",
        "label": "Pareto Plot",
        "tier": "discussion-diagnostic",
        "optional": True,
        "filename_patterns": ["pareto_plot.png"],
    },
    {
        "id": "failure_mode_classification",
        "label": "Failure-Mode Classification",
        "tier": "discussion-diagnostic",
        "filename_patterns": [
            "failure_mode_classification.csv",
            "failure_mode_classification_rules.md",
        ],
    },
    # QA / writing-aid tier
    {
        "id": "completeness_artifact_audit",
        "label": "Completeness / Artifact Audit",
        "tier": "qa-writing",
    },
    {
        "id": "outlier_anomaly_report",
        "label": "Outlier / Anomaly Report",
        "tier": "qa-writing",
    },
    {
        "id": "narrative_summary_generator",
        "label": "Narrative Summary Generator",
        "tier": "qa-writing",
    },
    # Existing supporting derived table (pre-expansion; not headline evidence)
    {
        "id": "instance_metrics_table",
        "label": "Normalized instance metrics table",
        "tier": "supporting",
        "filename_patterns": ["instance_metrics.csv"],
    },
    {
        "id": "ultralytics_val_table",
        "label": "YOLO patch Ultralytics val table",
        "tier": "supporting",
        "filename_patterns": ["ultralytics_val.csv"],
    },
    {
        "id": "yolo_patch_val_panel",
        "label": "Supporting YOLO patch Ultralytics val mAP panel",
        "tier": "supporting",
        "filename_patterns": ["yolo_patch_val_panel.png"],
    },
]

WAVE2_DEFERRED_OUTPUTS: list[ReportingOutput] = [
    {
        "id": "strictness_curve",
        "label": "Strictness Curve",
        "tier": "wave2-deferred",
        "filename_patterns": ["strictness_curve"],
    },
    {
        "id": "f1_precision_recall_over_iou_line_plots",
        "label": "F1 / Precision / Recall over IoU Line Plots",
        "tier": "wave2-deferred",
        "filename_patterns": ["precision_recall_over_iou", "f1_over_iou"],
    },
    {
        "id": "sensitivity_summary_across_iou_thresholds",
        "label": "Sensitivity Summary Across IoU Thresholds",
        "tier": "wave2-deferred",
        "filename_patterns": ["sensitivity_summary"],
    },
    {
        "id": "error_composition_bars",
        "label": "Error Composition Bars",
        "tier": "wave2-deferred",
        "filename_patterns": ["error_composition"],
    },
]

CUT_OUTPUTS: list[ReportingOutput] = [
    {
        "id": "dq_vs_sq_scatter",
        "label": "DQ vs SQ Scatter",
        "tier": "cut",
        "filename_patterns": ["dq_vs_sq_scatter"],
    },
    {
        "id": "pq_vs_input_count_frontier_plot",
        "label": "PQ vs Input Count Frontier Plot",
        "tier": "cut",
        "filename_patterns": ["pq_vs_input_count_frontier"],
    },
    {
        "id": "instance_count_calibration_plot",
        "label": "Instance Count Calibration Plot",
        "tier": "cut",
        "filename_patterns": ["instance_count_calibration"],
    },
    {
        "id": "producer_delta_plot",
        "label": "Producer Delta Plot",
        "tier": "cut",
        "filename_patterns": ["producer_delta"],
    },
    {
        "id": "best_per_variant_ranking_bar",
        "label": "Best-Per-Variant Ranking Bar",
        "tier": "cut",
        "filename_patterns": ["best_per_variant_ranking"],
    },
    {
        "id": "patch_to_whole_degradation_heatmap",
        "label": "Patch-to-Whole Degradation Heatmap",
        "tier": "cut",
        "filename_patterns": ["patch_to_whole_degradation"],
    },
    {
        "id": "count_error_heatmap",
        "label": "Count Error Heatmap",
        "tier": "cut",
        "filename_patterns": ["count_error_heatmap"],
    },
    {
        "id": "input_efficiency_plot",
        "label": "Input Efficiency Plot",
        "tier": "cut",
        "filename_patterns": ["input_efficiency"],
    },
    {
        "id": "input_efficiency_table",
        "label": "Input Efficiency Table",
        "tier": "cut",
        "filename_patterns": ["input_efficiency"],
    },
    {
        "id": "metric_correlation_matrix",
        "label": "Metric Correlation Matrix",
        "tier": "cut",
        "filename_patterns": ["metric_correlation"],
    },
    {
        "id": "rank_stability_strip",
        "label": "Rank Stability Strip",
        "tier": "cut",
        "filename_patterns": ["rank_stability"],
    },
    {
        "id": "rank_stability_table",
        "label": "Rank Stability Table",
        "tier": "cut",
        "filename_patterns": ["rank_stability"],
    },
    {
        "id": "yolo_val_vs_whole_pq_scatter",
        "label": "YOLO Val vs Whole PQ Scatter",
        "tier": "cut",
        "filename_patterns": ["yolo_val_vs_whole_pq"],
    },
    {
        "id": "ap_map_additions",
        "label": "AP/mAP additions",
        "tier": "cut",
        "filename_patterns": ["ap_map", "map50_bars", "map_curve"],
    },
    {
        "id": "pq_decomposition_slopegraph",
        "label": "PQ Decomposition Slopegraph",
        "tier": "cut",
        "filename_patterns": ["pq_decomposition_slopegraph"],
    },
    {
        "id": "whole_vs_patch_gap_plot",
        "label": "Whole vs Patch Gap Plot",
        "tier": "cut",
        "filename_patterns": ["whole_vs_patch_gap"],
    },
]

FORBIDDEN_REPORTING_OUTPUTS: list[ReportingOutput] = [
    *WAVE2_DEFERRED_OUTPUTS,
    *CUT_OUTPUTS,
]


def reporting_contract_metadata() -> dict[str, Any]:
    """Return approved/deferred/cut lists for bundle metadata."""
    return {
        "wave1_approved": WAVE1_APPROVED_OUTPUTS,
        "wave2_deferred": WAVE2_DEFERRED_OUTPUTS,
        "cut_outputs": CUT_OUTPUTS,
    }


def optional_wave1_outputs() -> list[ReportingOutput]:
    return [item for item in WAVE1_APPROVED_OUTPUTS if item.get("optional")]


def wave1_output_by_id(output_id: str) -> ReportingOutput | None:
    for item in WAVE1_APPROVED_OUTPUTS:
        if item["id"] == output_id:
            return item
    return None


def required_wave1_outputs() -> list[ReportingOutput]:
    return [item for item in WAVE1_APPROVED_OUTPUTS if not item.get("optional")]
