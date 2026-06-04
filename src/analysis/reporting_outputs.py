"""Catalog of post-eval reporting bundle outputs (tables, figures, audits)."""

from __future__ import annotations

from typing import TypedDict


class ReportingOutput(TypedDict, total=False):
    id: str
    label: str
    category: str
    optional: bool
    filename_patterns: list[str]


HEADLINE_POLICY = (
    "Whole-section PQ is the headline ranking and model-comparison metric on held-out "
    "test eval (sliding-window). AJI+ may appear only as a supporting diagnostic column "
    "or delta; no output may rank by AJI+, patch metrics, AP/mAP, count ratio, or input "
    "efficiency."
)

REPORTING_OUTPUTS: list[ReportingOutput] = [
    {
        "id": "headline_ranking_table",
        "label": "Headline Ranking Table",
        "category": "thesis-core",
        "filename_patterns": ["headline_pq_ranking.csv"],
    },
    {
        "id": "whole_section_pq_matrix_heatmap",
        "label": "Whole-section PQ Matrix / Heatmap",
        "category": "thesis-core",
        "filename_patterns": [
            "whole_section_pq_matrix.csv",
            "headline_heatmap.png",
        ],
    },
    {
        "id": "thesis_ready_results_table",
        "label": "Thesis-Ready Results Table",
        "category": "thesis-core",
        "filename_patterns": ["thesis_ready_results.csv"],
    },
    {
        "id": "per_variant_winner_table",
        "label": "Per-Variant Winner Table",
        "category": "thesis-core",
        "filename_patterns": ["per_variant_winner.csv"],
    },
    {
        "id": "ppl_baseline_gain_table",
        "label": "PPL Baseline Gain Table",
        "category": "thesis-core",
        "filename_patterns": ["ppl_baseline_gain.csv"],
    },
    {
        "id": "pq_decomposition_grouped_bars",
        "label": "PQ Decomposition Grouped Bars",
        "category": "thesis-core",
        "filename_patterns": ["pq_decomposition_grouped_bars.png"],
    },
    {
        "id": "model_variant_bars",
        "label": "Model × input configuration whole-section PQ bars",
        "category": "thesis-core",
        "filename_patterns": ["model_variant_bars.png"],
    },
    {
        "id": "ppl_relative_diagnostic_heatmap",
        "label": "PPL-Relative Diagnostic Heatmap",
        "category": "discussion-diagnostic",
        "filename_patterns": [
            "ppl_relative_diagnostic_heatmaps.png",
            "ppl_delta_heatmap.png",
        ],
    },
    {
        "id": "model_family_comparison_matrix",
        "label": "Model Family Comparison Matrix",
        "category": "discussion-diagnostic",
        "filename_patterns": ["model_family_comparison_matrix.csv"],
    },
    {
        "id": "patch_to_whole_gap_table",
        "label": "Patch-to-Whole Gap Table",
        "category": "discussion-diagnostic",
        "filename_patterns": ["patch_to_whole_gap.csv"],
    },
    {
        "id": "patch_to_whole_diagnostic_heatmap",
        "label": "Patch-to-Whole Diagnostic Heatmap",
        "category": "discussion-diagnostic",
        "filename_patterns": ["patch_to_whole_diagnostic_heatmap.png"],
    },
    {
        "id": "count_error_bar_chart",
        "label": "Count Error Bar Chart",
        "category": "discussion-diagnostic",
        "filename_patterns": ["count_error_bar_chart.png"],
    },
    {
        "id": "strictness_drop_plot",
        "label": "Strictness Drop Plot",
        "category": "discussion-diagnostic",
        "filename_patterns": ["strictness_drop_plot.png"],
    },
    {
        "id": "precision_recall_diagnostic_map_iou75",
        "label": "Precision-Recall Diagnostic Map at IoU75",
        "category": "discussion-diagnostic",
        "optional": True,
        "filename_patterns": ["precision_recall_diagnostic_map_iou75.png"],
    },
    {
        "id": "pareto_frontier_table",
        "label": "Pareto Frontier Table",
        "category": "discussion-diagnostic",
        "filename_patterns": ["pareto_frontier.csv"],
    },
    {
        "id": "pareto_plot",
        "label": "Pareto Plot",
        "category": "discussion-diagnostic",
        "optional": True,
        "filename_patterns": ["pareto_plot.png"],
    },
    {
        "id": "failure_mode_classification",
        "label": "Failure-Mode Classification",
        "category": "discussion-diagnostic",
        "filename_patterns": [
            "failure_mode_classification.csv",
            "failure_mode_classification_rules.md",
        ],
    },
    {
        "id": "completeness_artifact_audit",
        "label": "Completeness / Artifact Audit",
        "category": "qa-writing",
        "filename_patterns": ["completeness_artifact_audit.csv"],
    },
    {
        "id": "outlier_anomaly_report",
        "label": "Outlier / Anomaly Report",
        "category": "qa-writing",
        "filename_patterns": ["outlier_anomaly_report.csv"],
    },
    {
        "id": "narrative_summary_generator",
        "label": "Narrative Summary Generator",
        "category": "qa-writing",
        "filename_patterns": ["narrative_summary.md"],
    },
    {
        "id": "instance_metrics_table",
        "label": "Normalized instance metrics table",
        "category": "supporting",
        "filename_patterns": ["instance_metrics.csv"],
    },
    {
        "id": "ultralytics_val_table",
        "label": "YOLO patch Ultralytics val table",
        "category": "supporting",
        "filename_patterns": ["ultralytics_val.csv"],
    },
    {
        "id": "yolo_patch_val_panel",
        "label": "Supporting YOLO patch Ultralytics val mAP panel",
        "category": "supporting",
        "filename_patterns": ["yolo_patch_val_panel.png"],
    },
]


def optional_reporting_outputs() -> list[ReportingOutput]:
    return [item for item in REPORTING_OUTPUTS if item.get("optional")]


def required_reporting_outputs() -> list[ReportingOutput]:
    return [item for item in REPORTING_OUTPUTS if not item.get("optional")]
