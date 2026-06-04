"""Thesis figure set for post-eval reporting (requires optional analysis deps)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis.diagnostic_derivation import (
    INPUT_IMAGE_COUNT_COL,
    SIGNED_COUNT_BIAS_COL,
    WHOLE_SECTION_PQ_LABEL,
    count_error_bar_points,
    count_error_metrics_available,
    pareto_frontier_table,
    pareto_plot_informative,
    patch_to_whole_gap_metrics_available,
    patch_to_whole_relative_gap_matrix_table,
    pq_decomposition_long_table,
    pq_decomposition_metrics_available,
    precision_recall_iou75_informative,
    precision_recall_iou75_points,
    strictness_drop_matrix_table,
    strictness_drop_metrics_available,
)
from analysis.derived_tables import (
    INPUT_CONFIGURATION_COL,
    MODEL_COL,
    PPL_RELATIVE_DIAGNOSTIC_METRICS,
    available_ppl_relative_diagnostic_metrics,
    ppl_relative_gain_matrix_table,
    whole_section_metric_matrix_table,
    whole_section_pq_matrix_table,
)
from analysis.reporting_labels import (
    MODEL_AXIS_LABEL,
    MODEL_DISPLAY_NAMES,
    MODEL_LEGEND_ORDER,
    model_display_name,
)
from analysis.variant_order import thesis_ordered_display_names

HEADLINE_PQ_COLUMNS = ("pq", "dq", "sq")


class HeadlineFigureError(ValueError):
    """Raised when instance metrics lack required whole-section PQ headline fields."""


HEADLINE_PQ_TITLE = "Headline: whole-section PQ (held-out test)"
PQ_DIAGNOSTIC_DQ_TITLE = "PQ diagnostic: DQ (whole-section test)"
PQ_DIAGNOSTIC_SQ_TITLE = "PQ diagnostic: SQ (whole-section test)"
MODEL_VARIANT_BARS_TITLE = "Headline: whole-section PQ by input configuration"
PQ_DECOMPOSITION_GROUPED_BARS_TITLE = (
    "PQ decomposition: whole-section PQ, DQ, and SQ (held-out test)"
)
PPL_DELTA_PQ_TITLE = "Headline: whole-section PQ gain vs PPL baseline"
PPL_RELATIVE_DIAGNOSTIC_HEATMAP_TITLE = (
    "PPL-relative diagnostic gain vs PPL baseline (whole-section test)"
)
ULTRALYTICS_VAL_PANEL_TITLE = (
    "Supporting: YOLO patch Ultralytics val mAP@0.5 (not whole-section SAHI)"
)
PATCH_TO_WHOLE_DIAGNOSTIC_HEATMAP_TITLE = (
    "Diagnostic: patch-to-whole relative gap (grain-weighted patch aggregate)"
)
STRICTNESS_DROP_PLOT_TITLE = (
    "Diagnostic: F1 strictness drop (F1@IoU0.50 − F1@IoU0.75, whole-section test)"
)
PRECISION_RECALL_IOU75_TITLE = (
    "Diagnostic: precision vs recall at IoU 0.75 (whole-section test)"
)
COUNT_ERROR_BAR_CHART_TITLE = (
    "Diagnostic: signed count error (pred/GT ratio − 1, whole-section test)"
)
PARETO_PLOT_TITLE = (
    "Diagnostic: whole-section PQ vs input image count (Pareto frontier)"
)

PATCH_TO_WHOLE_HEATMAP_METRICS: tuple[tuple[str, str], ...] = (
    ("Whole-section PQ", "pq"),
    ("DQ", "dq"),
    ("SQ", "sq"),
    ("AJI+", "aji_plus"),
)


def _rename_model_index(pivot: pd.DataFrame) -> pd.DataFrame:
    return pivot.rename(index=MODEL_DISPLAY_NAMES)


def _rename_model_columns(pivot: pd.DataFrame) -> pd.DataFrame:
    return pivot.rename(columns=MODEL_DISPLAY_NAMES)


def _require_plotting() -> None:
    try:
        import matplotlib  # noqa: F401
        import seaborn  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Plotting requires the analysis optional extra: "
            "uv sync --group analysis"
        ) from exc


def _whole_headline_table(df: pd.DataFrame) -> pd.DataFrame:
    whole = df[(df["unit"] == "whole") & (df["source"] == "instance")].copy()
    return whole.sort_values(["producer", "display_name"])


def require_headline_pq_table(df: pd.DataFrame) -> pd.DataFrame:
    """Whole-section instance rows with headline PQ and required PQ diagnostics."""
    whole = _whole_headline_table(df)
    if whole.empty:
        raise HeadlineFigureError(
            "no whole-section instance rows for headline figures; "
            "run held-out whole-section eval before post-eval reporting"
        )
    missing = [col for col in HEADLINE_PQ_COLUMNS if col not in whole.columns]
    if missing:
        raise HeadlineFigureError(
            "headline figures require whole-section PQ bundle columns "
            f"{list(HEADLINE_PQ_COLUMNS)!r}; missing {missing!r} - "
            "regenerate eval artifacts under the PQ bundle policy"
        )
    if whole[list(HEADLINE_PQ_COLUMNS)].isna().any().any():
        raise HeadlineFigureError(
            "headline figures require finite whole-section PQ, DQ, and SQ values"
        )
    return whole


def _headline_metric_matrix(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric == "pq":
        return whole_section_pq_matrix_table(df)
    return whole_section_metric_matrix_table(df, metric)


def figure_headline_heatmap(df: pd.DataFrame, path: Path) -> None:
    _require_plotting()
    import matplotlib.pyplot as plt
    import seaborn as sns

    require_headline_pq_table(df)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5), constrained_layout=True)
    for ax, metric, title in zip(
        axes,
        HEADLINE_PQ_COLUMNS,
        (HEADLINE_PQ_TITLE, PQ_DIAGNOSTIC_DQ_TITLE, PQ_DIAGNOSTIC_SQ_TITLE),
        strict=True,
    ):
        pivot = _headline_metric_matrix(df, metric)
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Input configuration")
        ax.set_ylabel(MODEL_AXIS_LABEL)
        ax.tick_params(axis="x", labelrotation=0)
        plt.setp(ax.get_xticklabels(), ha="center")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_model_variant_bars(df: pd.DataFrame, path: Path) -> None:
    _require_plotting()
    import matplotlib.pyplot as plt
    import seaborn as sns

    whole = require_headline_pq_table(df).copy()
    whole["model"] = whole["producer"].map(MODEL_DISPLAY_NAMES)
    whole["display_name"] = pd.Categorical(
        whole["display_name"],
        categories=thesis_ordered_display_names(whole["display_name"]),
        ordered=True,
    )
    whole["model"] = pd.Categorical(
        whole["model"],
        categories=[m for m in MODEL_LEGEND_ORDER if m in set(whole["model"])],
        ordered=True,
    )
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    sns.barplot(
        data=whole,
        x="display_name",
        y="pq",
        hue="model",
        ax=ax,
    )
    ax.set_title(MODEL_VARIANT_BARS_TITLE)
    ax.set_xlabel("Input configuration")
    ax.set_ylabel("PQ")
    ax.legend(title=MODEL_AXIS_LABEL)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _ppl_relative_pq_gain_matrix(df: pd.DataFrame) -> pd.DataFrame:
    try:
        return ppl_relative_gain_matrix_table(df, "pq")
    except ValueError as exc:
        raise HeadlineFigureError(str(exc)) from exc


def figure_pq_decomposition_grouped_bars(df: pd.DataFrame, path: Path) -> None:
    _require_plotting()
    import matplotlib.pyplot as plt
    import seaborn as sns

    if not pq_decomposition_metrics_available(df):
        return
    long = pq_decomposition_long_table(df)
    if long.empty:
        return
    long = long.copy()
    long[INPUT_CONFIGURATION_COL] = pd.Categorical(
        long[INPUT_CONFIGURATION_COL],
        categories=thesis_ordered_display_names(long[INPUT_CONFIGURATION_COL]),
        ordered=True,
    )
    long[MODEL_COL] = pd.Categorical(
        long[MODEL_COL],
        categories=[m for m in MODEL_LEGEND_ORDER if m in set(long[MODEL_COL])],
        ordered=True,
    )
    long["Metric"] = pd.Categorical(
        long["Metric"],
        categories=["PQ", "DQ", "SQ"],
        ordered=True,
    )
    g = sns.catplot(
        data=long,
        kind="bar",
        x=INPUT_CONFIGURATION_COL,
        y="Value",
        hue="Metric",
        col=MODEL_COL,
        height=4,
        aspect=1.1,
        legend_out=False,
    )
    g.set_axis_labels("Input configuration", "Score")
    g.set_titles("{col_name}")
    for ax in g.axes.flat:
        ax.tick_params(axis="x", labelrotation=0)
        plt.setp(ax.get_xticklabels(), ha="center")
    g.fig.suptitle(PQ_DECOMPOSITION_GROUPED_BARS_TITLE, y=1.02)
    g.fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(g.fig)


def figure_ppl_delta_heatmap(df: pd.DataFrame, path: Path) -> None:
    _require_plotting()
    import matplotlib.pyplot as plt
    import seaborn as sns

    require_headline_pq_table(df)
    pivot = _ppl_relative_pq_gain_matrix(df).T
    if pivot.empty:
        return
    pivot = pivot.reindex(thesis_ordered_display_names(pivot.index))
    fig, ax = plt.subplots(figsize=(8, 3.5), constrained_layout=True)
    sns.heatmap(
        pivot,
        annot=True,
        fmt="+.3f",
        cmap="RdBu_r",
        center=0.0,
        ax=ax,
    )
    ax.set_title(PPL_DELTA_PQ_TITLE)
    ax.set_xlabel(MODEL_AXIS_LABEL)
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelrotation=0)
    plt.setp(ax.get_xticklabels(), ha="center")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_ppl_relative_diagnostic_heatmaps(df: pd.DataFrame, path: Path) -> None:
    _require_plotting()
    import matplotlib.pyplot as plt
    import seaborn as sns

    require_headline_pq_table(df)
    metrics = available_ppl_relative_diagnostic_metrics(df)
    if not metrics:
        return
    label_by_metric = {key: label for label, key in PPL_RELATIVE_DIAGNOSTIC_METRICS}
    panels: list[tuple[str, pd.DataFrame]] = []
    for metric in metrics:
        try:
            pivot = ppl_relative_gain_matrix_table(df, metric).T
        except ValueError:
            continue
        pivot = pivot.reindex(thesis_ordered_display_names(pivot.index))
        if pivot.empty:
            continue
        panels.append((label_by_metric[metric], pivot))

    if not panels:
        return

    ncols = min(3, len(panels))
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.5 * ncols, 3.5 * nrows),
        constrained_layout=True,
        squeeze=False,
    )
    for ax, (title, pivot) in zip(axes.flat, panels, strict=False):
        sns.heatmap(
            pivot,
            annot=True,
            fmt="+.3f",
            cmap="RdBu_r",
            center=0.0,
            ax=ax,
        )
        ax.set_title(f"{title} vs PPL baseline")
        ax.set_xlabel(MODEL_AXIS_LABEL)
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelrotation=0)
        plt.setp(ax.get_xticklabels(), ha="center")
    for ax in axes.flat[len(panels) :]:
        ax.set_visible(False)
    fig.suptitle(PPL_RELATIVE_DIAGNOSTIC_HEATMAP_TITLE)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _available_patch_to_whole_heatmap_panels(
    df: pd.DataFrame,
) -> list[tuple[str, pd.DataFrame]]:
    panels: list[tuple[str, pd.DataFrame]] = []
    for label, metric_key in PATCH_TO_WHOLE_HEATMAP_METRICS:
        pivot = patch_to_whole_relative_gap_matrix_table(df, metric_key)
        if pivot.empty:
            continue
        panels.append((label, pivot))
    return panels


def figure_patch_to_whole_diagnostic_heatmap(df: pd.DataFrame, path: Path) -> None:
    _require_plotting()
    import matplotlib.pyplot as plt
    import seaborn as sns

    if not patch_to_whole_gap_metrics_available(df):
        return
    panels = _available_patch_to_whole_heatmap_panels(df)
    if not panels:
        return

    ncols = min(2, len(panels))
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.0 * ncols, 3.5 * nrows),
        constrained_layout=True,
        squeeze=False,
    )
    for ax, (title, pivot) in zip(axes.flat, panels, strict=False):
        sns.heatmap(
            pivot,
            annot=True,
            fmt="+.2f",
            cmap="RdBu_r",
            center=0.0,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("Input configuration")
        ax.set_ylabel(MODEL_AXIS_LABEL)
        ax.tick_params(axis="x", labelrotation=0)
        plt.setp(ax.get_xticklabels(), ha="center")
    for ax in axes.flat[len(panels) :]:
        ax.set_visible(False)
    fig.suptitle(PATCH_TO_WHOLE_DIAGNOSTIC_HEATMAP_TITLE)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_strictness_drop_plot(df: pd.DataFrame, path: Path) -> None:
    _require_plotting()
    import matplotlib.pyplot as plt
    import seaborn as sns

    if not strictness_drop_metrics_available(df):
        return
    pivot = strictness_drop_matrix_table(df)
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 3.5), constrained_layout=True)
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax)
    ax.set_title(STRICTNESS_DROP_PLOT_TITLE)
    ax.set_xlabel("Input configuration")
    ax.set_ylabel(MODEL_AXIS_LABEL)
    ax.tick_params(axis="x", labelrotation=0)
    plt.setp(ax.get_xticklabels(), ha="center")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_precision_recall_diagnostic_map_iou75(df: pd.DataFrame, path: Path) -> None:
    _require_plotting()
    import matplotlib.pyplot as plt
    import seaborn as sns

    if not precision_recall_iou75_informative(df):
        return
    points = precision_recall_iou75_points(df)
    points = points.copy()
    points[MODEL_COL] = pd.Categorical(
        points[MODEL_COL],
        categories=[m for m in MODEL_LEGEND_ORDER if m in set(points[MODEL_COL])],
        ordered=True,
    )
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    sns.scatterplot(
        data=points,
        x="recall_iou75",
        y="precision_iou75",
        hue=MODEL_COL,
        style=INPUT_CONFIGURATION_COL,
        ax=ax,
    )
    ax.set_title(PRECISION_RECALL_IOU75_TITLE)
    ax.set_xlabel("Recall @ IoU 0.75")
    ax.set_ylabel("Precision @ IoU 0.75")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_count_error_bar_chart(df: pd.DataFrame, path: Path) -> None:
    _require_plotting()
    import matplotlib.pyplot as plt
    import seaborn as sns

    if not count_error_metrics_available(df):
        return
    points = count_error_bar_points(df)
    if points.empty:
        return
    points = points.copy()
    points[INPUT_CONFIGURATION_COL] = pd.Categorical(
        points[INPUT_CONFIGURATION_COL],
        categories=thesis_ordered_display_names(points[INPUT_CONFIGURATION_COL]),
        ordered=True,
    )
    points[MODEL_COL] = pd.Categorical(
        points[MODEL_COL],
        categories=[m for m in MODEL_LEGEND_ORDER if m in set(points[MODEL_COL])],
        ordered=True,
    )
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    sns.barplot(
        data=points,
        x=INPUT_CONFIGURATION_COL,
        y=SIGNED_COUNT_BIAS_COL,
        hue=MODEL_COL,
        ax=ax,
    )
    ax.axhline(0.0, color="0.4", linewidth=0.8, linestyle="--")
    ax.set_title(COUNT_ERROR_BAR_CHART_TITLE)
    ax.set_xlabel("Input configuration")
    ax.set_ylabel("Signed count bias (pred/GT − 1)")
    ax.legend(title=MODEL_AXIS_LABEL)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_pareto_plot(df: pd.DataFrame, path: Path) -> None:
    _require_plotting()
    import matplotlib.pyplot as plt
    import seaborn as sns

    if not pareto_plot_informative(df):
        return
    table = pareto_frontier_table(df)
    if table.empty:
        return
    table = table.copy()
    table[MODEL_COL] = pd.Categorical(
        table[MODEL_COL],
        categories=[m for m in MODEL_LEGEND_ORDER if m in set(table[MODEL_COL])],
        ordered=True,
    )
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    sns.scatterplot(
        data=table,
        x=INPUT_IMAGE_COUNT_COL,
        y=WHOLE_SECTION_PQ_LABEL,
        hue=MODEL_COL,
        style="On Pareto frontier",
        ax=ax,
    )
    frontier = table[table["On Pareto frontier"]].sort_values(
        INPUT_IMAGE_COUNT_COL, kind="mergesort"
    )
    if len(frontier) >= 2:
        ax.plot(
            frontier[INPUT_IMAGE_COUNT_COL],
            frontier[WHOLE_SECTION_PQ_LABEL],
            color="0.35",
            linestyle="--",
            linewidth=1.0,
            label="Pareto frontier",
        )
    ax.set_title(PARETO_PLOT_TITLE)
    ax.set_xlabel("Input image count")
    ax.set_ylabel("Whole-section PQ")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_ultralytics_val_panel(val_df: pd.DataFrame, path: Path) -> None:
    _require_plotting()
    import matplotlib.pyplot as plt
    import seaborn as sns

    if val_df.empty or "val_seg_map50" not in val_df.columns:
        return
    panel = val_df.copy()
    panel["display_name"] = pd.Categorical(
        panel["display_name"],
        categories=thesis_ordered_display_names(panel["display_name"]),
        ordered=True,
    )
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    sns.barplot(
        data=panel,
        x="display_name",
        y="val_seg_map50",
        color="steelblue",
        ax=ax,
    )
    ax.set_title(ULTRALYTICS_VAL_PANEL_TITLE)
    ax.set_xlabel("Input configuration")
    ax.set_ylabel("seg mAP@0.5")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


FIGURE_BUNDLE_FILENAMES: tuple[str, ...] = (
    "headline_heatmap.png",
    "model_variant_bars.png",
    "pq_decomposition_grouped_bars.png",
    "ppl_delta_heatmap.png",
    "ppl_relative_diagnostic_heatmaps.png",
    "patch_to_whole_diagnostic_heatmap.png",
    "count_error_bar_chart.png",
    "strictness_drop_plot.png",
    "precision_recall_diagnostic_map_iou75.png",
    "pareto_plot.png",
    "yolo_patch_val_panel.png",
)


def render_all_figures(
    instance_df: pd.DataFrame,
    val_df: pd.DataFrame,
    figures_dir: Path,
) -> list[str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("headline_heatmap.png", figure_headline_heatmap, instance_df),
        ("model_variant_bars.png", figure_model_variant_bars, instance_df),
        (
            "pq_decomposition_grouped_bars.png",
            figure_pq_decomposition_grouped_bars,
            instance_df,
        ),
        ("ppl_delta_heatmap.png", figure_ppl_delta_heatmap, instance_df),
        (
            "ppl_relative_diagnostic_heatmaps.png",
            figure_ppl_relative_diagnostic_heatmaps,
            instance_df,
        ),
        (
            "patch_to_whole_diagnostic_heatmap.png",
            figure_patch_to_whole_diagnostic_heatmap,
            instance_df,
        ),
        ("count_error_bar_chart.png", figure_count_error_bar_chart, instance_df),
        ("strictness_drop_plot.png", figure_strictness_drop_plot, instance_df),
        (
            "precision_recall_diagnostic_map_iou75.png",
            figure_precision_recall_diagnostic_map_iou75,
            instance_df,
        ),
        ("pareto_plot.png", figure_pareto_plot, instance_df),
        ("yolo_patch_val_panel.png", figure_ultralytics_val_panel, val_df),
    ]
    written: list[str] = []
    for filename, renderer, data in specs:
        out = figures_dir / filename
        renderer(data, out)
        if out.is_file():
            written.append(filename)
    return written
