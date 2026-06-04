"""Thesis figure set for post-eval reporting (requires optional analysis deps)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis.diagnostic_derivation import (
    pq_decomposition_long_table,
    pq_decomposition_metrics_available,
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
        ("yolo_patch_val_panel.png", figure_ultralytics_val_panel, val_df),
    ]
    written: list[str] = []
    for filename, renderer, data in specs:
        out = figures_dir / filename
        renderer(data, out)
        if out.is_file():
            written.append(filename)
    return written
