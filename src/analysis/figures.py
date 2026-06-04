"""Thesis figure set for post-eval reporting (requires optional analysis deps)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from common.variants import variant_display_names_in_thesis_order

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "yolo": "YOLO",
    "unet": "U-Net",
}
MODEL_AXIS_LABEL = "Model"
MODEL_LEGEND_ORDER = ("YOLO", "U-Net")
HEADLINE_PQ_COLUMNS = ("pq", "dq", "sq")


class HeadlineFigureError(ValueError):
    """Raised when instance metrics lack required whole-section PQ headline fields."""


HEADLINE_PQ_TITLE = "Headline: whole-section PQ (held-out test)"
PQ_DIAGNOSTIC_DQ_TITLE = "PQ diagnostic: DQ (whole-section test)"
PQ_DIAGNOSTIC_SQ_TITLE = "PQ diagnostic: SQ (whole-section test)"
MODEL_VARIANT_BARS_TITLE = "Headline: whole-section PQ by input configuration"
PPL_DELTA_PQ_TITLE = "Headline: whole-section PQ gain vs PPL baseline"
ULTRALYTICS_VAL_PANEL_TITLE = (
    "Supporting: YOLO patch Ultralytics val mAP@0.5 (not whole-section SAHI)"
)


def model_display_name(producer: str) -> str:
    """Thesis-facing label for a producer family on figures (data keeps `producer`)."""
    return MODEL_DISPLAY_NAMES.get(producer, producer)


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


def _ordered_display_names(df: pd.DataFrame) -> list[str]:
    thesis_order = list(variant_display_names_in_thesis_order())
    present = [name for name in thesis_order if name in set(df["display_name"])]
    extra = sorted(set(df["display_name"]) - set(present))
    return present + extra


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


def figure_headline_heatmap(df: pd.DataFrame, path: Path) -> None:
    _require_plotting()
    import matplotlib.pyplot as plt
    import seaborn as sns

    whole = require_headline_pq_table(df)
    display_order = _ordered_display_names(whole)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5), constrained_layout=True)
    for ax, metric, title in zip(
        axes,
        HEADLINE_PQ_COLUMNS,
        (HEADLINE_PQ_TITLE, PQ_DIAGNOSTIC_DQ_TITLE, PQ_DIAGNOSTIC_SQ_TITLE),
        strict=True,
    ):
        pivot = whole.pivot(index="producer", columns="display_name", values=metric)
        pivot = pivot.reindex(columns=display_order)
        pivot = _rename_model_index(pivot)
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
        categories=_ordered_display_names(whole),
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


def figure_ppl_delta_heatmap(df: pd.DataFrame, path: Path) -> None:
    _require_plotting()
    import matplotlib.pyplot as plt
    import seaborn as sns

    whole = require_headline_pq_table(df)
    ppl_name = "PPL"
    baseline = (
        whole[whole["display_name"] == ppl_name].set_index("producer")["pq"].to_dict()
    )
    rows: list[dict[str, object]] = []
    for _, row in whole.iterrows():
        if row["display_name"] == ppl_name:
            continue
        base = baseline.get(row["producer"])
        if base is None:
            continue
        rows.append(
            {
                "producer": row["producer"],
                "display_name": row["display_name"],
                "delta_pq": float(row["pq"]) - float(base),
            }
        )
    if not rows:
        raise HeadlineFigureError(
            "PPL baseline whole-section PQ is missing for one or more producers"
        )
    delta = pd.DataFrame(rows)
    pivot = delta.pivot(index="display_name", columns="producer", values="delta_pq")
    pivot = pivot.reindex(_ordered_display_names(delta))
    pivot = _rename_model_columns(pivot)
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


def figure_ultralytics_val_panel(val_df: pd.DataFrame, path: Path) -> None:
    _require_plotting()
    import matplotlib.pyplot as plt
    import seaborn as sns

    if val_df.empty or "val_seg_map50" not in val_df.columns:
        return
    panel = val_df.copy()
    panel["display_name"] = pd.Categorical(
        panel["display_name"],
        categories=_ordered_display_names(panel),
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


def render_all_figures(
    instance_df: pd.DataFrame,
    val_df: pd.DataFrame,
    figures_dir: Path,
) -> list[str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("headline_heatmap.png", figure_headline_heatmap, instance_df),
        ("model_variant_bars.png", figure_model_variant_bars, instance_df),
        ("ppl_delta_heatmap.png", figure_ppl_delta_heatmap, instance_df),
        ("yolo_patch_val_panel.png", figure_ultralytics_val_panel, val_df),
    ]
    written: list[str] = []
    for filename, renderer, data in specs:
        out = figures_dir / filename
        renderer(data, out)
        if out.is_file():
            written.append(filename)
    return written
