"""Bar charts of instance-segmentation metrics from evaluate JSON outputs."""

from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

from common.reporting import INSTANCE_METRIC_KEYS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot quantitative ablation results from evaluation JSON files.",
    )
    parser.add_argument("--json-files", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output-plot", required=True)
    args = parser.parse_args()
    if len(args.json_files) != len(args.labels):
        parser.error("Number of json files must match number of labels.")
    return args


def compute_ci(data, confidence: float = 0.95) -> float:
    import scipy.stats as st

    a = 1.0 * np.array(data)
    n = len(a)
    if n < 2:
        return 0.0
    se = st.sem(a)
    return float(se * st.t.ppf((1 + confidence) / 2.0, n - 1))


def per_sample_metrics_from_eval_json(data: dict) -> dict[str, dict]:
    samples = data.get("samples")
    if not isinstance(samples, list):
        raise ValueError('Evaluation JSON must include a "samples" list')
    out: dict[str, dict] = {}
    for row in samples:
        if not isinstance(row, dict):
            continue
        sid = str(row.get("sample_id", ""))
        out[sid] = {k: float(row[k]) for k in INSTANCE_METRIC_KEYS if k in row}
    return out


def _load_quantitative_metrics(
    json_files: list[str],
    metrics_to_plot: dict[str, str],
) -> tuple[dict[str, list[float]], dict[str, list[float]], list[int]]:
    means = {metric_name: [] for metric_name in metrics_to_plot}
    cis = {metric_name: [] for metric_name in metrics_to_plot}
    sample_counts: list[int] = []

    for jf in json_files:
        with open(jf, "r") as f:
            data = json.load(f)

        per_sample = per_sample_metrics_from_eval_json(data)
        sample_keys = list(per_sample.keys())
        sample_counts.append(len(sample_keys))

        for metric_name, metric_key in metrics_to_plot.items():
            vals = [
                float(per_sample[sk][metric_key])
                for sk in sample_keys
                if sk in per_sample
                and metric_key in per_sample[sk]
                and not np.isnan(float(per_sample[sk][metric_key]))
            ]
            if not vals:
                means[metric_name].append(float("nan"))
                cis[metric_name].append(float("nan"))
                continue
            means[metric_name].append(float(np.mean(vals)))
            cis[metric_name].append(float(compute_ci(vals)))

    return means, cis, sample_counts


def generate_quantitative_plot(
    json_files: list[str],
    labels: list[str],
    output_path: str,
) -> None:
    metrics_to_plot = {
        "AJI": "aji",
        "F1 @ IoU 0.5": "f1_iou50",
        "F1 @ IoU 0.75": "f1_iou75",
        "mF1 @ IoU 0.5:0.95": "mF1_iou50_95",
    }

    means, cis, sample_counts = _load_quantitative_metrics(json_files, metrics_to_plot)
    single_sample_mode = all(count == 1 for count in sample_counts)

    x = np.arange(len(metrics_to_plot))
    width = 0.8 / len(labels)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, label in enumerate(labels):
        offset = (i - len(labels) / 2 + 0.5) * width

        m_means = [means[m][i] for m in metrics_to_plot]
        if single_sample_mode:
            ax.bar(x + offset, m_means, width, label=label)
        else:
            m_cis = [cis[m][i] for m in metrics_to_plot]
            ax.bar(x + offset, m_means, width, yerr=m_cis, label=label, capsize=5)

    ax.set_ylabel("Score")
    if single_sample_mode:
        ax.set_title(
            "Quantitative Ablation Results (descriptive single-image comparison)"
        )
        print(
            "Single-sample input detected; plotting descriptive scores without confidence intervals."
        )
    else:
        ax.set_title("Quantitative Ablation Results")
    ax.set_xticks(x)
    ax.set_xticklabels(list(metrics_to_plot.keys()))
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, format="tiff")
    print(f"Saved quantitative plot to {output_path}")


def main() -> None:
    args = parse_args()
    print(f"Quantitative plot: {len(args.json_files)} model(s) -> {args.output_plot}")
    for label, json_file in zip(args.labels, args.json_files):
        print(f"  loading metrics for {label}: {json_file}")
    generate_quantitative_plot(args.json_files, args.labels, args.output_plot)


if __name__ == "__main__":
    main()
