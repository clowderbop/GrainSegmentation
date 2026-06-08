# Metrics

Definitions for evaluation numbers produced by SLURM eval jobs and **post-eval reporting**. For thesis ordering and YOLO vs U-Net comparison policy, see [ADR 0003](adr/0003-test-evaluation-policy.md). Domain vocabulary: [`CONTEXT.md`](../CONTEXT.md).

## Instance metrics (all producers)

- **PQ (Panoptic Quality):** Headline whole-section instance metric for individual grain instance recovery. Uses one-to-one matches with the standard strict IoU > 0.5 convention and combines detection quality with matched-mask quality.
- **DQ (Detection Quality):** PQ component measuring one-to-one instance detection quality.
- **SQ (Segmentation Quality):** PQ component measuring mean IoU of matched instances.
- **TP / FP / FN (IoU50 match counts):** Greedy one-to-one match counts at strict IoU > 0.5 — true positives, false positives (unmatched predictions), and false negatives (unmatched ground-truth grains). Same convention as **PQ** and tune-path **`MergedViewPqResult`**.
- **Precision / Recall / F1 at IoU thresholds:** Object-level matching diagnostics at IoU50 and IoU75 using strict IoU > threshold matching.
- **Mean precision / recall / F1 over IoU 0.50–0.95:** `mP_iou50_95`, `mR_iou50_95`, and `mF1_iou50_95` average precision, recall, and F1 at IoU thresholds 0.50, 0.55, …, 0.95 with the same strict-threshold matching rule.
- **Instance counts:** Predicted instance count, ground-truth instance count, and predicted/ground-truth instance ratio.
- **AJI+ (Aggregated Jaccard Index Plus):** Supporting microscopy-style overlap diagnostic with unique instance pairing. Not a headline metric.

### PQ diagnostics

Required companion fields whenever PQ is reported on held-out **eval** or in **post-eval reporting** (`compute_instance_metric_bundle`). Same strict IoU > threshold matching as PQ. Glossary: **PQ diagnostics** in [`CONTEXT.md`](../CONTEXT.md).

| Field group | Contents |
|-------------|----------|
| PQ decomposition | **DQ**, **SQ** |
| IoU50 match counts | **TP**, **FP**, **FN** (greedy one-to-one at strict IoU > 0.5) |
| Thresholded matching | Precision, recall, and F1 at IoU50 and IoU75 |
| Stricter mask summary | Mean precision, recall, and F1 over IoU 0.50–0.95 (`mP_iou50_95`, `mR_iou50_95`, `mF1_iou50_95`) |
| Instance counts | Predicted count, ground-truth count, predicted/ground-truth ratio |
| Overlap diagnostic | **AJI+** (supporting; not headline) |

Together with **PQ**, these form the **instance metric bundle** for each evaluated sample unit on the eval path (`INSTANCE_METRIC_BUNDLE_KEYS` in [`src/common/instance_metric_bundle.py`](../src/common/instance_metric_bundle.py)).

### Tune-path vs eval-path diagnostics

Train-side grid scoring (YOLO **profile selection**, U-Net watershed tune, CC-vs-watershed train pick) and held-out **eval** share the same **PQ** definition (greedy one-to-one match, strict IoU > 0.5). They differ in which fields are computed and persisted.

| Path | Entry point | Persisted record | Selection uses |
|------|-------------|------------------|----------------|
| **Tune** | `compute_merged_view_pq` | **`MergedViewPqResult`** — PQ/DQ/SQ, IoU50 P/R/F1, TP/FP/FN, instance counts, matched-IoU spread, overlap forensics | `pq` / `mean_pq` only |
| **Eval** | `compute_instance_metric_bundle` | Full **instance metric bundle** — shared IoU50 PQ/DQ/SQ, **TP**/**FP**/**FN**, IoU50 P/R/F1, plus IoU75 P/R/F1, mP/mR/mF1 0.5:0.95, **AJI+** | Headline **PQ**; full **PQ diagnostics** in reports |

Implementation: [`src/common/merged_view_pq.py`](../src/common/merged_view_pq.py), [`src/common/instance_metric_bundle.py`](../src/common/instance_metric_bundle.py). YOLO profile selection calls `compute_train_pq` → `compute_merged_view_pq` after cross-tile association.

## YOLO-only

- **Patch AP/mAP:** Optional detector diagnostics computed only by Ultralytics val on patch data. AP/mAP metrics are not part of the instance metric bundle and are not used for variant ranking or model comparison.

## Policy (headline vs supporting)

| Concept | Where defined |
|---------|----------------|
| **Variant test ranking** | Whole-section sliding-window test; rank input configurations by **PQ** with required diagnostics |
| **Model test comparison** | YOLO vs U-Net on same variant and test mosaic under shared **test inference recipe**, using **PQ** and the instance metric bundle |
| **Supporting test metrics** | Patch instance metric bundle for diagnosing sliding-window effects; optional YOLO patch AP/mAP from Ultralytics val |
| **Patch metric aggregate** | Unweighted mean over grain-bearing patches; grain-weighted mean by GT instance count |

## PQ-centered rerun policy

Authoritative policy for train-side selection, held-out ranking, and reporting. Decision record: [ADR 0003](adr/0003-test-evaluation-policy.md). YOLO profile-selection architecture: [ADR 0005](adr/0005-yolo-inference-profile-train-selection.md). Vocabulary: [instance metrics](#instance-metrics-all-producers), [PQ diagnostics](#pq-diagnostics), [tune-path vs eval-path](#tune-path-vs-eval-path-diagnostics), [`CONTEXT.md`](../CONTEXT.md).

| Stage | PQ objective | Diagnostics persisted | Runbook |
|-------|----------------|----------------------|---------|
| YOLO **profile selection** | Maximize `mean_pq` on train whole section | **`MergedViewPqResult`** per variant | [`runbooks/yolo.md`](runbooks/yolo.md#profile-selection) |
| **Profile promotion** | Install PQ winner into `config/test_inference.yaml` | — | [`runbooks/yolo.md`](runbooks/yolo.md#promotion) |
| U-Net watershed tune | `best_mean_pq` per variant | **`MergedViewPqResult`** | [`runbooks/unet.md`](runbooks/unet.md#watershed-tuning) |
| CC vs watershed (train) | Mean train PQ in `eval/extraction_method_selection.json` | **`MergedViewPqResult`** | [`runbooks/unet.md`](runbooks/unet.md#cc-vs-watershed-train-section) |
| Held-out **eval** / predict metrics | Headline **whole-section PQ** | Full **instance metric bundle** | YOLO / U-Net runbooks (test eval) |
| **Post-eval reporting** | Headline **whole-section PQ** + [PQ diagnostics](#pq-diagnostics); AP/mAP in YOLO-only patch panel | From eval artifacts | [`runbooks/analysis.md`](runbooks/analysis.md) |

## Post-eval reporting

After test eval jobs finish, build comparison tables and figures from scratch artifacts:

```bash
uv sync --group analysis
uv run --group analysis python -m analysis.build_report \
  --grainseg-root "$SCRATCH/GrainSeg"
```

Cluster: [`runbooks/analysis.md`](runbooks/analysis.md). **Eval run discovery** (v1): convention-based paths in `src/analysis/discover.py`; no catalog file in v1.

Outputs under `$SCRATCH/GrainSeg/eval/reporting/` (`derived/`, `figures/`, `analysis_summary.json`). Variant axis labels use `display_name` in `config/variants.yaml`.
