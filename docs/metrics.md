# Metrics

Definitions for evaluation numbers produced by SLURM eval jobs and **post-eval reporting**. For how metrics drive thesis ordering and YOLO vs U-Net comparison policy, see [`CONTEXT.md`](../CONTEXT.md) (**Test evaluation policy**, **Variant test ranking**, **Supporting test metrics**, **Mask AP**, **Model test comparison**).

## Instance metrics (all producers)

- **AJI (Aggregated Jaccard Index):** Instance-aware metric for microscopy/cell segmentation. Penalizes under-segmentation (merged grains) and over-segmentation (split grains) at the pixel level. Holistic view of detection and boundary adherence without confidence thresholds.
- **Precision:** Correctly predicted positives / total predicted positives.
- **Recall:** Correctly predicted positives / all actual positives.
- **F1 score:** Harmonic mean of precision and recall.
- **Mean P/R/F1 over IoU 0.50–0.95:** `mP_iou50_95`, `mR_iou50_95`, `mF1_iou50_95` average precision, recall, and F1 at IoU thresholds 0.50, 0.55, …, 0.95 with the same matching rule at each threshold.

## YOLO-only

- **COCO-style mask AP (Average Precision):** Decouples detection from spatial accuracy; averages across IoU thresholds (`mAP@0.5:0.95`) using YOLO **score** on the canonical **instance prediction set** after **score merge**. Not computed for U-Net (no per-instance confidence on extracted grains).

## Policy (headline vs supporting)

| Concept | Where defined |
|---------|----------------|
| **Variant test ranking** | Whole-section sliding-window test; rank input configurations by instance **AJI**, report **F1@IoU50** alongside |
| **Model test comparison** | YOLO vs U-Net on same variant and test mosaic under shared **test inference recipe** |
| **Supporting test metrics** | Patch AJI/F1, patch detector mAP, whole-section **Mask AP** — not headline rank |
| **Patch metric aggregate** | Unweighted mean over grain-bearing patches; grain-weighted mean by GT instance count |

## Post-eval reporting

After test eval jobs finish, build comparison tables and figures from scratch artifacts:

```bash
uv sync --group analysis
uv run --group analysis python -m analysis.build_report \
  --grainseg-root "$SCRATCH/GrainSeg"
```

Cluster: [`runbooks/analysis.md`](runbooks/analysis.md). **Eval run discovery** (v1): convention-based paths in `src/analysis/discover.py`; no catalog file in v1.

Outputs under `$SCRATCH/GrainSeg/eval/reporting/` (`derived/`, `figures/`, `analysis_summary.json`). Variant axis labels use `display_name` in `config/variants.yaml`.
