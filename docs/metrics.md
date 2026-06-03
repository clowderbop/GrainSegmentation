# Metrics

Definitions for evaluation numbers produced by SLURM eval jobs and **post-eval reporting**. For the authoritative thesis ordering and YOLO vs U-Net comparison policy, see [`CONTEXT.md`](../CONTEXT.md) and [ADR 0008](adr/0008-pq-headline-instance-evaluation.md).

## Instance metrics (all producers)

- **PQ (Panoptic Quality):** Headline whole-section instance metric for individual grain instance recovery. Uses one-to-one matches with the standard strict IoU > 0.5 convention and combines detection quality with matched-mask quality.
- **DQ (Detection Quality):** PQ component measuring one-to-one instance detection quality.
- **SQ (Segmentation Quality):** PQ component measuring mean IoU of matched instances.
- **Precision / Recall / F1 at IoU thresholds:** Object-level matching diagnostics at IoU50 and IoU75 using strict IoU > threshold matching.
- **Mean F1 over IoU 0.50–0.95:** `mF1_iou50_95` averages F1 at IoU thresholds 0.50, 0.55, …, 0.95 with the same strict-threshold matching rule.
- **Instance counts:** Predicted instance count, ground-truth instance count, and predicted/ground-truth instance ratio.
- **AJI+ (Aggregated Jaccard Index Plus):** Supporting microscopy-style overlap diagnostic with unique instance pairing. Not a headline metric.

## YOLO-only

- **Patch AP/mAP:** Optional detector diagnostics computed only by Ultralytics val on patch data. AP/mAP metrics are not part of the instance metric bundle and are not used for variant ranking or model comparison.

## Policy (headline vs supporting)

| Concept | Where defined |
|---------|----------------|
| **Variant test ranking** | Whole-section sliding-window test; rank input configurations by **PQ** with required diagnostics |
| **Model test comparison** | YOLO vs U-Net on same variant and test mosaic under shared **test inference recipe**, using **PQ** and the instance metric bundle |
| **Supporting test metrics** | Patch instance metric bundle for diagnosing sliding-window effects; optional YOLO patch AP/mAP from Ultralytics val |
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
