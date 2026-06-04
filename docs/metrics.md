# Metrics

Definitions for evaluation numbers produced by SLURM eval jobs and **post-eval reporting**. For the authoritative thesis ordering and YOLO vs U-Net comparison policy, see [`CONTEXT.md`](../CONTEXT.md) and [ADR 0003](adr/0003-test-evaluation-policy.md).

## Instance metrics (all producers)

- **PQ (Panoptic Quality):** Headline whole-section instance metric for individual grain instance recovery. Uses one-to-one matches with the standard strict IoU > 0.5 convention and combines detection quality with matched-mask quality.
- **DQ (Detection Quality):** PQ component measuring one-to-one instance detection quality.
- **SQ (Segmentation Quality):** PQ component measuring mean IoU of matched instances.
- **Precision / Recall / F1 at IoU thresholds:** Object-level matching diagnostics at IoU50 and IoU75 using strict IoU > threshold matching.
- **Mean precision / recall / F1 over IoU 0.50–0.95:** `mP_iou50_95`, `mR_iou50_95`, and `mF1_iou50_95` average precision, recall, and F1 at IoU thresholds 0.50, 0.55, …, 0.95 with the same strict-threshold matching rule.
- **Instance counts:** Predicted instance count, ground-truth instance count, and predicted/ground-truth instance ratio.
- **AJI+ (Aggregated Jaccard Index Plus):** Supporting microscopy-style overlap diagnostic with unique instance pairing. Not a headline metric.

### PQ diagnostics

Required companion fields whenever PQ is reported (train selection, held-out eval, **post-eval reporting**). Same strict IoU > threshold matching as PQ. Glossary: **PQ diagnostics** in [`CONTEXT.md`](../CONTEXT.md).

| Field group | Contents |
|-------------|----------|
| PQ decomposition | **DQ**, **SQ** |
| Thresholded matching | Precision, recall, and F1 at IoU50 and IoU75 |
| Stricter mask summary | Mean precision, recall, and F1 over IoU 0.50–0.95 (`mP_iou50_95`, `mR_iou50_95`, `mF1_iou50_95`) |
| Instance counts | Predicted count, ground-truth count, predicted/ground-truth ratio |
| Overlap diagnostic | **AJI+** (supporting; not headline) |

Together with **PQ**, these form the **instance metric bundle** for each evaluated sample unit.

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

Authoritative policy for train-side selection, held-out ranking, and reporting. Decision record: [ADR 0003](adr/0003-test-evaluation-policy.md). YOLO profile-selection architecture: [ADR 0005](adr/0005-yolo-inference-profile-train-selection.md). Vocabulary: [instance metrics](#instance-metrics-all-producers), [PQ diagnostics](#pq-diagnostics), [`CONTEXT.md`](../CONTEXT.md).

| Stage | PQ objective | Runbook |
|-------|----------------|---------|
| YOLO **profile selection** | Maximize `mean_pq` on train whole section | [`runbooks/yolo.md`](runbooks/yolo.md#profile-selection) |
| **Profile promotion** | Install PQ winner into `config/test_inference.yaml` | [`runbooks/yolo.md`](runbooks/yolo.md#promotion) |
| U-Net watershed tune | `best_mean_pq` per variant | [`runbooks/unet.md`](runbooks/unet.md#watershed-tuning) |
| CC vs watershed (train) | Mean train PQ in `eval/extraction_method_selection.json` | [`runbooks/unet.md`](runbooks/unet.md#cc-vs-watershed-train-section) |
| **Post-eval reporting** | Headline **whole-section PQ** + [PQ diagnostics](#pq-diagnostics); AP/mAP in YOLO-only patch panel | [`runbooks/analysis.md`](runbooks/analysis.md) |

### Stale AJI-selected scratch outputs

Older artifacts may rank or promote by AJI or `mean_aji` instead of PQ:

| Location | Do not use for final settings under PQ policy |
|----------|-----------------------------------------------|
| `runs/yolo_inference_profile_tune/` | AJI-era `grid/winner.json` or high-`mean_aji` rows for **profile promotion** |
| `config/test_inference.yaml` | Values promoted from an AJI-selected winner |
| `runs/watershed_tune/`, `eval/extraction_method_selection.json` | AJI-driven extraction winners |
| `eval/yolo_*`, `eval/unet_*`, `eval/reporting/` | Pre-policy or AJI-headline held-out eval / reporting |

Keep these trees for audit and explaining the policy change. Final thesis ranking, promoted **YOLO inference profile**, U-Net extraction choice, held-out test, and **reporting bundle** should come from PQ-centered train-side artifacts and matching held-out eval—then `analysis.build_report`. Runbooks link here for operational detail; they do not restate this table.

## Post-eval reporting

After test eval jobs finish, build comparison tables and figures from scratch artifacts:

```bash
uv sync --group analysis
uv run --group analysis python -m analysis.build_report \
  --grainseg-root "$SCRATCH/GrainSeg"
```

Cluster: [`runbooks/analysis.md`](runbooks/analysis.md). **Eval run discovery** (v1): convention-based paths in `src/analysis/discover.py`; no catalog file in v1.

Outputs under `$SCRATCH/GrainSeg/eval/reporting/` (`derived/`, `figures/`, `analysis_summary.json`). Variant axis labels use `display_name` in `config/variants.yaml`.
