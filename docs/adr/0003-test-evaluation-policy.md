# Test evaluation and ranking policy

Status: accepted

## Context / Problem

Held-out reporting needs one headline objective and one fixed inference setup so input configurations and model families are comparable. The scientific target is **individual grain instance recovery** on whole sandstone sections: each annotated grain should be recovered as a separate predicted instance, with object detection and mask quality both reflected.

## Decision

**Variant test ranking** and **model test comparison** use whole-section sliding-window inference, scored by **whole-section PQ** on the **merged instance view**. Reports also include DQ, SQ, precision/recall/F1 at IoU50 and IoU75, mean F1 over IoU50:95, predicted and ground-truth instance counts, predicted/ground-truth ratio, and AJI+. AJI+ remains a supporting microscopy overlap diagnostic, not the headline.

The same **instance metric bundle** is computed for every held-out **eval** instance evaluation whenever artifacts support it. Train-side grid scoring (YOLO **profile selection**, U-Net watershed tune, CC-vs-watershed pick) persists **`MergedViewPqResult`** from `compute_merged_view_pq` instead — same **PQ** definition, without IoU75, mP/mR/mF1 0.5:0.95, or AJI+ on the tune hot path ([tune-path vs eval-path](../metrics.md#tune-path-vs-eval-path-diagnostics)). Patch evaluations compute patch-level bundle fields as supporting evidence only. Patch metric aggregates exclude empty-GT patches from means, count them separately, and report both unweighted and grain-weighted means over grain-bearing patches.

AP/mAP metrics are outside the instance metric bundle. They are optional YOLO patch diagnostics from Ultralytics val only, not whole-section Mask AP and not cross-model ranking evidence.

The shared **test inference recipe** lives at `config/test_inference.yaml`. It governs window geometry, patch crop size, batching, the frozen **YOLO inference profile**, and patch-val settings. Per-variant test inference settings are out of scope. U-Net keeps per-checkpoint **U-Net extraction profile** settings selected on train; those are not stored as shared recipe knobs.

Before held-out test, YOLO profile selection chooses the shared profile on the whole train section by mean whole-section train PQ across registry variants, then promotes the winner into the recipe; see [ADR 0005](0005-yolo-inference-profile-train-selection.md). U-Net watershed tuning and CC-vs-watershed selection also use train whole-section PQ.

## Rejected Alternatives

Patch metrics or Ultralytics mAP as headline; AJI or AJI+ as headline; F1@IoU50 as headline; per-variant test profiles; a single shared U-Net watershed profile. These were rejected because they either miss the deployment unit, are YOLO-centric, over-reward area overlap, or confound input configuration with inference settings.

## Consequences

Evaluation code and reporting must expose the full **instance metric bundle** on held-out **eval**. Train-side tune paths must expose **`MergedViewPqResult`** audit fields with **PQ** as the selection objective. Existing AJI-selected profile or watershed runs are audit evidence only and should not be promoted as final test settings under this policy.

## Links

- YOLO profile selection: [ADR 0005](0005-yolo-inference-profile-train-selection.md)
- Metric definitions: [`docs/metrics.md`](../metrics.md) ([tune-path vs eval-path](../metrics.md#tune-path-vs-eval-path-diagnostics))
- Glossary: [`CONTEXT.md`](../../CONTEXT.md)
- YOLO runbook: [`docs/runbooks/yolo.md`](../runbooks/yolo.md)
- U-Net runbook: [`docs/runbooks/unet.md`](../runbooks/unet.md)
