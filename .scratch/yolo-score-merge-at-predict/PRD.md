# YOLO score merge at predict

## Problem

YOLO **instance prediction sets** on disk held overlapping **detector proposals**. Headline instance metrics applied **score merge** transiently; SAHI **prediction overlays** painted proposals in score order. Users saw split or duplicate grains at tile seams (**slice-boundary duplicate**) and overlapping duplicates, while AJI/F1 measured a different layout than the persisted file suggested. **Mask AP** ran on proposals, not on the grain layout used for **model test comparison**.

## Goal

One canonical YOLO **system output**: non-overlapping grains with **score** (winning proposal), written at predict time for whole and patch **sample unit**. All consumers (AJI/F1, **Mask AP**, overlays, reporting) read the same `prediction_sets/*.json`.

## Non-goals

- Fusing **slice-boundary duplicate** detections (separate future work).
- Changing U-Net extraction or **test inference recipe** geometry.
- Backwards-compatible readers for pre-change proposal JSON.
- Replacing Ultralytics patch val mAP (still native detector on patch crops).

## Success

- After implementation, delete old scratch YOLO eval dirs and re-run `submit_test_evaluations.sh`.
- Overlay TIFFs match instance metrics input without eval-time merge for YOLO.
- **Mask AP** and AJI/F1 both use canonical prediction sets; numbers differ from pre-ADR-0004 runs (expected).

## References

- [ADR 0004](../../docs/adr/0004-yolo-score-merge-at-predict.md)
- `CONTEXT.md` — **Score merge**, **Slice-boundary duplicate**, **Mask AP**
