# PRD: YOLO inference profile train selection

## Problem

YOLO whole-section inference uses SAHI slice merge and Ultralytics per-slice detection with hyperparameters that were hard-coded and not train-selected. Held-out **variant test ranking** requires a frozen **test inference recipe** shared across input variants (ADR 0003). We need **profile selection** on the train section that optimizes the deployed YOLO system (canonical **instance prediction set** after **score merge** — ADR 0004) and promotes winners into the recipe before test eval.

## Goals

- One **YOLO inference profile** shared across all registry variants on test.
- **Staged search** on train: SAHI merge settings, then minimum **score** (`conf`) and mask threshold.
- Primary objective: mean whole-section train **AJI** on canonical sets, averaged across variants.
- Winning values committed in `configs/test_inference.yaml`; scratch tune runs hold audit tables.

## Non-goals

- Per-variant inference profiles on test.
- Tuning on overlapping **detector proposals** as the primary objective.
- Fixing **slice-boundary duplicate** via merge thresholds.
- Full factorial or Bayesian search in v1.

## References

- [ADR 0005](../../docs/adr/0005-yolo-inference-profile-train-selection.md)
- [ADR 0004](../../docs/adr/0004-yolo-score-merge-at-predict.md) (prerequisite)
- `CONTEXT.md` — **YOLO inference profile**, **profile selection**
