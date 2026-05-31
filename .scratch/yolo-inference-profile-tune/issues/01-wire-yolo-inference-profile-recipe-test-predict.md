Status: ready-for-agent
Category: enhancement
Labels: ready-for-agent, enhancement
Depends-on: .scratch/yolo-score-merge-at-predict/issues/01-implement-score-merge-at-predict.md
Blocks: .scratch/yolo-inference-profile-tune/issues/02-train-staged-profile-selection.md

# Wire YOLO inference profile through recipe and test predict

## Parent

[PRD: YOLO inference profile train selection](../PRD.md) · [ADR-0005](../../../docs/adr/0005-yolo-inference-profile-train-selection.md)

## What to build

End-to-end wiring so whole-section and patch YOLO test predict read a **YOLO inference profile** from the shared **test inference recipe**, without running train selection yet.

Extend the recipe with fields for SAHI slice-merge postprocess (`postprocess_type`, `match_metric`, `match_threshold`), minimum **score** (`conf`), and `mask_threshold`. Teach the recipe loader and SLURM shell exports to surface them. Whole SAHI predict must apply those values instead of hard-coded GREEDYNMM / IOS / 0.5; patch predict must use `conf` and `mask_threshold` (SAHI merge fields apply to whole only). **Run provenance** on every predict run must record the resolved profile. Defaults in the committed recipe must match today’s behavior so existing semantics are unchanged until profile selection promotes new values.

Add tests that a non-default profile flows through predict (at least whole path; patch where applicable). Do not implement the train grid search or promotion tooling in this slice.

## Acceptance criteria

- [ ] `configs/test_inference.yaml` documents **YOLO inference profile** fields; loader and `emit_shell_exports` expose them
- [ ] Whole SAHI predict uses recipe-driven postprocess type, match metric, match threshold, **conf**, and mask threshold
- [ ] Patch test predict uses recipe **conf** and mask threshold
- [ ] **Run provenance** includes all profile fields used on that run
- [ ] Defaults preserve current behavior (GREEDYNMM, IOS, 0.5, conf 0.25, mask threshold 0.5)
- [ ] Tests cover non-default profile wiring

## Blocked by

- [.scratch/yolo-score-merge-at-predict/issues/01-implement-score-merge-at-predict.md](../../yolo-score-merge-at-predict/issues/01-implement-score-merge-at-predict.md) — canonical **instance prediction set** at predict must exist before profile tuning or test semantics align with ADR 0004
