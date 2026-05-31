Status: ready-for-human
Category: ops
Labels: ready-for-human
Depends-on: .scratch/yolo-inference-profile-tune/issues/02-train-staged-profile-selection.md
Blocks: —

# Run profile selection on cluster and lock recipe for paper

## Parent

[PRD: YOLO inference profile train selection](../PRD.md) · [ADR-0005](../../../docs/adr/0005-yolo-inference-profile-train-selection.md)

## What to build

Operator execution after implementation issues 01 (score merge) and 02 (profile wiring + tune tooling) are merged.

Run staged **profile selection** on the cluster when all four YOLO variant weights exist on scratch. Promote the winning **YOLO inference profile** into `configs/test_inference.yaml` and **commit** that file to git with the audit reference (scratch tune run id / results table path in lab notes or thesis methods). Re-run held-out YOLO test evaluation for all variants and regenerate the **post-eval reporting** bundle so headline tables match the committed recipe. Record the git commit hash used for held-out test in methods text.

Do not treat this as required after every per-variant training job—only when labels or weights changed materially or the search grid changed.

## Acceptance criteria

- [ ] Staged tune completed on train section; scratch audit artifacts retained
- [ ] `configs/test_inference.yaml` committed with promoted profile values
- [ ] Held-out YOLO whole + patch test eval re-run under committed recipe
- [ ] **Reporting bundle** regenerated from new eval outputs
- [ ] Methods note cites train-selection objective (mean train AJI, shared profile) and recipe commit for test

## Blocked by

- [.scratch/yolo-inference-profile-tune/issues/02-train-staged-profile-selection.md](02-train-staged-profile-selection.md)
