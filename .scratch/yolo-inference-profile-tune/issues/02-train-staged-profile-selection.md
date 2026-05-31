Status: ready-for-agent
Category: enhancement
Labels: ready-for-agent, enhancement
Depends-on: .scratch/yolo-inference-profile-tune/issues/01-wire-yolo-inference-profile-recipe-test-predict.md
Blocks: .scratch/yolo-inference-profile-tune/issues/03-hitl-run-tune-commit-recipe.md

# Train staged YOLO inference profile selection and promotion

## Parent

[PRD: YOLO inference profile train selection](../PRD.md) · [ADR-0005](../../../docs/adr/0005-yolo-inference-profile-train-selection.md)

## What to build

End-to-end **profile selection** on the train section, then tooling to freeze the winner in the **test inference recipe**.

Check in a search-grid config (staged lists for merge knobs, then **conf** and mask threshold). Implement a tune entrypoint (CLI and SLURM wrapper) that, for each candidate: runs whole-section train **sliding window** predict per registry variant (each variant’s trained weights), builds canonical **instance prediction sets**, scores whole-section train **AJI** vs train vector labels, and aggregates **mean AJI across variants**. Stage 1 selects SAHI merge settings with fixed default **conf** / mask threshold; stage 2 selects **conf** and mask threshold with stage-1 winners fixed. Write scratch audit artifacts (results table, stage winners). Provide a promote step that updates `configs/test_inference.yaml` with the winning profile (values intended to be committed to git).

Document the rare pipeline step in YOLO SLURM docs: run after all variant weights exist and after score merge at predict; re-run when train labels or weights change materially, not after every single-variant training job. Pre-merge metrics may be logged for diagnostics but must not drive selection.

## Acceptance criteria

- [ ] Grid spec committed (e.g. `configs/yolo_inference_profile_tune.yaml` or equivalent)
- [ ] Stage 1 and stage 2 searches run on train whole section for all registry variants
- [ ] Selection uses mean train whole-section **AJI** on canonical **instance prediction sets** only
- [ ] One shared winning profile (not per-variant winners)
- [ ] Promote step updates **test inference recipe** with winners; audit tables on scratch
- [ ] SLURM submit script and `SLURM/yolo/pipeline.md` describe when to run tune
- [ ] Patch and whole test jobs already read promoted recipe (from slice 01)

## Blocked by

- [.scratch/yolo-inference-profile-tune/issues/01-wire-yolo-inference-profile-recipe-test-predict.md](01-wire-yolo-inference-profile-recipe-test-predict.md)
