Status: ready-for-agent
Category: enhancement
Labels: ready-for-agent, enhancement
Phase: C
Depends-on: 01-phase-a-watershed-and-instance-extraction, 02-phase-b-variant-registry-paths
Blocks: 04-phase-d-yolo-staging-and-data-prep, 05-phase-e-visualization-and-tf-runtime

# Phase C: Eval profiles & family pipelines

Parent: [PRD overview](../PRD.md)

## Problem Statement

Four or more bash drivers duplicate the same eval sequence (stage → predict → extract → write-eval → evaluate instances) with family-specific variations. Orchestration logic in shell is hard to test, drifts between patch/whole/CC/watershed/YOLO/SAHI paths, and mixes repo-root `uv run` with family venvs. There is no normalized output layout.

## Solution

Add shared **eval profiles** (data only, no family imports). Add **U-Net eval pipeline** and **YOLO eval pipeline** modules that import step functions directly. Refactor CLIs to expose `run_*` functions used by pipelines. Thin SLURM scripts to `sbatch` + `--profile`. Stub **predict runtime** protocol on U-Net side for future TF isolation. Document dual-env job rules.

## User Stories

1. As a researcher, I want a single U-Net patch test profile (stage → predict → extract → write-eval → evaluate), so that bash duplication ends.
2. As a researcher, I want U-Net whole test eval with multi-model TSV, semantic metrics, and plots in one profile.
3. As a researcher, I want train-section CC vs watershed as one profile parameterized by instance method.
4. As a researcher, I want YOLO patch and SAHI whole profiles matching the same pattern.
5. As a maintainer, I want orchestration only in U-Net/YOLO packages, not shared importing both families.
6. As a maintainer, I want direct imports of step functions, not subprocess CLIs.
7. As a researcher, I want normalized eval output layout under `{output_root}/{family}/{unit}/{split}/{variant}/{run_id}/`.
8. As a developer, I want pytest with injectable step adapters on login nodes.
9. As a researcher, I want optional cluster smoke documented for one U-Net patch profile.
10. As a maintainer, I want U-Net jobs to use U-Net env for staging/metrics (`--no-sync`) and sync before predict; YOLO jobs entirely in YOLO env; no repo-root sync in eval.
11. As a maintainer, I want a predict-runtime protocol stub for future subprocess/container predict.

## Implementation Decisions

### Shared package (no U-Net/YOLO imports)

- **Eval profiles** — frozen dataclasses/enums: family, unit, split, instance method, flags (semantic, mask AP, plots, overlay, ultralytics val). Profile registry by name (`unet_patch_test`, `unet_whole_test`, `unet_whole_train`, `yolo_patch_test`, `yolo_sahi_whole_test`). Helper to compute normalized output paths.

### U-Net package

- **Eval pipeline** — `run_eval(profile, *, work_root, output_root, grainseg_root, variant, model_path(s), …, steps=None)`. Default steps call refactored `run_predict`, `run_extract_instances`, `run_evaluate_semantic`, shared stage/write-eval/evaluate_instances, plot helpers.
- **Injectable adapters** — optional `UnetEvalSteps` protocol/dataclass for tests.
- **Predict runtime protocol** — default in-process Keras; interface only in this phase (no container yet).
- Refactor predict, extract_instances, evaluate_semantic CLIs to call shared runners.

### YOLO package

- **Eval pipeline** — same pattern; optional ultralytics val behind flag.

### SLURM

- Replace bodies of whole/patch U-Net eval and YOLO patch/SAHI scripts with pipeline invocation + SBATCH headers/resources.
- Remove watershed bash arg builder (Phase A Python only).
- Paths from Phase B variant CLI.

### Runtime env documentation

- README job profile table; new `docs/agents/runtime-env.md`: U-Net vs YOLO env, never repo-root sync in eval.

### Output layout (breaking OK)

- Document new layout; do not migrate old scratch trees.

## Testing Decisions

- Unit: eval profile path builder; pipeline with fake steps (no GPU).
- Optional `@pytest.mark.integration`: tiny manifest + staged tmp dir for one U-Net patch profile.
- Prior art: staging integration marker; `test_predict_manifest`.
- Manual: cluster smoke doc for `unet_patch_test` with `srun`.
- Not in CI: full TF predict, SAHI on real TIFFs.

## Acceptance criteria

- [ ] Five profiles implemented and callable from CLI `-m unet.eval_pipeline` / `-m yolo.eval_pipeline`.
- [ ] SLURM eval scripts are thin wrappers (&lt; ~80 lines each excluding SBATCH).
- [ ] Existing metrics outputs equivalent for one variant smoke (document comparison command).
- [ ] Injectable adapter tests pass in CI.
- [ ] Predict runtime protocol defined; default implementation used.
- [ ] Runtime env doc + README table added.
- [ ] No `common` imports of `unet` or `yolo`.

## Out of Scope (this phase)

- YOLO train/tune manifest-only staging (Phase D) — YOLO eval may still use interim staging until D.
- TF subprocess runtime implementation (Phase E).
- Full visualization consolidation (Phase E) — basic plots in whole profile OK if already in bash today.

## Further Notes

- Implement **U-Net patch test** profile first, then whole, then YOLO.
- Multi-model whole eval: preserve TSV-driven model list behavior.

## Comments
