Status: ready-for-agent
Category: enhancement
Labels: ready-for-agent, enhancement
Phase: D
Depends-on: 02-phase-b-variant-registry-paths, 03-phase-c-eval-pipelines
Blocks: —

# Phase D: YOLO manifest staging & data prep

Parent: [PRD overview](../PRD.md)

## Problem Statement

YOLO train/tune copies entire patch directories to TMPDIR while eval also uses manifest staging — two contracts, redundant I/O. `split_tiff_gpkg_to_yolo` is a large monolith with no unit tests; preprocessing shell repeats per-variant `uv run` blocks. Patch constants are duplicated between scripts.

## Solution

Manifest-only staging for YOLO train, tune, and eval: copy manifest-listed files into Ultralytics directory layout; generate `data.yaml` from manifest. Remove redundant full-tree copy in eval. Split data prep into focused modules with a thin CLI; SLURM-only orchestration calling shared functions. Keep optional Ultralytics `val` behind explicit flag.

## User Stories

1. As a researcher training YOLO, I want only manifest-listed patches staged, so TMPDIR matches U-Net practice.
2. As a researcher training YOLO, I want `data.yaml` generated from staged layout, so Ultralytics train layout is preserved.
3. As a researcher evaluating YOLO patches, I want redundant full-dataset copy removed when manifest staging suffices.
4. As a researcher, I want optional Ultralytics validation behind an explicit flag (still needed).
5. As a maintainer, I want split-tiff-to-YOLO split into tiling, stratified split, patch export, label conversion modules.
6. As a maintainer, I want preprocessing SLURM-only — no login-node production pipeline CLI.
7. As a maintainer, I want golden manifest shape tests for data prep outputs.
8. As a researcher, I want preprocessing shell to call shared functions instead of four duplicated variant blocks.

## Implementation Decisions

### YOLO manifest staging (YOLO + shared packages)

- **Stage YOLO dataset from manifest** — given patch manifest + work root, materialize `images/{train,val,test}` and `labels/...` per Ultralytics expectations; rewrite paths in generated `data.yaml`.
- **Train/tune SLURM** — replace `cp -r` patch tree with manifest stage + yaml generation.
- **Eval** — remove `stage_yolo_patch_dataset` when eval pipeline manifest stage covers inputs; keep optional `yolo.val` via `RUN_ULTRALYTICS_VAL=1` (default off).

### Data prep refactor

- Split monolithic script into modules, e.g.:
  - Constants (patch size 1024, overlap 0.5, tile 4096, validation fraction 0.2)
  - Tiling / coverage stratification
  - Patch iteration and image write
  - YOLO label row builder (existing geometry helpers)
  - Dataset writer / directory layout
- Thin CLI entry preserves current command-line interface for SLURM callers.
- **No** `data_prep pipeline run` for login-node production use — functions imported from SLURM scripts only.

### Preprocessing SLURM

- Refactor create-patch-datasets (and related) to loop variants via registry `all-names` and call shared functions.

### Phase B alignment

- All paths via variant registry; no obsolete yolo root.

## Testing Decisions

- Unit: YOLO label row builder, tiling math, stratified split assignment on synthetic grid.
- Integration: tiny manifest → staged Ultralytics layout exists, yaml points at staged paths.
- Golden: patch manifest JSON shape after writer (extend patch manifest tests).
- Prior art: `test_patch_manifests`, `test_yolo_seg_labels`.
- Not in CI: full `split_tiff_gpkg_to_yolo` on real train TIFFs, SAM2.

## Acceptance criteria

- [ ] YOLO train/tune jobs use manifest staging only; documented in README.
- [ ] YOLO patch eval does not full-copy patch tree when manifest stage suffices.
- [ ] `RUN_ULTRALYTICS_VAL` still works when set.
- [ ] Monolith split into modules; SLURM scripts call modules; no behavior regression on patch counts (document manual verify on one variant).
- [ ] Constants module single source for patch/tile params.
- [ ] New unit tests pass; `uv run pytest` green.

## Out of Scope (this phase)

- Eval pipeline redesign (done in C).
- SAM2 / overlap-split refactors.
- Login-node data prep CLI for researchers.

## Further Notes

- Ultralytics may require relative paths in yaml — staging rewrite logic already exists partially in dataset yaml helpers; extend, do not fork.

## Comments
