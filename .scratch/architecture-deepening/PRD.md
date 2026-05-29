Status: ready-for-agent
Category: enhancement
Labels: ready-for-agent, enhancement

# PRD: Architecture deepening (overview)

Parent plan for eval pipelines, variant registry, YOLO staging, preprocessing, and runtime seams. **Implementation is split into phased issues** — agents should pick one issue and complete it before starting a dependent phase.

## Problem Statement

Grain segmentation research runs eight experiment pipelines (four input variants × U-Net and YOLO families) on a SLURM cluster. Manifests, variant registry, staging, and shared instance metrics are consolidated in the shared Python layer, but orchestration and configuration still leak across bash, shell helpers, and duplicate entry points. See phase issues for scoped problems and solutions.

## Solution

Five incremental phases (A→E). Each phase merges independently with tests green. Later phases assume earlier interfaces exist.

## Phased issues

| # | Issue | Depends on |
|---|--------|------------|
| 01 | [Phase A — Watershed params & instance extraction](issues/01-phase-a-watershed-and-instance-extraction.md) | — |
| 02 | [Phase B — Variant registry paths](issues/02-phase-b-variant-registry-paths.md) | 01 (soft: can parallel if careful) |
| 03 | [Phase C — Eval profiles & family pipelines](issues/03-phase-c-eval-pipelines.md) | 01, 02 |
| 04 | [Phase D — YOLO manifest staging & data prep](issues/04-phase-d-yolo-staging-and-data-prep.md) | 02, 03 (YOLO pipeline from C) |
| 05 | [Phase E — Visualization & TF runtime](issues/05-phase-e-visualization-and-tf-runtime.md) | 03 |

## Dependency rule (all phases)

The shared Python package must not import U-Net or YOLO packages. Eval orchestration that needs both families uses two family-specific pipeline modules plus shared profile data and staging/metrics modules.

## Global out of scope

- Merging TensorFlow and PyTorch into one environment
- Login-node production preprocessing CLI
- Replacing Ultralytics or SAHI
- Migrating historical scratch eval directories automatically
- Changing thesis experiment matrix or model architectures
- CI execution of SLURM or GPU training

## Cross-cutting notes

- Registry already has `resolve_paths` per variant; Phase B extends CLI and fixes stale YOLO paths.
- `WatershedParamSet` already exists in watershed tuning; Phase A moves and shares it.
- Update project domain glossary when naming new concepts (eval profile, instance extraction facade, etc.).

## Comments
