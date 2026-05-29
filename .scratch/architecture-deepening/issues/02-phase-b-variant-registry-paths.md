Status: ready-for-agent
Category: enhancement
Labels: ready-for-agent, enhancement
Phase: B
Depends-on: —
Blocks: 03-phase-c-eval-pipelines, 04-phase-d-yolo-staging-and-data-prep

# Phase B: Variant registry paths

Parent: [PRD overview](../PRD.md)

## Problem Statement

To determine which files a job needs, maintainers cross registry YAML, Python registry, shell `env` exports, README scratch layout, and hard-coded paths in SLURM scripts. YOLO config still points at obsolete `dataset/train/yolo/...` while real patches live under `dataset/train/patches/{variant}`. Multiple slug dialects are documented in YAML but not resolved through one CLI.

## Solution

Extend the existing variant registry CLI with path subcommands and `--require`. Fix registry YAML and YOLO-facing config to use patch directory templates (breaking change OK). Update SLURM scripts to resolve paths via CLI instead of string concatenation.

## User Stories

1. As a SLURM job author, I want CLI subcommands that resolve whole-section manifests, U-Net patch manifests, YOLO patch manifests, watershed tune directories, and default U-Net model paths, so that bash stops hard-coding scratch layout.
2. As a researcher, I want manifest path commands to fail when the file is missing (`--require`), so that jobs fail at startup with a clear message.
3. As a maintainer, I want YOLO dataset path configuration aligned with real patch directories, so that training does not point at obsolete paths.
4. As a researcher, I want breaking path changes documented in the README, so that I update scratch once.

## Implementation Decisions

### Extend variant registry CLI

Subcommands (each: `--variant`, `--grainseg-root` where needed, `--require`):

- Whole-section U-Net manifest path (`--split train|test`)
- U-Net patch test manifest path
- YOLO patch manifest path (`--split train|test`)
- Watershed tune directory
- Default U-Net model path

Implement using existing `VariantSpec.resolve_paths` and manifest path templates — do not duplicate path algebra in shell.

### Registry YAML

- Repoint `yolo_dataset_root` to derive from `train_patches_dir` + `dataset_subdir`, or remove field and compute in loader (breaking OK).

### YOLO config adapter

- Remove hardcoded legacy `GrainSeg/dataset/train/yolo` default; read from registry resolved paths.

### SLURM

- Replace hard-coded manifest/model strings in U-Net and YOLO eval/train scripts with CLI invocations.
- Use `run_common_in_unet_env` or equivalent for variant CLI from U-Net jobs.

### Documentation

- README: note breaking path change and new CLI examples.

### Depends on Phase A

- Soft dependency. Can land in parallel if watershed tune dir subcommand coordinates with Phase A `resolve_tune_json` slug rules — prefer sharing slug source from registry `slugs.job` only.

## Testing Decisions

- Unit tests with temporary grainseg tree: paths exist vs `--require` exits non-zero.
- Test all four input variants for manifest path shape (not file content on scratch).
- Prior art: `test_variants.py`, `test_variants_yaml.py`, manifest contract tests.

## Acceptance criteria

- [ ] All five subcommands implemented and tested with `--require`.
- [ ] Registry YAML and YOLO config no longer reference obsolete yolo-only tree as canonical.
- [ ] Primary SLURM eval/train scripts use CLI for manifest and default model paths.
- [ ] README updated for breaking change.
- [ ] `uv run pytest` green.

## Out of Scope (this phase)

- Eval pipeline modules (Phase C).
- YOLO manifest-only staging (Phase D).
- Automatic validation of every preprocessing script path.

## Further Notes

- `variants.yaml` header comments document slug mismatches; CLI should use `slugs.job` for watershed dirs, variant key for manifest filenames.

## Comments
