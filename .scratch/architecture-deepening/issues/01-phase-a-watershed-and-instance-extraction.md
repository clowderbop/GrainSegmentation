Status: ready-for-agent
Category: enhancement
Labels: ready-for-agent, enhancement
Phase: A
Depends-on: —
Blocks: 03-phase-c-eval-pipelines

# Phase A: Watershed params & instance extraction

Parent: [PRD overview](../PRD.md)

## Problem Statement

Watershed tuning writes JSON with `best_params`, but eval bridges that JSON through bash helpers, a standalone argv emitter script, and duplicate inline Python in whole-section eval shell. Connected-components and watershed extraction are invoked from separate CLIs and tuning loops without a shared facade, so production extraction can diverge from tuning. Semantic class IDs are implicit magic numbers scattered across modules.

## Solution

Move watershed parameter types and strict JSON resolution into the shared package. Add a single instance-extraction facade dispatching to connected components (shared) and watershed (U-Net). Wire extract and tune entry points through both. Remove bash and standalone argv bridging. Add semantic class constants used by tune, extract, and metrics.

## User Stories

1. As a researcher running U-Net eval, I want watershed hyperparameters loaded from the tune JSON in one strict code path, so that eval uses exactly what tuning optimized.
2. As a maintainer, I want watershed JSON parsing and tune-directory resolution implemented only in Python, so that bash does not duplicate schema knowledge.
3. As a researcher, I want missing watershed tune artifacts to fail loudly on production eval, so that I do not publish metrics with silent CLI defaults.
4. As a maintainer, I want connected-components and watershed extraction behind one facade interface, so that tuning and batch extraction cannot diverge.
5. As a maintainer, I want semantic class IDs (background, grain interior, grain boundary) defined once, so that tune, extract, and metrics agree on class semantics.
6. As an AFK agent, I want unit tests on watershed param loading and instance extraction without TIFF or model I/O, so that regressions are caught in CI.

## Implementation Decisions

### New modules (shared package)

- **Semantic class constants** — frozen integer IDs for background, grain interior, grain boundary.
- **Watershed params** — `WatershedParamSet` dataclass (relocated from watershed tuning), `load_best_params(json_path)`, strict `resolve_tune_json(variant, tune_root, explicit_path?, model_context?)`, conversion to extract kwargs or namespace.
- **Instance extraction facade** — `extract_instance_label_map(semantic, method, …)`; connected-components implementation stays in shared code; watershed calls U-Net watershed implementation internally (facade may live in shared with U-Net import only for watershed branch, or facade in U-Net with CC in shared — prefer shared facade importing U-Net watershed function to keep one call site for tune+extract).

### Modified modules (U-Net package)

- **Extract instances** — `--watershed-json` optional path; load via watershed params module; call facade.
- **Watershed tuning** — import `WatershedParamSet` from shared; use facade in AJI loop.

### Remove

- Standalone watershed JSON-to-argv script.
- Bash `build_watershed_extract_args` (delete or no-op with deprecation comment until Phase C SLURM rewrite).
- Inline JSON logging heredoc in whole eval shell (replace with Python logging when touched, or leave for Phase C).

### Strictness

- `resolve_tune_json` in strict mode: missing tune directory or JSON → raise with clear message.
- No silent fallback to CLI defaults for production callers.

### Not in this phase

- Eval pipeline orchestration (Phase C).
- Variant path CLI subcommands (Phase B) — tune dir resolution may duplicate slug logic briefly; align in Phase B.

## Testing Decisions

- **Good tests:** load valid/invalid JSON; strict resolve fails when dir empty; facade output shape and label count on tiny synthetic semantic arrays; CC vs watershed branches produce distinct maps on fixture.
- **Modules tested:** watershed params, instance extraction facade, semantic constants (via facade tests).
- **Prior art:** shared package manifest tests; U-Net semantic metrics tests.
- **Not in CI:** full TIFF pipelines, TensorFlow, SLURM.

## Acceptance criteria

- [ ] `WatershedParamSet` defined once in shared package; tuning imports it.
- [ ] `load_best_params` and strict `resolve_tune_json` unit-tested.
- [ ] Facade used by extract CLI and tune loop; no duplicate watershed kwargs validation in two places.
- [ ] Semantic class constants used in tune, extract, and at least one metrics path.
- [ ] JSON-to-argv script removed; tests pass.
- [ ] `uv run pytest` green for shared and U-Net test paths.

## Out of Scope (this phase)

- Eval profiles or SLURM thinning.
- YOLO changes.
- Registry CLI extensions.

## Further Notes

- Whole eval bash may still call old watershed helper until Phase C; Phase A must not break existing scripts — either keep thin bash wrapper calling Python one-liner or update callers in same PR if small.

## Comments
