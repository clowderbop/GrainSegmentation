Status: ready-for-agent
Category: enhancement
Labels: ready-for-agent, enhancement
Phase: E
Depends-on: 03-phase-c-eval-pipelines
Blocks: —

# Phase E: Visualization & TF runtime

Parent: [PRD overview](../PRD.md)

## Problem Statement

Whole-section U-Net eval still stages overlay manifests and invokes plotting through ad-hoc bash blocks separate from the eval pipeline. TensorFlow isolation was identified as in-scope for the architecture work, but Phase C only stubs a predict-runtime protocol. Qualitative figures and production predict remain coupled to in-process Keras in the U-Net environment.

## Solution

Fold quantitative and overlay plotting into U-Net whole eval profile hooks where possible. Implement a subprocess (or container-ready) **predict runtime** adapter behind the Phase C protocol, controlled by flag/env, defaulting to in-process until explicitly enabled. Optionally add a small shared visualization helper if it reduces duplication between plot modules.

## User Stories

1. As a researcher, I want whole-section quantitative and overlay figures from the U-Net whole eval profile, so plotting is not a separate bash block.
2. As a maintainer, I want visualization logic consolidated only after metrics pipeline is stable, without blocking correctness.
3. As a maintainer, I want TensorFlow predict isolatable via subprocess or container adapter, so future env separation does not rewrite orchestration.
4. As a maintainer, I want the predict runtime choice documented and off by default until container path is validated.

## Implementation Decisions

### Visualization

- Absorb overlay second-manifest stage into **U-Net whole test** profile (overlay variant parameter preserved).
- Call existing plot results and overlay plot modules from pipeline; avoid new plotting dependencies.
- Optional: thin **visualize eval** helper in shared package if it deduplicates manifest anchor lookup — not required if pipeline calls existing modules directly.

### Predict runtime (U-Net package)

- Implement **subprocess predict runtime** adapter: spawn isolated Python/venv or documented wrapper command with same CLI args as in-process predict.
- Env var or profile flag, e.g. `GRAINSEG_PREDICT_RUNTIME=subprocess|keras`.
- Document requirements (model path, manifest, output dir passed through).
- Container adapter: define interface; implementation may be follow-up if cluster container not ready — subprocess minimum for Phase E acceptance.

### Documentation

- README / runtime-env doc: how to enable subprocess predict, limitations (GPU binding, latency).

## Testing Decisions

- Unit: pipeline whole profile calls plot steps when flags set (mock steps).
- Subprocess runtime: integration test with mock predict script or `--help` stub if full TF too heavy; at minimum test adapter constructs command line from profile args.
- Prior art: SAHI visualization tests in YOLO package.
- Manual: cluster verify subprocess predict on one sample (documented).

## Acceptance criteria

- [ ] Whole eval SLURM/bash no longer contains standalone overlay/plot blocks — handled by pipeline.
- [ ] Subprocess predict runtime implemented and selectable; in-process remains default.
- [ ] Docs describe activation and constraints.
- [ ] Tests green; no regression to Phase C profile outputs when runtime=keras.

## Out of Scope (this phase)

- Full OCI container image build/publish for TF (interface + subprocess acceptable).
- YOLO visualization changes unless required for parity.
- Dashboard or web UI.

## Further Notes

- If subprocess predict is not viable on SLURM GPU binding in one PR, land adapter skeleton + docs and file follow-up issue — prefer working subprocess over empty stub.

## Comments
