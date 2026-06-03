# Canonical instance prediction output

Status: accepted

## Context / Problem

Whole-image YOLO inference once materialized dense `(N, H, W)` mask stacks, which was not viable for 10k×10k mosaics. U-Net and YOLO also wrote different instance artifacts, making cross-model evaluation harder to reason about.

## Decision

The canonical output is one **instance prediction set** per sample: JSON (`schema_version: 1`) at `prediction_sets/{sample_id}.json`, with COCO RLE mask geometry and manifest field `instance_prediction_set`.

- **YOLO** writes non-overlapping grains after **score merge** at predict time. Each grain keeps the winning proposal **score**.
- **U-Net** writes non-overlapping **extracted grains** after instance extraction from a **semantic prediction**. U-Net instance entries do not carry scores.
- Evaluation, overlays, and reporting all read the same prediction set. YOLO is not merged a second time at evaluation.
- Run parameters live once in a **run provenance** sidecar beside `prediction_sets/`, not duplicated in every prediction file.

Legacy instance label-map TIFFs, dense mask NPZs, and persisted overlapping YOLO proposal sets are not read or written as canonical outputs.

## Rejected Alternatives

Keep dense NPZs; persist YOLO proposals and merge only at eval; dual-write proposals plus merged grains; use RLE-only metrics without rasterizing. These were rejected for memory, artifact ambiguity, or metric parity reasons.

## Consequences

`common.evaluate_instances`, `yolo.predict`, `unet.extract_instances`, manifest schema, and SLURM eval scripts must agree on this shape. Patch and whole-section samples use the same prediction-set schema; sample unit stays on manifests. Whole-section Mask AP is outside the standard policy.

## Links

- Evaluation policy: [ADR 0003](0003-test-evaluation-policy.md)
- Glossary: [`CONTEXT.md`](../../CONTEXT.md)
- Manifest contract: [`docs/manifests.md`](../manifests.md)
