# Instance prediction set as canonical instance output

Status: accepted; whole-section Mask AP references superseded by [ADR 0008](0008-pq-headline-instance-evaluation.md)

Whole-image YOLO inference materialized dense `(N, H, W)` mask stacks, which required hundreds of gigabytes of RAM on 10k×10k test mosaics. U-Net and YOLO also diverged on disk (instance label-map TIFF vs mask NPZ) while sharing only some evaluation steps.

We store one **instance prediction set** per sample: JSON (`schema_version: 1`) under `prediction_sets/{sample_id}.json`, with COCO RLE per entry and manifest field `instance_prediction_set`. **U-Net** writes **extracted grains** (non-overlapping, no score field) after instance extraction from **semantic prediction**. **YOLO** writes the canonical **instance prediction set** (non-overlapping grains after **score merge** at predict, each with **score**); see [ADR 0004](0004-yolo-score-merge-at-predict.md). Instance metrics rasterize the stored set via **merged instance view** (no second **score merge** at eval for YOLO). U-Net keeps a separate **semantic prediction** raster for pixel metrics. Run parameters live in a **run provenance** sidecar beside `prediction_sets/`, not in every file. ADR 0008 removes whole-section Mask AP from the standard evaluation policy.

Legacy instance label-map TIFFs and dense mask NPZs are not written or read (clean break). Domain terms are defined in `CONTEXT.md`.

**Considered options:** Keep dense NPZ; store merged grains only; dual artifacts with legacy readers; RLE-only eval without ever rasterizing (rejected for metric parity and simplicity).

**Consequences:** `common.evaluate_instances`, `yolo.predict`, `unet.extract_instances`, manifest schema, and SLURM eval scripts must move together. Patch and whole use the same prediction set shape; `unit` stays on manifests.

**Superseded (YOLO persistence and eval merge):** Persisting overlapping **detector proposals** on disk and applying **score merge** only at eval — see [ADR 0004](0004-yolo-score-merge-at-predict.md).
