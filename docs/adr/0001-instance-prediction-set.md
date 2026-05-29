# Instance prediction set as canonical instance output

Status: accepted

Whole-image YOLO inference materialized dense `(N, H, W)` mask stacks for COCO mask AP, which required hundreds of gigabytes of RAM on 10k×10k test mosaics. U-Net and YOLO also diverged on disk (instance label-map TIFF vs mask NPZ) while sharing only some evaluation steps.

We store one **instance prediction set** per sample: JSON (`schema_version: 1`) under `prediction_sets/{sample_id}.json`, with COCO RLE per entry and manifest field `instance_prediction_set`. YOLO writes **detector proposals** (overlapping, with **confidence**); U-Net writes **extracted grains** (non-overlapping, no confidence). Instance metrics build a transient **merged instance view** via **confidence merge** for YOLO; **mask AP** reads proposals directly and runs on whole-section YOLO test samples only. U-Net keeps a separate **semantic prediction** raster for pixel metrics. Run parameters live in a **run provenance** sidecar beside `prediction_sets/`, not in every file.

Legacy instance label-map TIFFs and dense mask NPZs are not written or read (clean break). Domain terms are defined in `CONTEXT.md`.

**Considered options:** Keep dense NPZ; store merged grains only (breaks mask AP); dual artifacts with legacy readers; RLE-only eval without ever rasterizing (rejected for metric parity and simplicity).

**Consequences:** `common.evaluate_instances`, `yolo.predict`, `unet.extract_instances`, manifest schema, and SLURM eval scripts must move together. Patch and whole use the same prediction set shape; `unit` stays on manifests.
