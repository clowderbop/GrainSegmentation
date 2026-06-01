# Profile tune: OpenCV GT + fast proposal scoring

Implement [ADR 0006](../../docs/adr/0006-gpkg-ground-truth-rasterization.md) and [ADR 0007](../../docs/adr/0007-profile-selection-proposal-cache-and-scoring.md) in one salvage wave.

**Goal:** Profile selection grid (detector cache → GT cache → candidate array → finalize) completes on train-scale data without multi-hour candidate timeouts; ground truth rasterization uses OpenCV polygon painting as the single canonical GPKG path. Candidate tasks load shared train GT once per job (not per variant) and log fine-grained phase timings for operability.

**Salvage:** Delete entire prior `runs/yolo_inference_profile_tune/<run_id>/`; submit fresh `RUN_ID`; do not reuse v1 proposal caches.

**Parent context:** [ADR 0005](../../docs/adr/0005-yolo-inference-profile-train-selection.md) orchestration unchanged; [CONTEXT.md](../../CONTEXT.md) glossary terms **Profile selection ground truth cache**, **Tiled detector proposals**, **Profile selection scoring**, **Merged instance view**.
