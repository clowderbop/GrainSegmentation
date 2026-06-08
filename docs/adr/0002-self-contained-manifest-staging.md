# Self-contained manifest staging for cluster eval

Status: accepted

## Context / Problem

SLURM jobs stage dataset manifests into `$TMPDIR` so inference and evaluation read local files. Partial staging (images only, ground truth left on scratch) caused patch eval to fail with missing `gt_txt` under the work root. Whole-image YOLO predict also wrote `prediction_sets/{tiff_stem}.json` while manifests and eval used **sample id** `test` / `train`, breaking SAHI visualization and metrics.

## Decision

**Full staging** (`stage_manifest run`) copies every file referenced in manifest sample rows (`image` / `images`, `mask`, `gt_txt`, `gt_gpkg`, etc.) into a local work directory (typically `$TMPDIR`) so inference reads fast local files. Ground-truth paths keep their manifest-relative directory tree under `work_root` (e.g. `dataset/test/patches/.../labels/test/...`). Images may still be flattened to `{stem}.tif` at the work root; **sample id** remains the stable key (`train` / `test` for whole sections, patch stem for patches). Prediction sets are always `prediction_sets/{sample_id}.json`. Staging is for inference I/O; it is not the eval work directory once the run output tree exists.

**Metadata-only staging** (`stage_manifest run --metadata-only` / `stage_manifest_metadata`) rewrites `grainseg_root` and image path fields under `work_root` but does **not** copy rasters or ground-truth assets. The staged manifest is therefore **not** self-contained for GT or channel images. Use this mode only when downstream Python entry points read cached artifacts from scratch and supply GT out-of-band:

- **U-Net watershed tune** (`run_watershed_tuning.sh`): semantic prediction TIFFs live under `$PREDS_DIR` on scratch; the SLURM wrapper copies the GT GeoPackage separately to `$TMPDIR` and passes `--gt-gpkg` explicitly. Tune sample collection (`collect_manifest_tune_samples`) needs sample ids and channel counts from the manifest, not local microscopy rasters.
- Do **not** use metadata-only staging for predict, eval, or any job that must resolve all manifest paths under `work_root` without extra CLI overrides.

**Eval manifests** live under the run output directory (`$RUN_DIR` / eval manifest parent). That directory is the sole work directory for evaluation: every path eval reads — ground truth, anchor image, instance prediction sets — must resolve under it. `stage_manifest write-eval` calls `materialize_eval_assets` to copy ground truth (`gt_gpkg`, `gt_txt`) and anchor images out of ephemeral staging when they are not already local, mirroring staging layout (flat image at run root; GT keeps manifest-relative tree). Eval consumers (`common.evaluate_instances`, `yolo.evaluate_mask_ap`, overlay export) validate both GT and image paths against `grainseg_root`. `--gt-gpkg` overrides materialize on demand when outside the manifest parent; scratch `grainseg_root` paths that bypass staging are rejected.

## Rejected Alternatives

Images-only staging with scratch GT resolution; flatten all assets including GT next to images; whole-section **sample id** = mosaic stem; eval manifests pointing at `$TMPDIR` while prediction sets live on scratch. These were rejected because they are fragile on compute nodes, duplicate split semantics across variants, or depend on ephemeral paths after jobs finish.

Metadata-only staging for predict/eval jobs (relying on scratch paths in the staged manifest) was rejected for those workflows; it is accepted only for preds-only tune where GT and preds are wired explicitly outside the manifest.

## Consequences

`common.stage_manifest` owns `materialize_eval_assets` and `stage_manifest_metadata`. Eval CLIs validate image and GT paths against the eval work root. Metadata-only tune staging requires the SLURM wrapper GT copy and `--gt-gpkg`; that is intentional, not a workaround to remove.

## Links

- Manifest contract: [`docs/manifests.md`](../manifests.md)
- Staging reference: [`docs/reference/staging.md`](../reference/staging.md)
- Glossary: [`CONTEXT.md`](../../CONTEXT.md) (**Staging**, **Sample id**, **Eval manifest**)
