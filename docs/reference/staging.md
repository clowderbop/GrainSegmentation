# Staging on cluster nodes

**Full staging** copies every file a dataset manifest references into a local work directory and rewrites the manifest so paths are local (`path_base: "work_root"`). Downstream jobs must not depend on scratch or ephemeral paths once staged. See ADR [0002-self-contained-manifest-staging](../adr/0002-self-contained-manifest-staging.md) and **Staging** in [`CONTEXT.md`](../../CONTEXT.md).

**Metadata-only staging** (`stage_manifest run --metadata-only`) rewrites manifest paths without copying rasters or GT. Use only for preds-only U-Net watershed tune jobs that read semantic preds from scratch and pass `--gt-gpkg` explicitly (see ADR 0002).

## Basic pattern

SLURM jobs copy only manifest-listed files to `$TMPDIR`, then pass `--manifest` to Python:

```bash
uv run python -m common.stage_manifest run \
  "$GRAINSEG_ROOT/dataset/train/manifests/PPL+AllPPX.whole.json" \
  "$TMPDIR/unet_inputs"
```

The staged manifest under `$TMPDIR` uses `path_base: "work_root"` and relative paths beneath the work directory.

## Eval manifests

After inference, jobs call `stage_manifest write-eval` to produce an **eval manifest** with paths to predictions and ground truth, all resolvable under the job work directory. Instance metrics, overlays, and mask AP read the same canonical **instance prediction set** JSON (no second merge at eval for YOLO).

## U-Net whole-section eval

Whole-section eval uses `SLURM/unet/whole_eval_models.tsv` (model basename + variant per row) and `--manifest-split train|test` with staged whole manifests from `dataset/{train,test}/manifests/{variant}.whole.json`.

## Patch and train jobs

Patch training and eval require patch manifests under `dataset/{train,test}/patches/{variant}/manifest.json` (and `dataset/test/unet_from_yolo/{variant}/manifest.json` for U-Net patch test). Jobs do **not** scan dataset directories for TIFFs.

## Manual smoke test

After manifests exist and U-Net models are trained, compare connected components vs watershed on the train section:

```bash
sbatch SLURM/unet/submit_cc_vs_watershed_train_eval.sh
```

Both jobs evaluate all four variants via staged whole manifests. Outputs: `eval/instance_val_cc/` and `eval/instance_val_watershed/`.

## Profile selection

YOLO **profile selection** follows the same principle—jobs must not read large tune caches from scratch during scoring—but uses a different layout than manifest `stage_manifest run`:

| Artifact | Where it lives | Staging |
|----------|----------------|---------|
| Durable caches (`gt_cache/train/`, `{variant}/tiled_proposals/...`) | `OUTPUT_DIR/.cache/` on scratch | Copied per job into job-unique `$TMPDIR` (candidates via `yolo.profile_tune_cache_stage`; detectors stage only the train whole stacked TIFF) |
| Grid rows, `results.csv`, `winner.json` | `OUTPUT_DIR/grid/` on scratch | **Not** staged; written and resumed directly on scratch for parallel array tasks and audit |

Candidate scoring therefore uses `--work-root` under `$TMPDIR` while detector jobs still write canonical per-key proposal caches back to scratch `.cache/`. See [`docs/runbooks/yolo.md`](../runbooks/yolo.md#profile-selection) and [ADR 0005](../adr/0005-yolo-inference-profile-train-selection.md).
