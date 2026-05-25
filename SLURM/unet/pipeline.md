# U-Net pipeline

Run from the repo root (`GrainSegmentation/`). Persistent data lives under `$(grainseg_root)/` (see `SLURM/utils/paths.sh`; uses `$SCRATCH/GrainSeg` when `SCRATCH` is set).

**Prerequisites:** Complete `SLURM/preprocessing/pipeline.md` through rasterized labels and manifests (`train_labels.tif`, `dataset/train/manifests/{variant}.whole.json`, `dataset/test/manifests/{variant}.whole.json`, `dataset/test/unet_from_yolo/{variant}/manifest.json` for patch eval). See README “Dataset contracts” and `docs/manifests.md`.

Jobs stage **manifest-listed files only** into `$TMPDIR` via `python -m common.stage_manifest run` (see README “Staging on cluster nodes”).

## Order

1. **`submit_tune_and_train_variants.sh`** → **`run_tune_and_train_variant.sh`**
   Bayesian hyperparameter search (optional) + final training per variant. Requires `dataset/train/train_labels.tif` and `dataset/train/manifests/{variant}.whole.json`.
   - Options: `--ppl`, `--ppl-ppx-composite`, `--ppl-plus-ppx-composite`, `--all-ppx`, `--all`; `--resume`, `--skip-tuning`, `--verbose`.
   - Writes `models/unet/unet_finetuned_{variant}.keras` (registry `DEFAULT_MODEL_BASENAME`) and tuning logs under `tuning_logs/{run_name}/`.
   - Pretrained start: `models/unet/pretrained/starting_point.keras` on scratch (or repo `models/pretrained/` fallback).

2. **`submit_watershed_tuning.sh`** → **`run_watershed_tuning.sh`**
   Grid search over watershed postprocessing on the **train** section (sliding-window U-Net predictions vs `train_labels.gpkg`). One job per registry variant; needs finetuned model at `models/unet/{DEFAULT_MODEL_BASENAME}`.
   - `--dry-run` prints `sbatch` commands without submitting.
   - Writes `runs/watershed_tune/{slugs.job}/watershed_best_*.json` (`slugs.job` from `config/variants.yaml`, e.g. `PPL_AllPPX`).

3. **`submit_cc_vs_watershed_train_eval.sh`** → **`run_whole_test_eval.sh`** (×2)
   Compare instance extraction on the **train** section: connected components vs tuned watershed. Uses `whole_eval_models.tsv`, `--manifest-split train`, `--gt-gpkg dataset/train/train_labels.gpkg`.
   - Outputs: `eval/instance_val_cc/`, `eval/instance_val_watershed/`.

4. **Pick instance method** (manual) — Use train-section AJI (and overlays) to choose CC or watershed for downstream test eval.

5. **`submit_whole_test_eval.sh`** → **`run_whole_test_eval.sh`**
   Held-out **test** whole-section eval (default instance method: watershed; reads tune JSONs from `runs/watershed_tune/`).
   - Output: `eval/unet_test/`.
   - Override instance method or paths by calling `run_whole_test_eval.sh` directly (`--instance-method cc|watershed`, `--config-file`, etc.).

6. **`submit_patch_test_eval.sh`** → **`run_patch_test_eval.sh`**
   One job per variant; patch-wise test eval on `dataset/test/unet_from_yolo/{variant}/manifest.json`.
   - Output: `eval/unet_patches/{variant}/{job_id}/` (`instance_metrics.json`, predictions under job dir).

## Config

| File | Role |
|------|------|
| `whole_eval_models.tsv` | Model basename + variant rows for `run_whole_test_eval.sh` |
| `config/variants.yaml` | Channel counts, default model basenames, watershed tune dir slugs (`SLURM/utils/variants.sh`) |

## Example commands

```bash
# Train all four variants (tune + train)
sbatch SLURM/unet/submit_tune_and_train_variants.sh --all

# Watershed tuning (all variants)
bash SLURM/unet/submit_watershed_tuning.sh

# Train-section CC vs watershed comparison
sbatch SLURM/unet/submit_cc_vs_watershed_train_eval.sh

# Test whole + patch eval (after models and watershed JSONs exist)
bash SLURM/unet/submit_whole_test_eval.sh
bash SLURM/unet/submit_patch_test_eval.sh
```

## Upstream / downstream

| Inputs | From |
|--------|------|
| `train_labels.tif`, whole manifests | `SLURM/preprocessing/` |
| `test_labels.gpkg`, `unet_from_yolo` manifests | Preprocessing + `write_patch_manifests.py --write-unet-manifests` |

| Outputs | Used for |
|---------|----------|
| `models/unet/unet_finetuned_*.keras` | Watershed tune, whole/patch eval |
| `runs/watershed_tune/{slug}/` | Whole eval (`--instance-method watershed`) |
| `eval/unet_test/`, `eval/unet_patches/` | Thesis metrics / comparison with YOLO |
