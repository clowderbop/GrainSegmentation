# YOLO pipeline

Run from the repo root (`GrainSegmentation/`). Persistent data lives under `$(grainseg_root)/` (see `SLURM/utils/paths.sh`; uses `$SCRATCH/GrainSeg` when `SCRATCH` is set).

**Prerequisites:** Complete `SLURM/preprocessing/pipeline.md` through patch datasets (`dataset/train/patches/{variant}/` with `manifest.json` and `{variant}.yaml`, and `dataset/test/patches/{variant}/`). YOLO uses **stacked** train/test mosaics (`train_PPL+PPXblend.tif`, etc.) referenced from patch manifests—not the per-channel TIFFs used by U-Net whole manifests.

Jobs stage patch manifests (and listed images/labels) into `$TMPDIR` for train and patch eval. Whole-section SAHI eval builds a YOLO whole manifest at runtime and stages the test mosaic.

## Order

1. **`submit_tune_or_train_variants.sh`** → **`run_tune_or_train_variant.sh`**
   Hyperparameter tuning and/or training per variant on staged `dataset/train/patches/{variant}/`.
   - **Tune:** pass `--tune` (Ultralytics tuner over learning rate and dropout).
   - **Train:** submit script passes tuned `--lr` and `--dropout` per variant when not tuning; use `--all`, `--ppl`, `--ppl-ppx-composite`, `--ppl-plus-ppx-composite`, `--all-ppx`; `--resume`, `--verbose`.
   - Pretrained weights: `pretrained/yolo26l-seg.pt` on scratch.
   - Writes `runs/yolo26-seg/{variant}/weights/best.pt` (and run artifacts under that project dir).

2. **`submit_test_evaluations.sh`** → **`run_patch_test_eval.sh`** + **`run_sahi_test_eval.sh`**
   For each registry variant (`PPL`, `PPLPPXblend`, `PPL+PPXblend`, `PPL+AllPPX`), submits two jobs:
   - **Patch eval** — non-overlapping 1024×1024 test patches via `dataset/test/patches/{variant}/manifest.json`.
   - **Whole eval (SAHI)** — sliding-window inference on the held-out test mosaic; manifest written at job start from registry + test layout.
   - Requires `runs/yolo26-seg/{variant}/weights/best.pt` and `dataset/test/test_labels.gpkg`.
   - Outputs: `eval/yolo_patches/{variant}/{job_id}/`, `eval/yolo_{variant}/` (override with `OUTPUT_ROOT` / `SAHI_OUT` env on the run scripts).

## Variant memory

Training (`submit_tune_or_train_variants.sh`):

| Variant | `sbatch --mem` |
|---------|----------------|
| `PPL`, `PPLPPXblend` | 200G |
| `PPL+PPXblend` | 350G |
| `PPL+AllPPX` | 1000G |

Whole-section SAHI test (`submit_test_evaluations.sh` — higher than train because
`yolo.predict` materializes full-resolution mask stacks):

| Variant | `sbatch --mem` |
|---------|----------------|
| `PPL`, `PPLPPXblend` | 400G |
| `PPL+PPXblend` | 500G |
| `PPL+AllPPX` | 1000G |

`sbatch --mem=…` on the command line overrides `#SBATCH --mem` in the run script.

## Example commands

```bash
# Tune all variants
bash SLURM/yolo/submit_tune_or_train_variants.sh --all --tune

# Train all variants with baked-in best hyperparameters (after tuning)
bash SLURM/yolo/submit_tune_or_train_variants.sh --all

# Test patch + whole (SAHI) for all variants
bash SLURM/yolo/submit_test_evaluations.sh
```

Direct run (one variant, after weights exist):

```bash
export VARIANT='PPL+AllPPX'
sbatch --export=ALL,VARIANT SLURM/yolo/run_patch_test_eval.sh
sbatch --mem=1000G --export=ALL,VARIANT SLURM/yolo/run_sahi_test_eval.sh
```

## Upstream / downstream

| Inputs | From |
|--------|------|
| `dataset/*/patches/{variant}/`, YOLO `data.yaml` | `SLURM/preprocessing/create_patch_datasets.sh` |
| `test_labels.gpkg` | Preprocessing split + labels |

| Outputs | Used for |
|---------|----------|
| `runs/yolo26-seg/{variant}/weights/best.pt` | Patch and SAHI test eval |
| `eval/yolo_patches/`, `eval/yolo_{variant}/` | Metrics comparison with U-Net (`instance_metrics.json`, `mask_ap_metrics.json` on whole eval) |

**Note:** U-Net patch test crops (`dataset/test/unet_from_yolo/`) are derived from YOLO patch geometry but are consumed by `SLURM/unet/`, not this pipeline.
