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

2. **`submit_inference_profile_tune.sh`** → **`run_inference_profile_tune.sh`** (rare)
   Staged **profile selection** on the **train** whole section after **all registry** variant weights exist and after **score merge** at predict is in use. Maximizes mean train whole-section **AJI** on canonical **instance prediction sets**, averaged across registry variants; promotes one shared **YOLO inference profile** into `configs/test_inference.yaml` via `yolo.promote_inference_profile`. Re-run when train labels or YOLO weights change materially—not after every single-variant training job. Audit tables: `runs/yolo_inference_profile_tune/<job_id>/stage{1,2}/results.csv` and `winner.json`. Stage 1 holds `conf` and `mask_threshold` from the committed test inference recipe (`configs/test_inference.yaml`); `configs/yolo_inference_profile_tune.yaml` lists only the SAHI merge search axes (stage 2 lists `conf` and `mask_threshold`).

3. **`submit_test_evaluations.sh`** → **`run_patch_test_eval.sh`** + **`run_sahi_test_eval.sh`**
   For each registry variant (`PPL`, `PPLPPXblend`, `PPL+PPXblend`, `PPL+AllPPX`), submits two jobs:
   - **Patch eval** — non-overlapping 1024×1024 test patches via `dataset/test/patches/{variant}/manifest.json`.
   - **Whole eval (SAHI)** — sliding-window inference on the held-out test mosaic; manifest written at job start from registry + test layout.
   - Requires `runs/yolo26-seg/{variant}/weights/best.pt` and `dataset/test/test_labels.gpkg`.
   - Outputs: `eval/yolo_patches/{variant}/{job_id}/` (`prediction_sets/*.json`, `instance_metrics.json`), `eval/yolo_{variant}/` for whole SAHI (`prediction_sets/*.json`, `run_provenance.json`, `instance_metrics.json`, `mask_ap_metrics.json`; override with `OUTPUT_ROOT` / `SAHI_OUT` env on the run scripts).

## Variant memory

Training (`submit_tune_or_train_variants.sh`):

| Variant | `sbatch --mem` |
|---------|----------------|
| `PPL`, `PPLPPXblend` | 200G |
| `PPL+PPXblend` | 350G |
| `PPL+AllPPX` | 1000G |

Whole-section SAHI test (`submit_test_evaluations.sh` → `run_sahi_test_eval.sh`).
Peak RSS ~12–15G on the current test mosaic; **score merge** at predict decodes one
proposal mask at a time (`yolo_detections_to_instance_map_by_score`).

| Job | `sbatch --mem` | `sbatch --time` |
|-----|----------------|-----------------|
| SAHI whole (all variants) | 32G | 00:20:00 |
| Patch test (`run_patch_test_eval.sh`) | 32G | 00:08:00 |

`sbatch --mem` / `--time` on the command line override `#SBATCH` in the run script.

## Example commands

```bash
# Tune all variants
bash SLURM/yolo/submit_tune_or_train_variants.sh --all --tune

# Train all variants with baked-in best hyperparameters (after tuning)
bash SLURM/yolo/submit_tune_or_train_variants.sh --all

# Profile selection on train (after all variant weights exist)
bash SLURM/yolo/submit_inference_profile_tune.sh
# Promote stage2 winner into the committed recipe (then commit configs/test_inference.yaml)
uv run --directory src/yolo python -m yolo.promote_inference_profile \
  --winner-json "$SCRATCH/GrainSeg/runs/yolo_inference_profile_tune/<job_id>/stage2/winner.json"

# Test patch + whole (SAHI) for all variants
bash SLURM/yolo/submit_test_evaluations.sh
```

Direct run (one variant, after weights exist):

```bash
export VARIANT='PPL+AllPPX'
sbatch --export=ALL,VARIANT SLURM/yolo/run_patch_test_eval.sh
sbatch --export=ALL,VARIANT SLURM/yolo/run_sahi_test_eval.sh
```

## Upstream / downstream

| Inputs | From |
|--------|------|
| `dataset/*/patches/{variant}/`, YOLO `data.yaml` | `SLURM/preprocessing/create_patch_datasets.sh` |
| `test_labels.gpkg` | Preprocessing split + labels |

| Outputs | Used for |
|---------|----------|
| `runs/yolo26-seg/{variant}/weights/best.pt` | Profile tune (train) and patch/SAHI test eval |
| `runs/yolo_inference_profile_tune/<job_id>/` | Staged profile-selection audit (results CSV, winners) |
| `configs/test_inference.yaml` | Shared test recipe (YOLO profile promoted from tune) |
| `eval/yolo_patches/`, `eval/yolo_{variant}/` | Metrics comparison with U-Net (`instance_metrics.json`, `mask_ap_metrics.json` on whole eval) |

Whole SAHI and patch predict write **`prediction_sets/{sample_id}.json`** as the canonical **instance prediction set** (non-overlapping grains after **score merge** at predict, each with **score**). **`run_provenance.json`** records `score_merge_at_predict: true` plus the resolved **YOLO inference profile** for that run: **`conf`** and **`mask_threshold`** on both paths; whole also records **`postprocess_type`**, **`match_metric`**, **`match_threshold`**, plus **`slice_height`**, **`slice_width`**, **`overlap_height_ratio`**, **`overlap_width_ratio`** and **`imgsz`**; patch records **`imgsz`** only (no SAHI merge fields). Pre-change eval trees on scratch are invalid — delete `eval/yolo_{variant}/` and `eval/yolo_patches/{variant}/*/` and re-run `bash SLURM/yolo/submit_test_evaluations.sh` before refreshing reporting. Neither path writes `instances/*_instances.tif` or `masks/*.npz`. Eval uses `stage_manifest write-eval` → `eval_manifest.json`; instance metrics, overlays, and mask AP read the same merged JSON (no second merge at eval).

**Note:** U-Net patch test crops (`dataset/test/unet_from_yolo/`) are derived from YOLO patch geometry but are consumed by `SLURM/unet/`, not this pipeline.
