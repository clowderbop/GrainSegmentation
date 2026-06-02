# Preprocessing runbook

Builds scratch datasets under `$(grainseg_root)/dataset/` from uncropped source data. Persistent paths: [`docs/reference/scratch-layout.md`](../reference/scratch-layout.md).

**Prerequisites:** Repo checkout, `uv sync` in `src/data_prep` (optional extras per step — see below). Scratch writable at `$SCRATCH/GrainSeg`.

`src/data_prep/pyproject.toml` optional extras: default (patchify, blends, manifests); `--extra download` (Google Drive); `--extra opencv` (rasterize); `--extra raster` (prediction GeoPackages); `--extra sam2` (starting masks; combine with `opencv` for RLE).

## Pipeline overview

```mermaid
flowchart LR
  A[download] --> B[SAM2 optional]
  B --> C[QGIS optional]
  C --> D[split/crop]
  D --> E[blends]
  E --> F[multichannel stacks]
  F --> G[rasterize labels]
  G --> H[patch datasets]
  H --> I[manifests]
```

| Step | Script | Outputs |
|------|--------|---------|
| 1 | `download_source_data.sh` | `dataset/uncropped/`, cached `uncropped.tar.lz4` |
| 2 | `generate_sam2_starting_masks.sh` (optional) | `out/*.geojson` starting polygons |
| 3 | Manual QGIS (optional) | `uncropped/train_raw.gpkg`, `test_raw.gpkg`, mosaics |
| 4 | `split_overlaps_and_crop_train_test.sh` | `dataset/train|test/train_*.tif`, `*_labels.gpkg` |
| 5 | `create_ppx_and_ppl_ppx_blends.sh` | `*_PPXblend.tif`, `*_PPLPPXblend.tif` |
| 6 | `create_multichannel_input_tiffs.sh` | `train_PPL+PPXblend.tif`, `train_PPL+AllPPX.tif` (+ test) |
| 7 | `rasterize_labels.sh` | `train_labels.tif`, `test_labels.tif` |
| 8 | `create_patch_datasets.sh` | `patches/{variant}/`, whole manifests, `unet_from_yolo/` |
| 9 | Manifest-only regen (optional) | See [Regenerate manifests](#regenerate-manifests-only) |

Scripts live under `SLURM/preprocessing/`.

## Workflows

### 1. Download source data

**Script:** `SLURM/preprocessing/download_source_data.sh`

Downloads `uncropped.tar.lz4` from Google Drive, extracts to `dataset/uncropped/` (`PPL.tif`, `PPX*.tif`, `train_raw.gpkg`, `test_raw.gpkg`).

### 2. SAM2 starting masks (optional)

**Script:** `SLURM/preprocessing/generate_sam2_starting_masks.sh`

PPL sliding-window SAM2 → starting polygons for QGIS correction.

### 3. Split overlaps and crop train/test

**Script:** `SLURM/preprocessing/split_overlaps_and_crop_train_test.sh`

`uncropped/*.gpkg` + mosaics → per-channel `dataset/train/train_{PPL,PPX1..6}.tif`, `train_labels.gpkg`, and matching `dataset/test/` files.

### 4. Blends and multichannel stacks

| Script | Needs | Writes |
|--------|-------|--------|
| `create_ppx_and_ppl_ppx_blends.sh` | Step 4 | `*_PPXblend.tif`, `*_PPLPPXblend.tif` |
| `create_multichannel_input_tiffs.sh` | Steps 4–5 | `train_PPL+PPXblend.tif`, `train_PPL+AllPPX.tif` (+ test) |

### 5. Rasterize labels

**Script:** `SLURM/preprocessing/rasterize_labels.sh` (submits `rasterize_polygons.sh`)

`train_labels.gpkg` + `train_PPL.tif` → `train_labels.tif` (and test). Required before U-Net training.

### 6. Patch datasets

**Script:** `SLURM/preprocessing/create_patch_datasets.sh`

Full mosaics + label GeoPackages → `dataset/train/patches/{variant}/`, `dataset/test/patches/...`, YOLO `data.yaml`, `dataset/{train,test}/manifests/{variant}.whole.json`, `dataset/test/unet_from_yolo/{variant}/` when run end-to-end.

## Regenerate manifests only

After manual file changes without re-patchifying ([`docs/reference/scratch-layout.md`](../reference/scratch-layout.md#regenerating-manifests)):

```bash
export GRAINSEG_ROOT="${SCRATCH:-/scratch/$USER}/GrainSeg"

uv run --directory src/data_prep python write_whole_manifests.py \
  --grainseg-root "$GRAINSEG_ROOT"

uv run --directory src/data_prep python write_patch_manifests.py \
  --grainseg-root "$GRAINSEG_ROOT" \
  --write-yolo-yamls \
  --write-unet-manifests
```

## Upstream / downstream

| Outputs | Used by |
|---------|---------|
| `train_*.tif`, `train_labels.tif` | [U-Net runbook](unet.md) |
| `test_*.tif`, `test_labels.gpkg` / `.tif` | YOLO and U-Net test eval |
| `*/patches/*`, `*/manifests/*.whole.json` | [YOLO runbook](yolo.md), U-Net eval |
| `test/unet_from_yolo/*` | U-Net patch test eval |

**Not in this chain:** `export_prediction_rasters_to_polygons.sh` converts prediction rasters to GeoPackage after inference.

**Downstream runbooks:** [yolo.md](yolo.md), [unet.md](unet.md).
