# Preprocessing pipeline

Run from the repo root (`GrainSegmentation/`). Persistent data lives under `$(grainseg_root)/dataset/` (see `SLURM/utils/paths.sh`; uses `$SCRATCH/GrainSeg` when `SCRATCH` is set).

## Order

1. **`download_source_data.sh`** — Download `uncropped.tar.lz4` from Google Drive → extract to `dataset/uncropped/` (`PPL.tif`, `PPX*.tif`, `train_raw.gpkg`, `test_raw.gpkg`). Archive is cached as `dataset/uncropped.tar.lz4`.
2. **`generate_sam2_starting_masks.sh`** — (Optional, for regenerating data) PPL sliding-window SAM2 → `GrainSeg/out/*.geojson` (starting polygons).
3. **Manual (QGIS)** — (Optional, for regenerating data) Correct/refine polygons, fix overlaps offline, export train/test sections → `dataset/uncropped/train_raw.gpkg`, `test_raw.gpkg`, and full-section `PPL.tif` / `PPX*.tif`.
4. **`split_overlaps_and_crop_train_test.sh`**
  `uncropped/*.gpkg` + mosaics → `dataset/train/train_{PPL,PPX1..6}.tif`, `train_labels.gpkg`, and the same under `dataset/test/` (`test_*`).
5. **`create_ppx_and_ppl_ppx_blends.sh`**
  Per-channel TIFFs → `train_PPXblend.tif`, `train_PPLPPXblend.tif` (and `test_*`).
6. **`create_multichannel_input_tiffs.sh`**
  Stacks channels → `train_PPL+PPXblend.tif`, `train_PPL+AllPPX.tif` (and `test_*`). Needs step 5 (`*_PPXblend`) and step 4 (`train_PPL`, `train_PPX1..6`).
7. **`rasterize_labels.sh`** (submits **`rasterize_polygons.sh`**)
  `train_labels.gpkg` + `train_PPL.tif` → `train_labels.tif`; `test_labels.gpkg` + `test_PPL.tif` → `test_labels.tif`. Required before U-Net training.
8. **`create_patch_datasets.sh`**
  Full mosaics + `*_labels.gpkg` → `dataset/train/patches/{PPL,PPLPPXblend,PPL+PPXblend,PPL+AllPPX}/` and `dataset/test/patches/...` (YOLO train/val/test layouts). Also writes patch manifests, YOLO `data.yaml` files, test `unet_from_yolo/{variant}/` crops, and `dataset/{train,test}/manifests/{variant}.whole.json` when run end-to-end.

9. **Regenerate manifests only** (optional, after manual file changes without re-patchifying)

  From the repo root:

  ```bash
  export GRAINSEG_ROOT="${SCRATCH:-/scratch/$USER}/GrainSeg" # or export SCRATCH first

  uv run --directory src/data_prep python write_whole_manifests.py \
    --grainseg-root "$GRAINSEG_ROOT"

  uv run --directory src/data_prep python write_patch_manifests.py \
    --grainseg-root "$GRAINSEG_ROOT" \
    --write-yolo-yamls \
    --write-unet-manifests
  ```

  See `docs/manifests.md` and README “Dataset contracts”.

## Downstream use


| Outputs                                   | Used by                                  |
| ----------------------------------------- | ---------------------------------------- |
| `train_*.tif`, `train_labels.tif`         | U-Net tune/train (`SLURM/unet/`)         |
| `test_*.tif`, `test_labels.gpkg` / `.tif` | U-Net and YOLO test eval                 |
| `*/patches/*`, `*/manifests/*.whole.json` | YOLO tune/train, patch/whole eval (manifest-driven) |
| `test/unet_from_yolo/*`                   | U-Net patch test eval                    |


**`export_prediction_rasters_to_polygons.sh`** is not part of this chain; it converts prediction rasters to GeoPackage after inference.
