# Dataset manifests

JSON inventories list the exact files that constitute each **sample** for a given variant, split, and unit. They complement the variant **registry** (`config/variants.yaml`), which defines what each variant means; manifests define which on-disk paths belong to each run.

Manifests live on scratch under `$SCRATCH/GrainSeg` (see `SLURM/utils/paths.sh`). Paths in scratch manifests use `path_base: "grainseg_root"` and are relative to `grainseg_root`. SLURM jobs copy listed files to `$TMPDIR` via `common.stage_manifest`, which rewrites the manifest with `path_base: "work_root"`.

## Schema (version 1)

Top-level object:

| Field | Required | Description |
|-------|----------|-------------|
| `schema_version` | yes | Must be `1` |
| `variant` | yes | Registry key (`PPL`, `PPL+AllPPX`, …) |
| `unit` | yes | `whole` or `patch` |
| `grainseg_root` | yes | Absolute scratch root when written |
| `path_base` | yes | `grainseg_root` or `work_root` after staging |
| `samples` | yes | Non-empty array of sample rows |

### Sample row

Each row describes one logical sample (one train mosaic, one test mosaic, or one patch).

| Field | Required | Description |
|-------|----------|-------------|
| `sample_id` | yes | Stable id (`train`, `test`, or patch stem) |
| `image` | one of | Single stacked or anchor TIFF (YOLO, eval overlays) |
| `images` | one of | U-Net multi-input channel TIFFs; length must equal `unet.num_inputs` |
| `mask` | no | Semantic raster mask (`.tif`) |
| `gt_gpkg` | no | Vector labels for instance metrics |
| `gt_origin` | no | `whole_image` or `patch_stem` |
| `gt_txt` | no | YOLO segment labels for patch eval |
| `instance_prediction_set` | no | Path to `prediction_sets/{sample_id}.json` in eval manifests after inference |
| `semantic` | no | Optional path to semantic prediction TIFF |

**Rules**

- Use `image` **or** `images`, never both.
- U-Net whole-section manifests list **per-channel** TIFFs only. Do not reference stacked YOLO mosaics (`train_PPL+AllPPX.tif`, etc.).
- `images` length is validated against `config/variants.yaml` → `unet.num_inputs`.
- Whole-section train/test rows typically include `gt_gpkg` and `gt_origin: "whole_image"`. Train rows also include `mask` (raster labels).

### Example (U-Net whole, train)

```json
{
  "schema_version": 1,
  "variant": "PPL+AllPPX",
  "unit": "whole",
  "grainseg_root": "/scratch/user/GrainSeg",
  "path_base": "grainseg_root",
  "samples": [
    {
      "sample_id": "train",
      "images": [
        "dataset/train/train_PPL.tif",
        "dataset/train/train_PPX1.tif",
        "dataset/train/train_PPX2.tif",
        "dataset/train/train_PPX3.tif",
        "dataset/train/train_PPX4.tif",
        "dataset/train/train_PPX5.tif",
        "dataset/train/train_PPX6.tif"
      ],
      "mask": "dataset/train/train_labels.tif",
      "gt_gpkg": "dataset/train/train_labels.gpkg",
      "gt_origin": "whole_image"
    }
  ]
}
```

## On-disk layout

| Unit | Path (relative to grainseg root) |
|------|----------------------------------|
| U-Net train whole | `dataset/train/manifests/{variant}.whole.json` |
| U-Net test whole | `dataset/test/manifests/{variant}.whole.json` |
| YOLO / patch eval | `dataset/{train,test}/patches/{variant}/manifest.json` |
| U-Net patch test | `dataset/test/unet_from_yolo/{variant}/manifest.json` |

Replace `{variant}` with the registry key literally (including `+`).

## Generating manifests

From the repo root after preprocessing (see [`docs/runbooks/preprocessing.md`](runbooks/preprocessing.md#regenerate-manifests-only)):

```bash
GRAINSEG_ROOT="${SCRATCH:-/scratch/$USER}/GrainSeg"

uv run --directory src/data_prep python write_whole_manifests.py \
  --grainseg-root "$GRAINSEG_ROOT"

uv run --directory src/data_prep python write_patch_manifests.py \
  --grainseg-root "$GRAINSEG_ROOT" \
  --write-yolo-yamls

uv run --directory src/data_prep python validate_dataset_manifests.py \
  --grainseg-root "$GRAINSEG_ROOT"
```

## Staging

Staging is **self-contained**: every file referenced in sample rows is copied into the work directory and the manifest is rewritten with `path_base: "work_root"` (see ADR-0002). Ground-truth paths (`gt_txt`, `gt_gpkg`) keep their manifest-relative directory tree under the work root. Images are often flattened to `{stem}.tif` at the work root; **sample id** is unchanged.

```bash
uv run python -m common.stage_manifest run \
  "$GRAINSEG_ROOT/dataset/train/manifests/PPL+AllPPX.whole.json" \
  "$TMPDIR/unet_inputs"
```

Writes `$TMPDIR/unet_inputs/manifest.json` and copies referenced assets. Python CLIs take `--manifest` pointing at the staged file.

Eval manifests after instance extraction are built with:

```bash
uv run python -m common.stage_manifest write-eval \
  --source "$TMPDIR/unet_inputs/manifest.json" \
  --prediction-set-dir "$RUN_DIR" \
  --output "$RUN_DIR/eval_manifest.json"
```

`write-eval` materializes ground truth and anchor images into `$RUN_DIR` when they still live under ephemeral staging (see ADR-0002). Layout mirrors staging: anchor image flattened to `$RUN_DIR/{stem}.tif`; GT keeps manifest-relative paths under `$RUN_DIR`. The eval manifest sets `grainseg_root` to `$RUN_DIR` and `path_base: "work_root"`.

YOLO patch eval writes one JSON per sample under `$RUN_DIR/prediction_sets/` (see ADR-0001). `write-eval` records relative paths in `instance_prediction_set`.

```bash
# YOLO patch test eval (prediction sets only; no instances/*.tif)
uv run python -m yolo.predict --unit patch --weights ... --manifest ... --output-dir "$RUN_DIR"
uv run python -m common.stage_manifest write-eval \
  --source "$STAGED_MANIFEST" \
  --prediction-set-dir "$RUN_DIR" \
  --output "$RUN_DIR/eval_manifest.json"
```

Optional `--gt-gpkg` overrides each sample's `gt_gpkg` from the source manifest; the path is materialized into `$RUN_DIR` when not already there. Scratch `grainseg_root` paths that bypass staging are rejected. Eval consumers validate GT and image paths under `grainseg_root`.

## Validation

`validate_dataset_manifest` (in `common.manifest_io`) checks schema version, variant registry consistency, `image` vs `images`, and stacked-mosaic exclusions. `src/data_prep/validate_dataset_manifests.py` walks all expected manifest paths on a scratch tree.
