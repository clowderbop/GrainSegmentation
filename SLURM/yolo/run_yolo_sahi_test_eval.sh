#!/bin/bash
#SBATCH --job-name=test_yolo
#SBATCH --output=logs/test_yolo-%j.log
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --time=04:00:00

set -euo pipefail
# shellcheck source=SLURM/utils/source_job.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../utils/source_job.sh"
# shellcheck source=SLURM/utils/assertions.sh
source "$SLURM_ROOT/utils/assertions.sh"
# shellcheck source=SLURM/utils/variants.sh
source "$SLURM_ROOT/utils/variants.sh"
# shellcheck source=SLURM/utils/manifest_shell.sh
source "$SLURM_ROOT/utils/manifest_shell.sh"

GRAINSEG_ROOT="$(grainseg_root)"

VARIANT="${VARIANT:-PPL}"
DEVICE="0"
SLICE_H=1024
SLICE_W=1024
OV_H=0.5
OV_W=0.5

TEST_GPKG="$GRAINSEG_ROOT/dataset/test/test_labels.gpkg"

source "$SLURM_ROOT/prepare_env.sh"

WEIGHTS="$GRAINSEG_ROOT/runs/yolo26-seg/$VARIANT/weights/best.pt"
require_file "$WEIGHTS" "YOLO weights not found"
require_file "$TEST_GPKG" "Test labels GeoPackage not found"

SAHI_OUT="${SAHI_OUT:-$GRAINSEG_ROOT/eval/yolo_${VARIANT}}"
OUT_ROOT="${OUTPUT_ROOT:-$SAHI_OUT}"
INSTANCE_METRICS_JSON="$OUT_ROOT/instance_metrics.json"
MASK_AP_JSON="$OUT_ROOT/mask_ap_metrics.json"

WORK_ROOT="$TMPDIR/yolo_sahi_${VARIANT}_${SLURM_JOB_ID:-local}"
YOLO_MANIFEST="$WORK_ROOT/yolo_whole.json"
STAGED="$WORK_ROOT/staged"
rm -rf "$WORK_ROOT"
mkdir -p "$WORK_ROOT" "$OUT_ROOT"

echo "Writing YOLO whole-section manifest for $VARIANT ..."
write_yolo_whole_manifest_json "$VARIANT" test "$GRAINSEG_ROOT" "$YOLO_MANIFEST"
require_file "$YOLO_MANIFEST" "YOLO whole manifest not written"

echo "Staging test mosaic to TMPDIR..."
uv run --directory "$REPO_ROOT" python -m common.stage_manifest run \
    "$YOLO_MANIFEST" "$STAGED"
STAGED_MANIFEST="$STAGED/manifest.json"
require_file "$STAGED_MANIFEST" "Staged YOLO whole manifest missing"

echo "Syncing YOLO environment..."
cd "$REPO_ROOT/src/yolo"
uv sync

export YOLO_DISABLE_TQDM=True

echo "1/4 yolo.predict (whole-image SAHI instance TIFFs and mask NPZs)..."
uv run python -u -m yolo.predict \
    --unit whole \
    --weights "$WEIGHTS" \
    --variant "$VARIANT" \
    --manifest "$STAGED_MANIFEST" \
    --device "$DEVICE" \
    --imgsz "$SLICE_H" \
    --conf "${CONF:-0.25}" \
    --slice-height "$SLICE_H" \
    --slice-width "$SLICE_W" \
    --overlap-height-ratio "$OV_H" \
    --overlap-width-ratio "$OV_W" \
    --output-dir "$OUT_ROOT"

EVAL_MANIFEST="$OUT_ROOT/eval_manifest.json"
echo "Building eval manifest..."
uv run --directory "$REPO_ROOT" python -m common.stage_manifest write-eval \
    --source "$STAGED_MANIFEST" \
    --pred-instances-dir "$OUT_ROOT/instances" \
    --output "$EVAL_MANIFEST" \
    --gt-gpkg "$TEST_GPKG"

echo "2/4 yolo.export_sahi_visualization (prediction overlay TIFF)..."
uv run python -u -m yolo.export_sahi_visualization \
    --output-dir "$OUT_ROOT" \
    --pred-dir "$OUT_ROOT" \
    --manifest "$EVAL_MANIFEST"

echo "3/4 common.evaluate_instances..."
uv run python -u -m common.evaluate_instances \
    --unit whole \
    --model-type yolo \
    --variant "$VARIANT" \
    --manifest "$EVAL_MANIFEST" \
    --output-json "$INSTANCE_METRICS_JSON"

echo "4/4 yolo.evaluate_mask_ap (COCO mask AP from pred mask NPZ)..."
uv run python -u -m yolo.evaluate_mask_ap \
    --variant "$VARIANT" \
    --manifest "$EVAL_MANIFEST" \
    --pred-dir "$OUT_ROOT" \
    --output-json "$MASK_AP_JSON"

echo "Wrote $INSTANCE_METRICS_JSON and $MASK_AP_JSON"
