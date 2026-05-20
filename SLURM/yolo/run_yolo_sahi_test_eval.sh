#!/bin/bash
#SBATCH --job-name=test_yolo
#SBATCH --output=logs/test_yolo-%j.log
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --time=04:00:00

set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_ROOT="$(cd "$THIS_DIR/.." && pwd)"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$SLURM_ROOT/.." && pwd)}"
cd "$REPO_ROOT"

VARIANT="${VARIANT:-PPL}"
DEVICE="0"
SLICE_H=1024
SLICE_W=1024
OV_H=0.5
OV_W=0.5

MANIFEST=""
TEST_GPKG="$SCRATCH/GrainSeg/dataset/test/test_labels.gpkg"

source "$SLURM_ROOT/prepare_env.sh"

case "$VARIANT" in
    PPL)
        TEST_TIFF="$SCRATCH/GrainSeg/dataset/test/test_PPL.tif"
        ;;
    PPLPPXblend)
        TEST_TIFF="$SCRATCH/GrainSeg/dataset/test/test_PPLPPXblend.tif"
        ;;
    PPL+PPXblend)
        TEST_TIFF="$SCRATCH/GrainSeg/dataset/test/test_PPL+PPXblend.tif"
        ;;
    PPL+AllPPX)
        TEST_TIFF="$SCRATCH/GrainSeg/dataset/test/test_PPL+AllPPX.tif"
        ;;
    *)
        echo "Unknown YOLO variant: $VARIANT" >&2
        exit 1
        ;;
esac

WEIGHTS="$SCRATCH/GrainSeg/runs/yolo26-seg/$VARIANT/weights/best.pt"
SAHI_OUT="${SAHI_OUT:-$SCRATCH/GrainSeg/eval/yolo_${VARIANT}}"
OUT_ROOT="${OUTPUT_ROOT:-$SAHI_OUT}"
INSTANCE_METRICS_JSON="$OUT_ROOT/instance_metrics.json"
MASK_AP_JSON="$OUT_ROOT/mask_ap_metrics.json"

mkdir -p "$OUT_ROOT"

if [[ -z "$MANIFEST" ]]; then
    echo "Staging test TIFF to TMPDIR..."
    TMP_TEST_ROOT="${TMPDIR}/test_yolo"
    mkdir -p "$TMP_TEST_ROOT"
    TIFF_BASENAME="$(basename "$TEST_TIFF")"
    cp -f "$TEST_TIFF" "$TMP_TEST_ROOT/$TIFF_BASENAME"
    TEST_TIFF="$TMP_TEST_ROOT/$TIFF_BASENAME"
fi

echo "Syncing YOLO environment..."
cd "$REPO_ROOT/src/yolo"
uv sync

export YOLO_DISABLE_TQDM=True

PREDICT_CMD=(
    uv run python -u -m yolo.predict
    --unit whole
    --weights "$WEIGHTS"
    --variant "$VARIANT"
    --device "$DEVICE"
    --imgsz "$SLICE_H"
    --conf "${CONF:-0.25}"
    --slice-height "$SLICE_H"
    --slice-width "$SLICE_W"
    --overlap-height-ratio "$OV_H"
    --overlap-width-ratio "$OV_W"
    --output-dir "$OUT_ROOT"
)

if [[ -n "$MANIFEST" ]]; then
    PREDICT_CMD+=(--manifest "$MANIFEST")
else
    PREDICT_CMD+=(--test-tiff "$TEST_TIFF")
fi

echo "1/4 yolo.predict (whole-image SAHI labels)..."
"${PREDICT_CMD[@]}"

MANIFEST_PATH="$OUT_ROOT/eval_manifest.json"
if [[ -n "$MANIFEST" ]]; then
    MANIFEST_PATH="$MANIFEST"
else
    SAMPLE_ID="$(basename "$TEST_TIFF" .tif)"
    SAMPLE_ID="${SAMPLE_ID%.tiff}"
    python3 - "$MANIFEST_PATH" "$SAMPLE_ID" "$TEST_TIFF" "$TEST_GPKG" "$OUT_ROOT/labels" <<'PY'
import json
import sys
from pathlib import Path

out_path, sample_id, tiff, gpkg, labels_dir = sys.argv[1:6]
payload = [
    {
        "sample_id": sample_id,
        "image": tiff,
        "gt_gpkg": gpkg,
        "gt_origin": "whole_image",
        "pred_txt": str(Path(labels_dir) / f"{sample_id}.txt"),
    }
]
with open(out_path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
PY
fi

echo "2/4 yolo.export_sahi_visualization (prediction overlay TIFF)..."
EXPORT_CMD=(
    uv run python -u -m yolo.export_sahi_visualization
    --output-dir "$OUT_ROOT"
    --pred-labels-dir "$OUT_ROOT/labels"
)
if [[ -n "$MANIFEST" ]]; then
    EXPORT_CMD+=(--manifest "$MANIFEST_PATH")
else
    EXPORT_CMD+=(--test-tiff "$TEST_TIFF")
fi
"${EXPORT_CMD[@]}"

echo "3/4 common.evaluate_instances..."
INSTANCE_CMD=(
    uv run python -u -m common.evaluate_instances
    --unit whole
    --model-type yolo
    --variant "$VARIANT"
    --manifest "$MANIFEST_PATH"
    --pred-labels-dir "$OUT_ROOT/labels"
    --gt-gpkg "$TEST_GPKG"
    --output-json "$INSTANCE_METRICS_JSON"
)
"${INSTANCE_CMD[@]}"

echo "4/4 yolo.evaluate_mask_ap (COCO mask AP from pred txt scores)..."
MASK_AP_CMD=(
    uv run python -u -m yolo.evaluate_mask_ap
    --variant "$VARIANT"
    --pred-labels-dir "$OUT_ROOT/labels"
    --output-json "$MASK_AP_JSON"
)
if [[ -n "$MANIFEST" ]]; then
    MASK_AP_CMD+=(--manifest "$MANIFEST_PATH")
else
    MASK_AP_CMD+=(--test-tiff "$TEST_TIFF" --test-gpkg "$TEST_GPKG")
fi
"${MASK_AP_CMD[@]}"

echo "Wrote $INSTANCE_METRICS_JSON and $MASK_AP_JSON"
