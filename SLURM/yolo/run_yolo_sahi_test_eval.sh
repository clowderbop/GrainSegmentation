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
SAHI_OUT="$SCRATCH/GrainSeg/eval/yolo_${VARIANT}"
OUTPUT_JSON="$SAHI_OUT/metrics-${VARIANT}-${SLURM_JOB_ID}.json"

if [[ -n "$OUTPUT_JSON" ]]; then
    mkdir -p "$(dirname "$OUTPUT_JSON")"
fi
if [[ -n "$SAHI_OUT" ]]; then
    mkdir -p "$SAHI_OUT"
fi

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

EVAL_CMD=(
    uv run python -u evaluate.py
    --mode sahi
    --weights "$WEIGHTS"
    --device "$DEVICE"
    --slice-height "$SLICE_H"
    --slice-width "$SLICE_W"
    --overlap-height-ratio "$OV_H"
    --overlap-width-ratio "$OV_W"
)

if [[ -n "$MANIFEST" ]]; then
    EVAL_CMD+=(--manifest "$MANIFEST")
else
    EVAL_CMD+=(--test-tiff "$TEST_TIFF" --test-gpkg "$TEST_GPKG")
fi

if [[ -n "$OUTPUT_JSON" ]]; then
    EVAL_CMD+=(--output-json "$OUTPUT_JSON")
fi

if [[ -n "$SAHI_OUT" ]]; then
    EVAL_CMD+=(--sahi-out-dir "$SAHI_OUT")
fi

"${EVAL_CMD[@]}"
