#!/bin/bash
#SBATCH --job-name=test_yolo_patches
#SBATCH --output=logs/test_yolo_patches-%j.log
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --time=04:00:00

set -euo pipefail

# Edit these paths / hyperparameters as needed (Ultralytics val on the YOLO dataset test split).
# Override per job: sbatch --export=ALL,VARIANT=PPL+AllPPX SLURM/test_yolo_patches.sh
VARIANT="${VARIANT:-PPL}"
DEVICE="0"
IMGSZ=1024
BATCH=16
RUN_NAME="test"
PROJECT_DIR="$SCRATCH/GrainSeg/runs/yolo26-seg-val/$VARIANT"
# Leave empty to stage from $SCRATCH into TMPDIR (same layout as train_yolo.sh).
DATA_YAML=""

source SLURM/prepare_env.sh

case "$VARIANT" in
    PPL)
        DATASET_SUBDIR="PPL"
        YAML_NAME="PPL.yaml"
        ;;
    PPLPPXblend)
        DATASET_SUBDIR="PPLPPXblend"
        YAML_NAME="PPLPPXblend.yaml"
        ;;
    PPL+PPXblend)
        DATASET_SUBDIR="PPL+PPXblend"
        YAML_NAME="PPL_PPXblend.yaml"
        ;;
    PPL+AllPPX)
        DATASET_SUBDIR="PPL+AllPPX"
        YAML_NAME="PPL+AllPPX.yaml"
        ;;
    *)
        echo "Unknown YOLO variant: $VARIANT" >&2
        exit 1
        ;;
esac

WEIGHTS="$SCRATCH/GrainSeg/runs/yolo26-seg/$VARIANT/weights/best.pt"

if [[ -z "$DATA_YAML" ]]; then
    echo "Staging YOLO dataset to TMPDIR for validation..."
    TMP_YOLO_ROOT="$TMPDIR/yolo"
    TMP_DATASET_DIR="$TMP_YOLO_ROOT/$DATASET_SUBDIR"
    mkdir -p "$TMP_YOLO_ROOT"
    cp -r "$SCRATCH/GrainSeg/dataset/test/yolo/$DATASET_SUBDIR" "$TMP_YOLO_ROOT/"
    DATA_YAML="$TMP_DATASET_DIR/$YAML_NAME"

    # Run from src/yolo so uv binds to that project's environment (no pyproject at repo root).
    (
        cd src/yolo
        uv run python - "$DATA_YAML" "$TMP_DATASET_DIR" <<'PY'
from pathlib import Path
import sys

yaml_path = Path(sys.argv[1])
dataset_root = Path(sys.argv[2])
text = yaml_path.read_text(encoding="utf-8")
lines = text.splitlines()
for index, line in enumerate(lines):
    if line.startswith("path:"):
        lines[index] = f"path: {dataset_root}"
        break
else:
    raise SystemExit(f"Dataset YAML missing path entry: {yaml_path}")
for index, line in enumerate(lines):
    if line.strip() == "test:":
        lines[index] = "test: images/val"
        break
trailing_newline = "\n" if text.endswith("\n") else ""
yaml_path.write_text("\n".join(lines) + trailing_newline, encoding="utf-8")
PY
    )
fi

echo "Syncing YOLO environment..."
cd src/yolo
uv sync

export YOLO_DISABLE_TQDM=True

uv run python -u evaluate.py \
    --mode val \
    --weights "$WEIGHTS" \
    --data "$DATA_YAML" \
    --device "$DEVICE" \
    --imgsz "$IMGSZ" \
    --batch "$BATCH" \
    --name "$RUN_NAME" \
    --project "$PROJECT_DIR"
