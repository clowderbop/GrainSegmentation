#!/bin/bash
#SBATCH --job-name=test_unet
#SBATCH --output=logs/%x-%j.log
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --time=04:00:00

set -euo pipefail
# shellcheck source=SLURM/utils/source_job.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../utils/source_job.sh"
# shellcheck source=SLURM/utils/variants.sh
source "$SLURM_ROOT/utils/variants.sh"
# shellcheck source=SLURM/utils/watershed.sh
source "$SLURM_ROOT/utils/watershed.sh"
# shellcheck source=SLURM/utils/tensorflow.sh
source "$SLURM_ROOT/utils/tensorflow.sh"

MODEL_DIR=""
IMAGE_DIR=""
MASK_DIR=""
GT_GPKG=""
OUTPUT_DIR=""
CONFIG_FILE=""
PPL_IMAGE=""
GT_PATH=""
PATCH_SIZE=1024
STRIDE=512
BATCH_SIZE=1
MASK_EXT=".tif"
MASK_STEM_SUFFIX="_labels"

WATERSHED_TUNE_ROOT=""
INSTANCE_METHOD="${INSTANCE_METHOD:-watershed}"

function usage {
    cat <<'EOF' >&2
Usage: run_unet_whole_test_eval.sh --model-dir DIR --image-dir DIR --mask-dir DIR --output-dir DIR [options]

Required:
  --model-dir, --image-dir, --mask-dir, --output-dir

Options:
  --gt-gpkg PATH
  --config-file PATH
  --instance-method cc|watershed   (default: watershed)
  --watershed-tune-root DIR        (watershed only: load watershed_best_*.json)
  --ppl-image, --gt-path, --patch-size, --stride, --batch-size
  --mask-ext, --mask-stem-suffix
EOF
    exit 1
}

function stage_optional_path {
    local original="$1"
    local original_root="$2"
    local local_root="$3"

    if [ -z "$original" ]; then
        printf '\n'
        return
    fi

    if [[ "$original" == "$original_root"/* ]]; then
        local relative="${original#"$original_root"/}"
        printf '%s\n' "$local_root/$relative"
        return
    fi

    printf '%s\n' "$original"
}

function resolve_config_model_path {
    local model_ref="$1"

    if [[ "$model_ref" = /* ]]; then
        local staged_path="$LOCAL_MODEL_DIR/$(basename "$model_ref")"
        if [ ! -f "$staged_path" ]; then
            cp "$model_ref" "$staged_path"
        fi
        printf '%s\n' "$staged_path"
        return
    fi

    printf '%s\n' "$LOCAL_MODEL_DIR/$model_ref"
}

function build_extract_instance_args {
    local model_path="$1"
    local explicit_ws="${2:-}"

    if [[ "$INSTANCE_METHOD" == "cc" ]]; then
        extract_args=(--instance-method cc)
        if [[ -n "${CC_MIN_AREA_PX:-}" ]]; then
            extract_args+=(--min-area-px "$CC_MIN_AREA_PX")
        fi
        return 0
    fi

    if [[ "$INSTANCE_METHOD" != "watershed" ]]; then
        echo "Unknown --instance-method: $INSTANCE_METHOD (expected cc or watershed)" >&2
        return 1
    fi

    local resolved_ws_json=""
    if ! resolved_ws_json="$(resolve_watershed_json_for_model "$MODEL_DIR" "$explicit_ws" "$model_path")"; then
        return 1
    fi

    build_watershed_extract_args "$resolved_ws_json" "$WATERSHED_JSON_HELPER"
}

function find_default_ppl_image {
    shopt -s nullglob
    local matches=("$LOCAL_IMAGE_DIR"/*_PPL.*)
    shopt -u nullglob

    if [ "${#matches[@]}" -eq 0 ]; then
        echo "Unable to infer --ppl-image; no *_PPL.* file found in $IMAGE_DIR"
        exit 1
    fi

    printf '%s\n' "${matches[0]}"
}

function infer_overlay_sample_id {
    local ppl_path="$1"
    local base_name
    local stem

    base_name="$(basename "$ppl_path")"
    stem="${base_name%.*}"

    if [[ "$stem" != *_PPL ]]; then
        echo "Unable to infer overlay sample id from PPL image: $ppl_path"
        exit 1
    fi

    printf '%s\n' "${stem%_PPL}"
}

function find_mask_for_sample {
    local sample_id="$1"
    local candidate=""

    if [ -n "$MASK_EXT" ]; then
        candidate="$LOCAL_MASK_DIR/${sample_id}${MASK_STEM_SUFFIX}${MASK_EXT}"
        require_file "$candidate" "Mask not found for overlay sample"
        printf '%s\n' "$candidate"
        return
    fi

    local ext=""
    for ext in .tif .tiff; do
        candidate="$LOCAL_MASK_DIR/${sample_id}${MASK_STEM_SUFFIX}${ext}"
        if [ -f "$candidate" ]; then
            printf '%s\n' "$candidate"
            return
        fi
    done

    echo "Unable to infer ground-truth mask for overlay sample: $sample_id"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --image-dir)
            IMAGE_DIR="$2"
            shift 2
            ;;
        --mask-dir)
            MASK_DIR="$2"
            shift 2
            ;;
        --gt-gpkg)
            GT_GPKG="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --config-file)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --ppl-image)
            PPL_IMAGE="$2"
            shift 2
            ;;
        --gt-path)
            GT_PATH="$2"
            shift 2
            ;;
        --patch-size)
            PATCH_SIZE="$2"
            shift 2
            ;;
        --stride)
            STRIDE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --mask-ext)
            MASK_EXT="$2"
            shift 2
            ;;
        --mask-stem-suffix)
            MASK_STEM_SUFFIX="$2"
            shift 2
            ;;
        --watershed-tune-root)
            WATERSHED_TUNE_ROOT="$2"
            shift 2
            ;;
        --instance-method)
            INSTANCE_METHOD="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

if [ -z "$MODEL_DIR" ] || [ -z "$IMAGE_DIR" ] || [ -z "$MASK_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    usage
fi

case "$INSTANCE_METHOD" in
    cc | watershed) ;;
    *)
        echo "Invalid --instance-method: $INSTANCE_METHOD (expected cc or watershed)" >&2
        exit 1
        ;;
esac

if [[ "$INSTANCE_METHOD" == "cc" && -n "$WATERSHED_TUNE_ROOT" ]]; then
    echo "Note: --watershed-tune-root is ignored when --instance-method cc." >&2
fi

require_dir "$MODEL_DIR" "Model directory not found"
require_dir "$IMAGE_DIR" "Image directory not found"
require_dir "$MASK_DIR" "Mask directory not found"
if [ -z "$GT_GPKG" ]; then
    GT_GPKG="$MASK_DIR/test_labels.gpkg"
fi
require_file "$GT_GPKG" "Ground-truth GeoPackage not found"
if [ -n "$CONFIG_FILE" ]; then
    require_file "$CONFIG_FILE" "Config file not found"
fi

mkdir -p "$OUTPUT_DIR"

source "$SLURM_ROOT/prepare_env.sh"
export TF_CPP_MIN_LOG_LEVEL=2

WORK_DIR="$TMPDIR/eval_models_${SLURM_JOB_ID:-$$}"
LOCAL_MODEL_DIR="$WORK_DIR/models"
LOCAL_IMAGE_DIR="$WORK_DIR/images"
LOCAL_MASK_DIR="$WORK_DIR/masks"
LOCAL_GT_GPKG="$WORK_DIR/$(basename "$GT_GPKG")"
mkdir -p "$LOCAL_MODEL_DIR" "$LOCAL_IMAGE_DIR" "$LOCAL_MASK_DIR"

echo "Copying models and dataset to TMPDIR..."
cp -r "$MODEL_DIR"/. "$LOCAL_MODEL_DIR"/
cp -r "$IMAGE_DIR"/. "$LOCAL_IMAGE_DIR"/
cp -r "$MASK_DIR"/. "$LOCAL_MASK_DIR"/
cp -f "$GT_GPKG" "$LOCAL_GT_GPKG"

cd "$REPO_ROOT/src/unet"
echo "Syncing U-Net environment..."
uv sync

install_unet_tensorflow_wheel

MODEL_LABELS=()
MODEL_PATHS=()
MODEL_NUM_INPUTS=()
MODEL_SUFFIXES=()
MODEL_WATERSHED_JSONS=()
JSON_FILES=()
PRED_PATHS=()

if [ -n "$CONFIG_FILE" ]; then
    while IFS=$'\t' read -r label model_ref num_inputs suffix_csv watershed_json_opt || [ -n "$label" ]; do
        if [ -z "${label// }" ] || [[ "$label" == \#* ]]; then
            continue
        fi

        if [ -z "${model_ref:-}" ] || [ -z "${num_inputs:-}" ] || [ -z "${suffix_csv:-}" ]; then
            echo "Invalid config row in $CONFIG_FILE: $label"
            exit 1
        fi

        local_model_path="$(resolve_config_model_path "$model_ref")"
        require_file "$local_model_path" "Configured model not found"

        MODEL_LABELS+=("$label")
        MODEL_PATHS+=("$local_model_path")
        MODEL_NUM_INPUTS+=("$num_inputs")
        MODEL_SUFFIXES+=("$suffix_csv")
        MODEL_WATERSHED_JSONS+=("${watershed_json_opt:-}")
    done < "$CONFIG_FILE"
else
    shopt -s nullglob globstar
    local_models=("$LOCAL_MODEL_DIR"/**/*.keras)
    shopt -u nullglob globstar

    if [ "${#local_models[@]}" -eq 0 ]; then
        echo "No .keras models found in $MODEL_DIR"
        exit 1
    fi

    for model_path in "${local_models[@]}"; do
        if ! inferred="$(infer_model_config "$model_path")"; then
            echo "Unable to infer config for model; use --config-file instead: $model_path"
            exit 1
        fi

        IFS=$'\t' read -r label num_inputs suffix_csv <<< "$inferred"
        MODEL_LABELS+=("$label")
        MODEL_PATHS+=("$model_path")
        MODEL_NUM_INPUTS+=("$num_inputs")
        MODEL_SUFFIXES+=("$suffix_csv")
        MODEL_WATERSHED_JSONS+=("")
    done
fi

if [ "${#MODEL_PATHS[@]}" -eq 0 ]; then
    echo "No models configured for evaluation."
    exit 1
fi

WATERSHED_JSON_HELPER="$REPO_ROOT/src/unet/watershed_json_to_eval_args.py"

echo "Running staged evaluations (instance_method=$INSTANCE_METHOD: predict -> extract -> semantic -> instances)..."
for i in "${!MODEL_PATHS[@]}"; do
    model_path="${MODEL_PATHS[$i]}"
    model_file="$(basename "$model_path")"
    model_stem="${model_file%.keras}"
    pred_root="$OUTPUT_DIR/run_${model_stem}"
    instance_json="$pred_root/instance_metrics.json"
    suffix_csv="${MODEL_SUFFIXES[$i]}"
    IFS=',' read -r -a suffix_array <<< "$suffix_csv"

    mkdir -p "$pred_root"

    predict_cmd=(
        uv run --no-sync python -u -m unet.predict
        --model-path "$model_path"
        --image-dir "$LOCAL_IMAGE_DIR"
        --mask-dir "$LOCAL_MASK_DIR"
        --output-dir "$pred_root"
        --num-inputs "${MODEL_NUM_INPUTS[$i]}"
        --image-suffixes
        "${suffix_array[@]}"
        --patch-size "$PATCH_SIZE"
        --stride "$STRIDE"
        --batch-size "$BATCH_SIZE"
        --mask-stem-suffix "$MASK_STEM_SUFFIX"
        --unit whole
    )
    if [ -n "$MASK_EXT" ]; then
        predict_cmd+=(--mask-ext "$MASK_EXT")
    fi

    explicit_ws="${MODEL_WATERSHED_JSONS[$i]:-}"
    if ! build_extract_instance_args "$model_path" "$explicit_ws"; then
        exit 1
    fi

    echo "Model ${MODEL_LABELS[$i]}: predict"
    "${predict_cmd[@]}"

    echo "Model ${MODEL_LABELS[$i]}: extract_instances"
    extract_cmd=(
        uv run --no-sync python -u -m unet.extract_instances
        --semantic-dir "$pred_root/semantic"
        --output-dir "$pred_root"
        "${extract_args[@]}"
    )
    "${extract_cmd[@]}"

    echo "Model ${MODEL_LABELS[$i]}: evaluate_semantic"
    semantic_cmd=(
        uv run --no-sync python -u -m unet.evaluate_semantic
        --semantic-dir "$pred_root/semantic"
        --mask-dir "$LOCAL_MASK_DIR"
        --mask-stem-suffix "$MASK_STEM_SUFFIX"
        --output-json "$pred_root/semantic_metrics.json"
        --unit whole
    )
    if [ -n "$MASK_EXT" ]; then
        semantic_cmd+=(--mask-ext "$MASK_EXT")
    fi
    "${semantic_cmd[@]}"

    echo "Model ${MODEL_LABELS[$i]}: evaluate_instances"
    instance_cmd=(
        uv run --no-sync python -u -m common.evaluate_instances
        --model-type unet
        --unit whole
        --image-dir "$LOCAL_IMAGE_DIR"
        --pred-instances-dir "$pred_root/instances"
        --gt-gpkg "$LOCAL_GT_GPKG"
        --gt-origin whole_image
        --image-stem-suffix "${suffix_array[0]}"
        --output-json "$instance_json"
    )
    "${instance_cmd[@]}"

    JSON_FILES+=("$instance_json")
    PRED_PATHS+=("$pred_root/semantic")
done

LOCAL_PPL_IMAGE="$(stage_optional_path "$PPL_IMAGE" "$IMAGE_DIR" "$LOCAL_IMAGE_DIR")"
if [ -z "$LOCAL_PPL_IMAGE" ]; then
    LOCAL_PPL_IMAGE="$(find_default_ppl_image)"
fi
require_file "$LOCAL_PPL_IMAGE" "Overlay PPL image not found"

OVERLAY_SAMPLE_ID="$(infer_overlay_sample_id "$LOCAL_PPL_IMAGE")"
LOCAL_GT_PATH="$(stage_optional_path "$GT_PATH" "$MASK_DIR" "$LOCAL_MASK_DIR")"
if [ -z "$LOCAL_GT_PATH" ]; then
    LOCAL_GT_PATH="$(find_mask_for_sample "$OVERLAY_SAMPLE_ID")"
fi
require_file "$LOCAL_GT_PATH" "Overlay ground-truth mask not found"

OVERLAY_PRED_PATHS=()
for i in "${!MODEL_PATHS[@]}"; do
    model_path="${MODEL_PATHS[$i]}"
    model_stem="$(basename "${model_path%.keras}")"
    pred_path="$OUTPUT_DIR/run_${model_stem}/semantic/${OVERLAY_SAMPLE_ID}_pred.tif"
    require_file "$pred_path" "Overlay prediction not found"
    OVERLAY_PRED_PATHS+=("$pred_path")
done

echo "Generating comparison plots..."
plot_cmd=(
    uv run --no-sync python -u -m common.plot_results
    --json-files
    "${JSON_FILES[@]}"
    --labels
    "${MODEL_LABELS[@]}"
    --output-plot "$OUTPUT_DIR/quantitative_plot.tif"
)
"${plot_cmd[@]}"

overlay_cmd=(
    uv run --no-sync python -u -m unet.plot_overlay
    --image-path "$LOCAL_PPL_IMAGE"
    --gt-path "$LOCAL_GT_PATH"
    --pred-paths
    "${OVERLAY_PRED_PATHS[@]}"
    --labels
    "${MODEL_LABELS[@]}"
    --output-overlay "$OUTPUT_DIR/overlay.tif"
)
"${overlay_cmd[@]}"

echo "Saved evaluation outputs to $OUTPUT_DIR"
