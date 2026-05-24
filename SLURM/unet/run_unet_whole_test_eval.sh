#!/bin/bash
#SBATCH --job-name=test_unet
#SBATCH --output=logs/%x-%j.log
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --time=04:00:00

set -euo pipefail
# shellcheck source=SLURM/utils/enter_job.sh
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/SLURM/utils/enter_job.sh"
# shellcheck source=SLURM/utils/variants.sh
source "$SLURM_ROOT/utils/variants.sh"
# shellcheck source=SLURM/utils/manifest_shell.sh
source "$SLURM_ROOT/utils/manifest_shell.sh"
# shellcheck source=SLURM/utils/watershed.sh
source "$SLURM_ROOT/utils/watershed.sh"
# shellcheck source=SLURM/utils/tensorflow.sh
source "$SLURM_ROOT/utils/tensorflow.sh"

GRAINSEG_ROOT="$(grainseg_root)"
MANIFEST_SPLIT=""

MODEL_DIR=""
GT_GPKG=""
OUTPUT_DIR=""
CONFIG_FILE=""
OVERLAY_VARIANT="${OVERLAY_VARIANT:-PPL}"
GT_PATH=""
DEFAULT_CONFIG_FILE="$SLURM_ROOT/unet/whole_eval_models.tsv"
PATCH_SIZE=1024
STRIDE=512
BATCH_SIZE=1
MASK_EXT=".tif"
MASK_STEM_SUFFIX="_labels"

WATERSHED_TUNE_ROOT=""
INSTANCE_METHOD="${INSTANCE_METHOD:-watershed}"

function usage {
    cat <<'EOF' >&2
Usage: run_unet_whole_test_eval.sh --model-dir DIR --manifest-split train|test --output-dir DIR [options]

Required:
  --model-dir, --manifest-split, --output-dir

Options:
  --gt-gpkg PATH                 (default: dataset/{split}/{split}_labels.gpkg)
  --config-file PATH             (default: SLURM/unet/whole_eval_models.tsv)
  --instance-method cc|watershed (default: watershed)
  --watershed-tune-root DIR      (watershed only)
  --overlay-variant VARIANT      (default: PPL)
  --gt-path PATH                 (overlay raster mask; default from staged overlay manifest)
  --patch-size, --stride, --batch-size
  --mask-ext, --mask-stem-suffix
EOF
    exit 1
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

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --manifest-split)
            MANIFEST_SPLIT="$2"
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
        --overlay-variant)
            OVERLAY_VARIANT="$2"
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

if [ -z "$MODEL_DIR" ] || [ -z "$MANIFEST_SPLIT" ] || [ -z "$OUTPUT_DIR" ]; then
    usage
fi

case "$MANIFEST_SPLIT" in
    train | test) ;;
    *)
        echo "Invalid --manifest-split: $MANIFEST_SPLIT (expected train or test)" >&2
        exit 1
        ;;
esac

if [ -z "$CONFIG_FILE" ]; then
    CONFIG_FILE="$DEFAULT_CONFIG_FILE"
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
if [ -z "$GT_GPKG" ]; then
    GT_GPKG="$(default_whole_labels_gpkg "$MANIFEST_SPLIT" "$GRAINSEG_ROOT")"
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
LOCAL_GT_GPKG="$WORK_DIR/$(basename "$GT_GPKG")"
mkdir -p "$LOCAL_MODEL_DIR"

echo "Copying models and ground-truth GeoPackage to TMPDIR..."
cp -r "$MODEL_DIR"/. "$LOCAL_MODEL_DIR"/
cp -f "$GT_GPKG" "$LOCAL_GT_GPKG"

cd "$REPO_ROOT/src/unet"
echo "Syncing U-Net environment..."
uv sync

install_unet_tensorflow_wheel

MODEL_LABELS=()
MODEL_PATHS=()
MODEL_VARIANTS=()
MODEL_WATERSHED_JSONS=()
JSON_FILES=()
PRED_PATHS=()

require_file "$CONFIG_FILE" "Eval config file not found"
while IFS=$'\t' read -r label model_ref variant watershed_json_opt || [ -n "$label" ]; do
    if [ -z "${label// }" ] || [[ "$label" == \#* ]]; then
        continue
    fi

    if [ -z "${model_ref:-}" ] || [ -z "${variant:-}" ]; then
        echo "Invalid config row in $CONFIG_FILE (need label, model, variant): $label"
        exit 1
    fi

    local_model_path="$(resolve_config_model_path "$model_ref")"
    require_file "$local_model_path" "Configured model not found"

    MODEL_LABELS+=("$label")
    MODEL_PATHS+=("$local_model_path")
    MODEL_VARIANTS+=("$variant")
    MODEL_WATERSHED_JSONS+=("${watershed_json_opt:-}")
done < "$CONFIG_FILE"

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
    mkdir -p "$pred_root"

    variant="${MODEL_VARIANTS[$i]}"

    canonical_manifest="$GRAINSEG_ROOT/dataset/$MANIFEST_SPLIT/manifests/${variant}.whole.json"
    require_file "$canonical_manifest" \
        "Whole-section manifest missing for $variant ($MANIFEST_SPLIT); run write_whole_manifests.py"

    model_image_dir="$WORK_DIR/images/$variant"
    rm -rf "$model_image_dir"
    mkdir -p "$model_image_dir"
    echo "Staging manifest inputs for variant=$variant ($MANIFEST_SPLIT)..."
    uv run --directory "$REPO_ROOT" python -m common.stage_manifest run \
        "$canonical_manifest" "$model_image_dir"

    predict_cmd=(
        uv run --no-sync python -u -m unet.predict
        --model-path "$model_path"
        --manifest "$model_image_dir/manifest.json"
        --output-dir "$pred_root"
        --variant "$variant"
        --patch-size "$PATCH_SIZE"
        --stride "$STRIDE"
        --batch-size "$BATCH_SIZE"
        --mask-stem-suffix "$MASK_STEM_SUFFIX"
        --unit whole
    )
    if [ -n "$MASK_EXT" ]; then
        predict_cmd+=(--mask-ext "$MASK_EXT")
    fi

    export VARIANT="$variant"
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
        --manifest "$model_image_dir/manifest.json"
        "${extract_args[@]}"
    )
    "${extract_cmd[@]}"

    echo "Model ${MODEL_LABELS[$i]}: evaluate_semantic"
    semantic_cmd=(
        uv run --no-sync python -u -m unet.evaluate_semantic
        --semantic-dir "$pred_root/semantic"
        --mask-dir "$model_image_dir"
        --mask-stem-suffix "$MASK_STEM_SUFFIX"
        --output-json "$pred_root/semantic_metrics.json"
        --unit whole
    )
    if [ -n "$MASK_EXT" ]; then
        semantic_cmd+=(--mask-ext "$MASK_EXT")
    fi
    "${semantic_cmd[@]}"

    eval_manifest="$pred_root/eval_manifest.json"
    echo "Model ${MODEL_LABELS[$i]}: write eval manifest"
    uv run --directory "$REPO_ROOT" python -m common.stage_manifest write-eval \
        --source "$model_image_dir/manifest.json" \
        --pred-instances-dir "$pred_root/instances" \
        --output "$eval_manifest" \
        --gt-gpkg "$LOCAL_GT_GPKG"

    echo "Model ${MODEL_LABELS[$i]}: evaluate_instances"
    instance_cmd=(
        uv run --no-sync python -u -m common.evaluate_instances
        --model-type unet
        --unit whole
        --variant "$variant"
        --manifest "$eval_manifest"
        --output-json "$instance_json"
    )
    "${instance_cmd[@]}"

    JSON_FILES+=("$instance_json")
    PRED_PATHS+=("$pred_root/semantic")
done

OVERLAY_CANONICAL="$GRAINSEG_ROOT/dataset/$MANIFEST_SPLIT/manifests/${OVERLAY_VARIANT}.whole.json"
require_file "$OVERLAY_CANONICAL" \
    "Overlay whole manifest missing for $OVERLAY_VARIANT; run write_whole_manifests.py"
OVERLAY_STAGE="$WORK_DIR/overlay"
rm -rf "$OVERLAY_STAGE"
echo "Staging overlay inputs from $OVERLAY_CANONICAL ..."
uv run --directory "$REPO_ROOT" python -m common.stage_manifest run \
    "$OVERLAY_CANONICAL" "$OVERLAY_STAGE"
export_overlay_env_from_whole_manifest "$OVERLAY_STAGE/manifest.json"
LOCAL_PPL_IMAGE="$OVERLAY_STAGE/$OVERLAY_IMAGE_REL"
require_file "$LOCAL_PPL_IMAGE" "Overlay PPL image not found"

if [ -n "$GT_PATH" ]; then
    LOCAL_GT_PATH="$GT_PATH"
else
    if [ -z "${OVERLAY_MASK_REL:-}" ]; then
        echo "Overlay manifest missing mask path" >&2
        exit 1
    fi
    LOCAL_GT_PATH="$OVERLAY_STAGE/$OVERLAY_MASK_REL"
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
