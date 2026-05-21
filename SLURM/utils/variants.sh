# shellcheck shell=bash
_slurm_utils="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${REPO_ROOT:-}" ] || [ -z "${SLURM_ROOT:-}" ]; then
    # shellcheck source=SLURM/utils/paths.sh
    source "$_slurm_utils/paths.sh"
fi
unset _slurm_utils

# shellcheck disable=SC2034
MICROSCOPY_VARIANTS=(PPL PPLPPXblend "PPL+PPXblend" "PPL+AllPPX")

job_slug() {
    printf '%s' "$1" | tr '+#' '__'
}

# Match longest variant names first (same order as watershed / model-stem inference).
infer_microscopy_variant_from_model_stem() {
    local model_stem="$1"

    if [[ "$model_stem" == *"PPL+AllPPX"* ]]; then
        printf '%s\n' "PPL+AllPPX"
        return 0
    fi
    if [[ "$model_stem" == *"PPL+PPXblend"* ]]; then
        printf '%s\n' "PPL+PPXblend"
        return 0
    fi
    if [[ "$model_stem" == *"PPLPPXblend"* ]]; then
        printf '%s\n' "PPLPPXblend"
        return 0
    fi
    if [[ "$model_stem" == *"PPL"* ]]; then
        printf '%s\n' "PPL"
        return 0
    fi

    return 1
}

watershed_tune_subdir_for_variant() {
    local variant="$1"
    case "$variant" in
        PPL+AllPPX) printf '%s\n' "PPL_AllPPX" ;;
        PPL+PPXblend) printf '%s\n' "PPL_PlusPPXblend" ;;
        PPLPPXblend) printf '%s\n' "PPLPPXblend" ;;
        PPL) printf '%s\n' "PPL" ;;
        *)
            echo "Unknown microscopy variant for watershed tune dir: $variant" >&2
            return 1
            ;;
    esac
}

# Args: exact variant name. Prints "num_inputs<TAB>suffix_csv" (comma-separated _suffixes).
unet_variant_metadata_tsv() {
    local variant="$1"
    case "$variant" in
        PPL) printf '%s\t%s\n' "1" "_PPL" ;;
        PPLPPXblend) printf '%s\t%s\n' "1" "_PPLPPXblend" ;;
        PPL+PPXblend) printf '%s\t%s\n' "2" "_PPL,_PPXblend" ;;
        PPL+AllPPX)
            printf '%s\t%s\n' "7" "_PPL,_PPX1,_PPX2,_PPX3,_PPX4,_PPX5,_PPX6"
            ;;
        *)
            echo "Unknown microscopy variant: $variant" >&2
            echo "Expected one of: PPL, PPLPPXblend, PPL+PPXblend, PPL+AllPPX" >&2
            return 1
            ;;
    esac
}

# Sets NUM_INPUTS, IMAGE_SUFFIXES (array), DEFAULT_MODEL_BASENAME for patch eval.
unet_patch_config_for_variant() {
    local variant="$1"
    local metadata num_inputs suffix_csv

    if ! metadata="$(unet_variant_metadata_tsv "$variant")"; then
        exit 1
    fi
    IFS=$'\t' read -r num_inputs suffix_csv <<< "$metadata"
    NUM_INPUTS="$num_inputs"
    IFS=',' read -r -a IMAGE_SUFFIXES <<< "$suffix_csv"
    DEFAULT_MODEL_BASENAME="unet_finetuned_${variant}.keras"
}

strip_prefix() {
    local value="$1"
    local prefix="$2"
    if [[ "$value" == "$prefix"* ]]; then
        printf '%s\n' "${value#"$prefix"}"
    else
        printf '%s\n' "$value"
    fi
}

# Args: model path. Prints "label<TAB>num_inputs<TAB>suffix_csv" for whole-image eval.
infer_model_config() {
    local model_path="$1"
    local model_file model_stem variant metadata num_inputs suffix_csv label

    model_file="$(basename "$model_path")"
    model_stem="${model_file%.keras}"

    if ! variant="$(infer_microscopy_variant_from_model_stem "$model_stem")"; then
        return 1
    fi
    if ! metadata="$(unet_variant_metadata_tsv "$variant")"; then
        return 1
    fi

    IFS=$'\t' read -r num_inputs suffix_csv <<< "$metadata"
    label="$(strip_prefix "$model_stem" "unet_finetuned_")"
    printf '%s\t%s\t%s\n' "$label" "$num_inputs" "$suffix_csv"
}

yolo_dataset_names_for_variant() {
    local variant="$1"
    case "$variant" in
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
            echo "Unknown YOLO variant: $variant" >&2
            return 1
            ;;
    esac
}

yolo_test_tiff_for_variant() {
    local variant="$1"
    local grainseg
    grainseg="$(grainseg_root)"
    case "$variant" in
        PPL) printf '%s\n' "$grainseg/dataset/test/test_PPL.tif" ;;
        PPLPPXblend) printf '%s\n' "$grainseg/dataset/test/test_PPLPPXblend.tif" ;;
        PPL+PPXblend) printf '%s\n' "$grainseg/dataset/test/test_PPL+PPXblend.tif" ;;
        PPL+AllPPX) printf '%s\n' "$grainseg/dataset/test/test_PPL+AllPPX.tif" ;;
        *)
            echo "Unknown YOLO variant: $variant" >&2
            return 1
            ;;
    esac
}
