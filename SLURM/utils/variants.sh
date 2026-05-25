# shellcheck shell=bash
# Microscopy variant metadata from config/variants.yaml via common.variants CLI.
_slurm_utils="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${REPO_ROOT:-}" ] || [ -z "${SLURM_ROOT:-}" ]; then
    # shellcheck source=SLURM/utils/paths.sh
    source "$_slurm_utils/paths.sh"
fi
unset _slurm_utils

if [ -z "${REPO_ROOT:-}" ]; then
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi

# After prepare_env + install_unet_tensorflow_wheel, workspace-root `uv run` pulls YOLO/torch
# and upgrades NumPy/protobuf past TensorFlow's pin. Use the U-Net project venv with --no-sync
# when UV_PROJECT_ENVIRONMENT is set (see SLURM/utils/manifest_shell.sh).
_variants_cli() {
    if [ -n "${UV_PROJECT_ENVIRONMENT:-}" ]; then
        uv run --directory "$REPO_ROOT/src/unet" --no-sync python -m common.variants "$@"
    else
        uv run --directory "$REPO_ROOT" python -m common.variants "$@"
    fi
}

# shellcheck disable=SC2034
read -r -a MICROSCOPY_VARIANTS <<< "$(_variants_cli all-names)"

job_slug() {
    printf '%s' "$1" | tr '+#' '__'
}

watershed_tune_subdir_for_variant() {
    _variants_cli watershed-subdir --variant "$1"
}

unet_variant_metadata_tsv() {
    _variants_cli unet-metadata-tsv --variant "$1"
}

unet_patch_config_for_variant() {
    local variant="$1"
    # shellcheck disable=SC1090
    eval "$(_variants_cli --grainseg-root "$(grainseg_root)" env --variant "$variant")"
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

yolo_dataset_names_for_variant() {
    local variant="$1"
    # shellcheck disable=SC1090
    eval "$(_variants_cli --grainseg-root "$(grainseg_root)" env --variant "$variant")"
}

yolo_test_tiff_for_variant() {
    _variants_cli yolo-test-tiff --variant "$1" --grainseg-root "$(grainseg_root)"
}
