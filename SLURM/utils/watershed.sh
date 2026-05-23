# shellcheck shell=bash
_slurm_utils="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${REPO_ROOT:-}" ] || [ -z "${SLURM_ROOT:-}" ]; then
    # shellcheck source=SLURM/utils/paths.sh
    source "$_slurm_utils/paths.sh"
fi
# shellcheck source=SLURM/utils/assertions.sh
source "$_slurm_utils/assertions.sh"
# shellcheck source=SLURM/utils/variants.sh
source "$_slurm_utils/variants.sh"
unset _slurm_utils

watershed_tune_subdir() {
    if [ -z "${VARIANT:-}" ]; then
        echo "VARIANT must be set to resolve watershed tune subdirectory" >&2
        return 1
    fi
    watershed_tune_subdir_for_variant "$VARIANT"
}

pick_latest_watershed_best_json() {
    local dir="$1"
    shopt -s nullglob
    local matches=("$dir"/watershed_best_*.json)
    shopt -u nullglob

    if [ "${#matches[@]}" -eq 0 ]; then
        echo "No watershed_best_*.json files in: $dir" >&2
        return 1
    fi

    local newest=""
    local newest_mtime=0
    for f in "${matches[@]}"; do
        local m
        m="$(stat -c '%Y' "$f" 2>/dev/null || stat -f '%m' "$f")"
        if [ "$m" -gt "$newest_mtime" ]; then
            newest_mtime="$m"
            newest="$f"
        fi
    done
    printf '%s\n' "$newest"
}

# Args: model_dir explicit_json model_path
# Prints resolved JSON path or empty. Uses WATERSHED_TUNE_ROOT and VARIANT.
resolve_watershed_json_for_model() {
    local model_dir="$1"
    local explicit_json="${2:-}"
    local model_path="$3"

    if [ -n "$explicit_json" ]; then
        if [[ "$explicit_json" = /* ]]; then
            printf '%s\n' "$explicit_json"
            return 0
        fi
        printf '%s\n' "$model_dir/$explicit_json"
        return 0
    fi

    if [ -z "${WATERSHED_TUNE_ROOT:-}" ]; then
        printf '\n'
        return 0
    fi

    local subdir variant_dir
    if ! subdir="$(watershed_tune_subdir)"; then
        return 1
    fi

    variant_dir="$WATERSHED_TUNE_ROOT/$subdir"
    if [ ! -d "$variant_dir" ]; then
        echo "Watershed tune variant directory not found: $variant_dir" >&2
        return 1
    fi

    pick_latest_watershed_best_json "$variant_dir"
}

# Best-effort resolve for patch eval (warns instead of failing when tune dir/json missing).
resolve_watershed_json_lenient() {
    local model_dir="$1"
    local explicit_json="${2:-}"
    local model_path="$3"
    local tune_root="${4:-${WATERSHED_TUNE_ROOT:-}}"

    if [ -n "$explicit_json" ]; then
        if [[ "$explicit_json" = /* ]]; then
            printf '%s\n' "$explicit_json"
        else
            printf '%s\n' "$model_dir/$explicit_json"
        fi
        return 0
    fi

    if [ -z "$tune_root" ]; then
        printf '\n'
        return 0
    fi

    if [ -z "${VARIANT:-}" ]; then
        echo "Note: VARIANT unset; using default watershed args." >&2
        printf '\n'
        return 0
    fi

    local subdir variant_tune_dir picked
    if ! subdir="$(watershed_tune_subdir_for_variant "$VARIANT")"; then
        echo "Note: cannot resolve watershed tune subdir for VARIANT=$VARIANT; using defaults." >&2
        printf '\n'
        return 0
    fi

    variant_tune_dir="$tune_root/$subdir"
    if [[ ! -d "$variant_tune_dir" ]]; then
        echo "Note: watershed tune directory not found: $variant_tune_dir; using default watershed args." >&2
        printf '\n'
        return 0
    fi
    if picked="$(pick_latest_watershed_best_json "$variant_tune_dir" 2>/dev/null)"; then
        printf '%s\n' "$picked"
        return 0
    fi
    echo "Note: no watershed_best_*.json under $variant_tune_dir; using default watershed args." >&2
    printf '\n'
}

# Sets global extract_args for the caller's extract_instances command.
build_watershed_extract_args() {
    local json_path="$1"
    local helper="${2:-$REPO_ROOT/src/unet/watershed_json_to_eval_args.py}"
    # shellcheck disable=SC2034
    extract_args=(--instance-method watershed)
    if [[ -n "$json_path" ]]; then
        require_file "$json_path" "Watershed tuning JSON not found"
        if [[ ! -f "$helper" ]]; then
            echo "Missing helper script: $helper" >&2
            return 1
        fi
        # shellcheck disable=SC2034
        mapfile -t extract_args < <(python3 "$helper" "$json_path")
    fi
}
