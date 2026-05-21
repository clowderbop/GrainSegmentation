# shellcheck shell=bash
# Stage a YOLO patch dataset under TMPDIR and rewrite its data YAML path.
# Args: variant split  (split = train | test)
# Sets: DATA_YAML, DATASET_ROOT (directory containing the yaml)

_slurm_utils="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=SLURM/utils/variants.sh
source "$_slurm_utils/variants.sh"
unset _slurm_utils

stage_yolo_patch_dataset() {
    local variant="$1"
    local split="$2"
    local grainseg
    grainseg="$(grainseg_root)"

    if ! yolo_dataset_names_for_variant "$variant"; then
        return 1
    fi

    local src_root dst_root rewrite_test_split=false
    case "$split" in
        train)
            src_root="$grainseg/dataset/train/patches/$DATASET_SUBDIR"
            ;;
        test)
            src_root="$grainseg/dataset/test/patches/$DATASET_SUBDIR"
            rewrite_test_split=true
            ;;
        *)
            echo "stage_yolo_patch_dataset: split must be train or test (got: $split)" >&2
            return 1
            ;;
    esac

    echo "Staging YOLO dataset ($split) to TMPDIR..."
    local tmp_yolo_root="$TMPDIR/yolo"
    dst_root="$tmp_yolo_root/$DATASET_SUBDIR"
    mkdir -p "$tmp_yolo_root"
    cp -r "$src_root" "$tmp_yolo_root/"
    DATA_YAML="$dst_root/$YAML_NAME"
    DATASET_ROOT="$dst_root"

    local rewrite_args=("$DATA_YAML" "$DATASET_ROOT")
    if [ "$rewrite_test_split" = true ]; then
        rewrite_args+=(test)
    fi
    (
        cd "$REPO_ROOT/src/yolo" || exit
        uv run python "$SLURM_UTILS/rewrite_yolo_dataset_yaml.py" "${rewrite_args[@]}"
    )
}
