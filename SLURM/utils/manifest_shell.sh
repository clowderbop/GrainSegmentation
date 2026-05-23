# shellcheck shell=bash
# Helpers to export manifest-derived paths for SLURM bash drivers.

_manifest_shell_utils="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${REPO_ROOT:-}" ]; then
    # shellcheck source=SLURM/utils/paths.sh
    source "$_manifest_shell_utils/paths.sh"
fi
unset _manifest_shell_utils

default_whole_labels_gpkg() {
    local split="$1"
    local grainseg_root="$2"
    if [[ "$split" == "test" ]]; then
        printf '%s\n' "$grainseg_root/dataset/test/test_labels.gpkg"
    else
        printf '%s\n' "$grainseg_root/dataset/train/train_labels.gpkg"
    fi
}

export_overlay_env_from_whole_manifest() {
    local manifest_path="$1"
    if [ ! -f "$manifest_path" ]; then
        echo "Whole manifest not found: $manifest_path" >&2
        return 1
    fi
    # shellcheck disable=SC2034
    eval "$(
        uv run --directory "$REPO_ROOT" python -c "
import shlex
import sys
from pathlib import Path
from common.manifest_io import load_dataset_manifest, whole_manifest_overlay_anchor

doc = load_dataset_manifest(Path(sys.argv[1]))
sample_id, image_rel, mask_rel = whole_manifest_overlay_anchor(doc)
print(f'export OVERLAY_SAMPLE_ID={shlex.quote(sample_id)}')
print(f'export OVERLAY_IMAGE_REL={shlex.quote(image_rel)}')
if mask_rel:
    print(f'export OVERLAY_MASK_REL={shlex.quote(mask_rel)}')
" "$manifest_path"
    )"
}

write_yolo_whole_manifest_json() {
    local variant="$1"
    local split="$2"
    local grainseg_root="$3"
    local output_path="$4"
    uv run --directory "$REPO_ROOT" python -c "
import sys
from pathlib import Path
from common.manifest_io import build_yolo_whole_manifest, write_dataset_manifest

variant, split, grainseg, out = sys.argv[1:5]
write_dataset_manifest(
    Path(out),
    build_yolo_whole_manifest(
        variant=variant,
        split=split,
        grainseg_root=Path(grainseg),
    ),
)
" "$variant" "$split" "$grainseg_root" "$output_path"
}
