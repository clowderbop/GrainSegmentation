# shellcheck shell=bash
# Shared YOLO venv for profile-tune candidate/finalize (sync once on scratch, copy per job).

yolo_venv_lockfile() {
    printf '%s\n' "$REPO_ROOT/src/yolo/uv.lock"
}

yolo_venv_fingerprint() {
    sha256sum "$(yolo_venv_lockfile)" | awk '{print $1}'
}

yolo_venv_shared_root() {
    local fp
    fp="$(yolo_venv_fingerprint)"
    printf '%s\n' "${SCRATCH:-/scratch/${USER:?}}/.venvs/yolo-profile-tune/${fp}"
}

yolo_venv_ready_marker() {
    printf '%s\n' "$(yolo_venv_shared_root)/.ready"
}

yolo_venv_shared_is_ready() {
    local marker root fp
    root="${1:-${SHARED_VENV_ROOT:-$(yolo_venv_shared_root)}}"
    marker="${root}/.ready"
    fp="$(yolo_venv_fingerprint)"
    [ -f "$marker" ] && [ "$(cat "$marker")" = "$fp" ] && [ -x "$root/bin/python" ]
}

yolo_venv_prepare_shared() {
    local root fp marker
    root="$(yolo_venv_shared_root)"
    fp="$(yolo_venv_fingerprint)"
    marker="$(yolo_venv_ready_marker)"
    if yolo_venv_shared_is_ready; then
        echo "[$(date -Is)] shared YOLO venv already ready: $root"
        return 0
    fi
    mkdir -p "$root"
    export UV_PROJECT_ENVIRONMENT="$root"
    export VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT"
    cd "$REPO_ROOT/src/yolo"
    echo "[$(date -Is)] uv sync → shared venv $root …"
    uv sync
    echo "[$(date -Is)] uv sync done ($(du -sh "$root" | cut -f1))"
    printf '%s' "$fp" >"$marker"
}

yolo_venv_stage_local() {
    local shared
    : "${UV_PROJECT_ENVIRONMENT:?UV_PROJECT_ENVIRONMENT must be set (prepare_env.sh)}"
    shared="${SHARED_VENV_ROOT:-$(yolo_venv_shared_root)}"
    if ! yolo_venv_shared_is_ready "$shared"; then
        echo "Shared YOLO venv not ready under $shared (missing .ready or python)" >&2
        exit 1
    fi
    mkdir -p "$UV_PROJECT_ENVIRONMENT"
    echo "[$(date -Is)] copying shared venv → $UV_PROJECT_ENVIRONMENT …"
    cp -a "$shared/." "$UV_PROJECT_ENVIRONMENT/"
    echo "[$(date -Is)] venv ready ($(du -sh "$UV_PROJECT_ENVIRONMENT" | cut -f1))"
}
