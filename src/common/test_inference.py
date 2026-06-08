"""Load the shared test inference recipe (config/test_inference.yaml)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from common import yaml_validate as yv
from common.variants import repo_root

_RECIPE_RELATIVE = Path("config") / "test_inference.yaml"


@dataclass(frozen=True)
class WholeInferenceSpec:
    window: int
    stride: int


@dataclass(frozen=True)
class PatchInferenceSpec:
    imgsz: int


@dataclass(frozen=True)
class YoloValSpec:
    imgsz: int
    batch: int


@dataclass(frozen=True)
class YoloPatchBatchSpec:
    batch: int


@dataclass(frozen=True)
class YoloInferenceProfile:
    """SAHI slice-merge and mask threshold settings (ADR 0005)."""

    postprocess_type: str
    match_metric: str
    match_threshold: float
    mask_threshold: float


@dataclass(frozen=True)
class YoloInferenceProfileCandidate:
    """Conf grid point for profile selection; mask threshold fixed in test recipe."""

    conf: float
    mask_threshold: float

    def candidate_id(self) -> str:
        return f"c{self.conf:g}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "conf": self.conf,
            "mask_threshold": self.mask_threshold,
        }


@dataclass(frozen=True)
class YoloInferenceSpec:
    conf: float
    profile: YoloInferenceProfile
    patch: YoloPatchBatchSpec
    val: YoloValSpec


@dataclass(frozen=True)
class UnetWholeInferenceSpec:
    patch_size: int
    stride: int


@dataclass(frozen=True)
class UnetPatchInferenceSpec:
    patch_size: int
    stride: int
    batch_size: int


@dataclass(frozen=True)
class UnetInferenceSpec:
    whole: UnetWholeInferenceSpec
    patch: UnetPatchInferenceSpec


@dataclass(frozen=True)
class TestInferenceRecipe:
    whole: WholeInferenceSpec
    patch: PatchInferenceSpec
    yolo: YoloInferenceSpec
    unet: UnetInferenceSpec


def inference_recipe_path() -> Path:
    return repo_root() / _RECIPE_RELATIVE


def parse_yolo_inference_profile(yolo_raw: dict[str, Any]) -> YoloInferenceProfile:
    return YoloInferenceProfile(
        postprocess_type=yv.require_str(
            yolo_raw.get("postprocess_type"), context="yolo.postprocess_type"
        ),
        match_metric=yv.require_str(
            yolo_raw.get("match_metric"), context="yolo.match_metric"
        ),
        match_threshold=yv.require_float(
            yolo_raw.get("match_threshold"), context="yolo.match_threshold"
        ),
        mask_threshold=yv.require_float(
            yolo_raw.get("mask_threshold"), context="yolo.mask_threshold"
        ),
    )


_YOLO_PROFILE_SCALAR_KEYS = frozenset({"conf", "mask_threshold"})


def _yaml_scalar(value: float | str) -> str:
    if isinstance(value, str):
        if value.isidentifier() or value.replace("+", "").replace("#", "").isalnum():
            return value
        return yaml.safe_dump(value, default_flow_style=True).strip()
    if isinstance(value, float) and value == int(value):
        return str(int(value))
    return repr(value)


def profile_tune_fixed_mask_threshold(
    recipe: TestInferenceRecipe | None = None,
) -> float:
    """Fixed mask threshold from the test inference recipe (not a grid axis)."""
    resolved = recipe or load_test_inference_recipe()
    return resolved.yolo.profile.mask_threshold


def profile_tune_candidate_from_conf(
    conf: float,
    recipe: TestInferenceRecipe | None = None,
) -> YoloInferenceProfileCandidate:
    """Build a profile-selection candidate; mask threshold comes from the test recipe."""
    resolved = recipe or load_test_inference_recipe()
    return YoloInferenceProfileCandidate(
        conf=conf,
        mask_threshold=resolved.yolo.profile.mask_threshold,
    )


def rewrite_yolo_profile_in_recipe_text(
    text: str,
    candidate: YoloInferenceProfileCandidate,
    *,
    keys: frozenset[str] | tuple[str, ...] | None = None,
) -> str:
    """Update YOLO profile scalars in recipe YAML without rewriting unrelated keys or comments."""
    all_replacements = {
        "conf": candidate.conf,
        "mask_threshold": candidate.mask_threshold,
    }
    selected = frozenset(keys) if keys is not None else frozenset(all_replacements)
    replacements = {key: all_replacements[key] for key in selected}
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    in_yolo = False
    yolo_child_indent: int | None = None
    for line in lines:
        stripped = line.lstrip()
        if not stripped:
            out.append(line)
            continue
        indent = len(line) - len(stripped)
        if stripped.startswith("yolo:") and indent == 0:
            in_yolo = True
            yolo_child_indent = None
            out.append(line)
            continue
        if not in_yolo:
            out.append(line)
            continue
        if stripped.startswith("#"):
            out.append(line)
            continue
        if yolo_child_indent is None:
            yolo_child_indent = indent
        elif indent <= yolo_child_indent and indent == 0:
            in_yolo = False
            yolo_child_indent = None
            out.append(line)
            continue
        key = stripped.split(":", 1)[0]
        if key in replacements:
            newline = "\n" if line.endswith("\n") else ""
            out.append(
                f"{' ' * indent}{key}: {_yaml_scalar(replacements.pop(key))}{newline}"
            )
            continue
        out.append(line)
    if replacements:
        missing = ", ".join(sorted(replacements))
        raise ValueError(f"recipe text missing yolo profile keys: {missing}")
    return "".join(out)


def _parse_recipe(raw: dict[str, Any]) -> TestInferenceRecipe:
    whole_raw = yv.require_mapping(raw.get("whole"), context="whole")
    patch_raw = yv.require_mapping(raw.get("patch"), context="patch")
    yolo_raw = yv.require_mapping(raw.get("yolo"), context="yolo")
    unet_raw = yv.require_mapping(raw.get("unet"), context="unet")
    yolo_patch_raw = yv.require_mapping(yolo_raw.get("patch"), context="yolo.patch")
    yolo_val_raw = yv.require_mapping(yolo_raw.get("val"), context="yolo.val")
    unet_whole_raw = yv.require_mapping(unet_raw.get("whole"), context="unet.whole")
    unet_patch_raw = yv.require_mapping(unet_raw.get("patch"), context="unet.patch")

    return TestInferenceRecipe(
        whole=WholeInferenceSpec(
            window=yv.require_int(whole_raw.get("window"), context="whole.window"),
            stride=yv.require_int(whole_raw.get("stride"), context="whole.stride"),
        ),
        patch=PatchInferenceSpec(
            imgsz=yv.require_int(patch_raw.get("imgsz"), context="patch.imgsz"),
        ),
        yolo=YoloInferenceSpec(
            conf=yv.require_float(yolo_raw.get("conf"), context="yolo.conf"),
            profile=parse_yolo_inference_profile(yolo_raw),
            patch=YoloPatchBatchSpec(
                batch=yv.require_int(
                    yolo_patch_raw.get("batch"), context="yolo.patch.batch"
                ),
            ),
            val=YoloValSpec(
                imgsz=yv.require_int(yolo_val_raw.get("imgsz"), context="yolo.val.imgsz"),
                batch=yv.require_int(yolo_val_raw.get("batch"), context="yolo.val.batch"),
            ),
        ),
        unet=UnetInferenceSpec(
            whole=UnetWholeInferenceSpec(
                patch_size=yv.require_int(
                    unet_whole_raw.get("patch_size"), context="unet.whole.patch_size"
                ),
                stride=yv.require_int(
                    unet_whole_raw.get("stride"), context="unet.whole.stride"
                ),
            ),
            patch=UnetPatchInferenceSpec(
                patch_size=yv.require_int(
                    unet_patch_raw.get("patch_size"), context="unet.patch.patch_size"
                ),
                stride=yv.require_int(
                    unet_patch_raw.get("stride"), context="unet.patch.stride"
                ),
                batch_size=yv.require_int(
                    unet_patch_raw.get("batch_size"), context="unet.patch.batch_size"
                ),
            ),
        ),
    )


@lru_cache(maxsize=1)
def load_test_inference_recipe(
    path: Path | None = None,
) -> TestInferenceRecipe:
    resolved = path or inference_recipe_path()
    with resolved.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    doc = yv.require_mapping(raw, context=str(resolved))
    return _parse_recipe(doc)


def sahi_overlap_ratio(*, window: int, stride: int) -> float:
    """SAHI overlap-height/width ratio from window and stride (0.5 for 1024/512)."""
    if window <= 0 or stride <= 0 or stride > window:
        raise ValueError(f"Invalid window={window} stride={stride} for overlap ratio")
    return float(1.0 - stride / window)


def emit_shell_exports(recipe: TestInferenceRecipe | None = None) -> None:
    """Print bash export statements for SLURM test eval scripts."""
    r = recipe or load_test_inference_recipe()
    overlap = sahi_overlap_ratio(window=r.whole.window, stride=r.whole.stride)
    lines = [
        f"export TEST_WHOLE_WINDOW={r.whole.window}",
        f"export TEST_WHOLE_STRIDE={r.whole.stride}",
        f"export TEST_SAHI_OVERLAP={overlap}",
        f"export TEST_PATCH_IMGSZ={r.patch.imgsz}",
        f"export YOLO_CONF={r.yolo.conf}",
        f"export YOLO_MASK_THRESHOLD={r.yolo.profile.mask_threshold}",
        f"export YOLO_POSTPROCESS_TYPE={r.yolo.profile.postprocess_type}",
        f"export YOLO_MATCH_METRIC={r.yolo.profile.match_metric}",
        f"export YOLO_MATCH_THRESHOLD={r.yolo.profile.match_threshold}",
        f"export YOLO_PATCH_BATCH={r.yolo.patch.batch}",
        f"export YOLO_VAL_IMGSZ={r.yolo.val.imgsz}",
        f"export YOLO_VAL_BATCH={r.yolo.val.batch}",
        f"export UNET_WHOLE_PATCH_SIZE={r.unet.whole.patch_size}",
        f"export UNET_WHOLE_STRIDE={r.unet.whole.stride}",
        f"export UNET_PATCH_SIZE={r.unet.patch.patch_size}",
        f"export UNET_PATCH_STRIDE={r.unet.patch.stride}",
        f"export UNET_PATCH_BATCH_SIZE={r.unet.patch.batch_size}",
    ]
    print("\n".join(lines))


def main() -> None:
    emit_shell_exports()


if __name__ == "__main__":
    main()
