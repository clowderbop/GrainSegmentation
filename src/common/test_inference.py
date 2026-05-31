"""Load the shared test inference recipe (configs/test_inference.yaml)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from common.variants import repo_root

_RECIPE_RELATIVE = Path("configs") / "test_inference.yaml"


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
class YoloInferenceSpec:
    conf: float
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


def _require_mapping(raw: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{context} must be a YAML mapping")
    return raw


def _require_int(raw: Any, *, context: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"{context} must be an integer, got {raw!r}")
    return raw


def _require_float(raw: Any, *, context: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"{context} must be a number, got {raw!r}")
    return float(raw)


def _parse_recipe(raw: dict[str, Any]) -> TestInferenceRecipe:
    whole_raw = _require_mapping(raw.get("whole"), context="whole")
    patch_raw = _require_mapping(raw.get("patch"), context="patch")
    yolo_raw = _require_mapping(raw.get("yolo"), context="yolo")
    unet_raw = _require_mapping(raw.get("unet"), context="unet")
    yolo_patch_raw = _require_mapping(yolo_raw.get("patch"), context="yolo.patch")
    yolo_val_raw = _require_mapping(yolo_raw.get("val"), context="yolo.val")
    unet_whole_raw = _require_mapping(unet_raw.get("whole"), context="unet.whole")
    unet_patch_raw = _require_mapping(unet_raw.get("patch"), context="unet.patch")

    return TestInferenceRecipe(
        whole=WholeInferenceSpec(
            window=_require_int(whole_raw.get("window"), context="whole.window"),
            stride=_require_int(whole_raw.get("stride"), context="whole.stride"),
        ),
        patch=PatchInferenceSpec(
            imgsz=_require_int(patch_raw.get("imgsz"), context="patch.imgsz"),
        ),
        yolo=YoloInferenceSpec(
            conf=_require_float(yolo_raw.get("conf"), context="yolo.conf"),
            patch=YoloPatchBatchSpec(
                batch=_require_int(
                    yolo_patch_raw.get("batch"), context="yolo.patch.batch"
                ),
            ),
            val=YoloValSpec(
                imgsz=_require_int(yolo_val_raw.get("imgsz"), context="yolo.val.imgsz"),
                batch=_require_int(yolo_val_raw.get("batch"), context="yolo.val.batch"),
            ),
        ),
        unet=UnetInferenceSpec(
            whole=UnetWholeInferenceSpec(
                patch_size=_require_int(
                    unet_whole_raw.get("patch_size"), context="unet.whole.patch_size"
                ),
                stride=_require_int(
                    unet_whole_raw.get("stride"), context="unet.whole.stride"
                ),
            ),
            patch=UnetPatchInferenceSpec(
                patch_size=_require_int(
                    unet_patch_raw.get("patch_size"), context="unet.patch.patch_size"
                ),
                stride=_require_int(
                    unet_patch_raw.get("stride"), context="unet.patch.stride"
                ),
                batch_size=_require_int(
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
    doc = _require_mapping(raw, context=str(resolved))
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
