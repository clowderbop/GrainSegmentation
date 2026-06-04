"""PQ-centered U-Net extraction profile tuning and method selection (issue 04)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from common.evaluate_instances import evaluate_instance_samples
from common.instance_eval_report import instance_metrics_report_path_for_variant
from common.instance_metric_bundle import INSTANCE_METRIC_BUNDLE_KEYS, compute_instance_metric_bundle
from common.metrics import compute_aji
from common.tests.evaluate_instances_fixtures import perfect_match_eval_sample
from unet.extraction_method_selection import (
    select_train_extraction_method,
    select_train_extraction_method_from_eval_dirs,
    select_train_extraction_method_from_reports,
    write_extraction_method_selection_json,
)
from unet.extraction_tune_scoring import (
    WatershedParamSet,
    mean_aji_for_watershed_params,
    mean_train_bundle_for_watershed_params,
    select_best_watershed_tune_row,
    watershed_tune_fieldnames,
    watershed_tune_row,
)


def _paint_box(
    label_map: np.ndarray, instance_id: int, r0: int, c0: int, r1: int, c1: int
) -> None:
    label_map[r0:r1, c0:c1] = instance_id


def _two_grain_gt(height: int = 64, width: int = 64) -> np.ndarray:
    gt = np.zeros((height, width), dtype=np.int32)
    _paint_box(gt, 1, 8, 8, 28, 28)
    _paint_box(gt, 2, 36, 36, 56, 56)
    return gt


def _merged_plus_perfect_grain_pred(height: int = 64, width: int = 64) -> np.ndarray:
    """High PQ, low AJI on two-grain GT (synthetic extraction candidate A)."""
    pred = np.zeros((height, width), dtype=np.int32)
    _paint_box(pred, 1, 6, 6, 58, 58)
    _paint_box(pred, 2, 36, 36, 56, 56)
    return pred


def _split_first_grain_pred(height: int = 64, width: int = 64) -> np.ndarray:
    """Lower PQ, higher AJI on two-grain GT (synthetic extraction candidate B)."""
    pred = np.zeros((height, width), dtype=np.int32)
    _paint_box(pred, 1, 8, 8, 18, 28)
    _paint_box(pred, 2, 18, 8, 28, 28)
    _paint_box(pred, 3, 36, 36, 56, 56)
    return pred


def test_mean_train_bundle_for_watershed_params_returns_full_bundle() -> None:
    gt = _two_grain_gt()
    semantic = np.zeros(gt.shape, dtype=np.uint8)
    semantic[8:28, 8:28] = 1
    semantic[36:56, 36:56] = 1
    params = WatershedParamSet(
        min_distance=5,
        boundary_dilate_iter=0,
        watershed_connectivity=1,
        min_area_px=0,
        exclude_border=False,
        ridge_level=None,
    )

    mean_bundle, per_sample = mean_train_bundle_for_watershed_params(
        [gt], [semantic], params
    )

    assert len(per_sample) == 1
    for key in INSTANCE_METRIC_BUNDLE_KEYS:
        assert key in mean_bundle
        assert key in per_sample[0]
    assert mean_bundle["pq"] == pytest.approx(per_sample[0]["pq"])


def test_select_best_watershed_tune_row_uses_mean_pq_not_mean_aji() -> None:
    rows = [
        {
            "candidate_id": "high_aji",
            "mean_pq": 0.4,
            "mean_aji": 0.6,
            "min_distance": 2,
        },
        {
            "candidate_id": "high_pq",
            "mean_pq": 0.5,
            "mean_aji": 0.296,
            "min_distance": 1,
        },
    ]

    best = select_best_watershed_tune_row(rows)

    assert best["candidate_id"] == "high_pq"
    assert best["mean_pq"] == pytest.approx(0.5)


def test_synthetic_extraction_maps_pq_and_aji_winners_differ() -> None:
    """Fixture maps invert PQ vs legacy AJI ranking (acceptance criterion)."""
    gt = _two_grain_gt()
    pred_pq_winner = _merged_plus_perfect_grain_pred()
    pred_aji_winner = _split_first_grain_pred()

    pq_winner_bundle = compute_instance_metric_bundle(gt, pred_pq_winner)
    aji_winner_bundle = compute_instance_metric_bundle(gt, pred_aji_winner)

    assert pq_winner_bundle["pq"] > aji_winner_bundle["pq"]
    assert compute_aji(gt, pred_aji_winner) > compute_aji(gt, pred_pq_winner)


def test_select_train_extraction_method_uses_pq_not_aji() -> None:
    gt = _two_grain_gt()
    cc_bundle = compute_instance_metric_bundle(gt, _merged_plus_perfect_grain_pred())
    watershed_bundle = compute_instance_metric_bundle(gt, _split_first_grain_pred())

    selection = select_train_extraction_method(
        cc_bundle=cc_bundle,
        watershed_bundle=watershed_bundle,
    )

    assert selection.selected_method == "cc"
    assert selection.objective_pq == pytest.approx(cc_bundle["pq"])
    assert selection.cc.bundle["pq"] == pytest.approx(cc_bundle["pq"])
    assert selection.watershed.bundle["aji_plus"] == pytest.approx(
        watershed_bundle["aji_plus"]
    )
    assert selection.watershed.bundle["pq"] < selection.cc.bundle["pq"]
    assert selection.watershed.bundle["aji_plus"] > selection.cc.bundle["aji_plus"]


def _notched_two_grain_semantic(height: int = 64, width: int = 64) -> tuple[np.ndarray, np.ndarray]:
    gt = _two_grain_gt(height, width)
    semantic = np.zeros((height, width), dtype=np.uint8)
    semantic[8:28, 8:28] = 1
    semantic[36:56, 36:56] = 1
    semantic[17:19, 8:28] = 0
    return gt, semantic


def test_watershed_param_candidates_pq_winner_differs_from_aji_winner() -> None:
    """Synthetic watershed grid: mean PQ and legacy AJI pick different param sets."""
    gt, semantic = _notched_two_grain_semantic()
    params_pq = WatershedParamSet(5, 0, 1, 0, False, None)
    params_aji = WatershedParamSet(2, 0, 1, 20, False, None)

    rows: list[dict[str, float | int]] = []
    for params in (params_aji, params_pq):
        mean_bundle, _ = mean_train_bundle_for_watershed_params([gt], [semantic], params)
        mean_aji, _ = mean_aji_for_watershed_params([gt], [semantic], params)
        rows.append(
            {
                "mean_pq": mean_bundle["pq"],
                "mean_aji": mean_aji,
                "min_distance": params.min_distance,
                "min_area_px": params.min_area_px,
            }
        )

    pq_winner = max(rows, key=lambda row: float(row["mean_pq"]))
    aji_winner = max(rows, key=lambda row: float(row["mean_aji"]))
    assert pq_winner["min_distance"] == 5
    assert aji_winner["min_distance"] == 2
    assert int(aji_winner["min_area_px"]) == 20
    assert pq_winner["mean_pq"] > aji_winner["mean_pq"]
    assert aji_winner["mean_aji"] > pq_winner["mean_aji"]

    best = select_best_watershed_tune_row(rows)
    assert best["min_distance"] == 5
    assert int(best["min_area_px"]) == 0


def test_select_train_extraction_method_from_eval_reports_rejects_patch_unit(
    tmp_path: Path,
) -> None:
    patch_report = evaluate_instance_samples(
        [perfect_match_eval_sample(tmp_path, sample_id="train")],
        model_type="unet",
        variant="PPL",
        unit="patch",
    )
    whole_report = evaluate_instance_samples(
        [perfect_match_eval_sample(tmp_path, sample_id="train")],
        model_type="unet",
        variant="PPL",
        unit="whole",
    )
    cc_path = tmp_path / "cc.json"
    ws_path = tmp_path / "ws.json"
    cc_path.write_text(json.dumps(patch_report), encoding="utf-8")
    ws_path.write_text(json.dumps(whole_report), encoding="utf-8")

    with pytest.raises(ValueError, match="unit='whole'"):
        select_train_extraction_method_from_reports(
            cc_report_path=cc_path,
            watershed_report_path=ws_path,
        )


def test_select_train_extraction_method_from_eval_reports(tmp_path: Path) -> None:
    gt_sample = perfect_match_eval_sample(tmp_path, sample_id="train")
    cc_report = evaluate_instance_samples(
        [gt_sample],
        model_type="unet",
        variant="PPL",
        unit="whole",
        extras={"instance_method": "cc"},
    )
    watershed_report = evaluate_instance_samples(
        [gt_sample],
        model_type="unet",
        variant="PPL",
        unit="whole",
        extras={"instance_method": "watershed"},
    )
    cc_path = tmp_path / "cc.json"
    ws_path = tmp_path / "watershed.json"
    cc_path.write_text(json.dumps(cc_report), encoding="utf-8")
    ws_path.write_text(json.dumps(watershed_report), encoding="utf-8")

    selection = select_train_extraction_method_from_reports(
        cc_report_path=cc_path,
        watershed_report_path=ws_path,
    )
    out_path = tmp_path / "selection.json"
    write_extraction_method_selection_json(out_path, selection)

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["selection_objective"] == "pq"
    assert payload["unit"] == "whole"
    assert payload["manifest_split"] == "train"
    assert payload["selected_method"] in {"cc", "watershed"}
    for method in ("cc", "watershed"):
        for key in INSTANCE_METRIC_BUNDLE_KEYS:
            assert key in payload[method]


def test_select_train_extraction_method_from_eval_dirs(tmp_path: Path) -> None:
    sample = perfect_match_eval_sample(tmp_path, sample_id="train")
    report = evaluate_instance_samples(
        [sample], model_type="unet", variant="PPL", unit="whole"
    )
    cc_dir = tmp_path / "cc"
    ws_dir = tmp_path / "watershed"
    for root in (cc_dir, ws_dir):
        for variant in ("PPL", "PPLPPXblend"):
            path = instance_metrics_report_path_for_variant(root, variant)
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(report), encoding="utf-8")

    selection = select_train_extraction_method_from_eval_dirs(
        cc_eval_dir=cc_dir,
        watershed_eval_dir=ws_dir,
        variant_names=("PPL", "PPLPPXblend"),
    )
    assert selection.cc.per_variant_bundles["PPL"]["pq"] == pytest.approx(1.0)
    assert selection.watershed.per_variant_bundles["PPLPPXblend"]["pq"] == pytest.approx(
        1.0
    )
    assert selection.selected_method in {"cc", "watershed"}


def test_watershed_tune_row_includes_bundle_and_audit_fields() -> None:
    params = WatershedParamSet(3, 0, 1, 0, False, None)
    mean_bundle = {key: 0.5 for key in INSTANCE_METRIC_BUNDLE_KEYS}
    mean_bundle["pq"] = 0.82
    mean_bundle["gt_instance_count"] = 2
    mean_bundle["pred_instance_count"] = 2

    row = watershed_tune_row(
        params,
        mean_bundle,
        mean_aji=0.91,
        per_sample_aji={"aji__train": "0.91000000"},
    )

    fieldnames = watershed_tune_fieldnames(["train"], sanitize_sample_id=lambda s: s)
    assert set(row.keys()) == set(fieldnames)
    assert row["mean_pq"] == "0.82000000"
    assert row["mean_aji"] == "0.91000000"
    assert row["mean_dq"] == "0.50000000"
