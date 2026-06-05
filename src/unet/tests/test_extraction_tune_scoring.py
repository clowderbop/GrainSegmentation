"""PQ-centered U-Net extraction profile tuning and method selection (issue 04)."""

from __future__ import annotations

import ast
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
from common.merged_view_pq import MERGED_VIEW_PQ_RESULT_KEYS
from unet.extraction_tune_scoring import (
    WatershedParamSet,
    mean_train_pq_for_watershed_params,
    select_best_watershed_tune_row,
    watershed_best_json_summary,
    watershed_per_sample_columns,
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


def test_mean_train_pq_for_watershed_params_returns_merged_view_pq_fields() -> None:
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

    mean_pq, per_sample = mean_train_pq_for_watershed_params([gt], [semantic], params)

    assert len(per_sample) == 1
    for key in MERGED_VIEW_PQ_RESULT_KEYS:
        assert key in mean_pq
        assert key in per_sample[0]
    assert mean_pq["pq"] == pytest.approx(per_sample[0]["pq"])
    for legacy in ("aji", "aji_plus", "iou75_precision", "mean_precision"):
        assert legacy not in mean_pq


def test_mean_train_pq_for_watershed_params_calls_watershed_once_per_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import unet.extraction_tune_scoring as scoring

    gt = _two_grain_gt()
    semantic = np.zeros(gt.shape, dtype=np.uint8)
    semantic[8:28, 8:28] = 1
    semantic[36:56, 36:56] = 1
    params = WatershedParamSet(5, 0, 1, 0, False, None)

    watershed_calls = 0
    real_map = scoring.instance_map_for_watershed_params

    def counting_map(pred_semantic: np.ndarray, params: WatershedParamSet) -> np.ndarray:
        nonlocal watershed_calls
        watershed_calls += 1
        return real_map(pred_semantic, params)

    monkeypatch.setattr(scoring, "instance_map_for_watershed_params", counting_map)

    mean_train_pq_for_watershed_params(
        [gt, gt.copy()],
        [semantic, semantic.copy()],
        params,
    )

    assert watershed_calls == 2


def test_mean_train_pq_for_watershed_params_logs_phase_timings(
    capsys: pytest.CaptureFixture[str],
) -> None:
    gt = _two_grain_gt()
    semantic = np.zeros(gt.shape, dtype=np.uint8)
    semantic[8:28, 8:28] = 1
    semantic[36:56, 36:56] = 1
    params = WatershedParamSet(5, 0, 1, 0, False, None)

    mean_train_pq_for_watershed_params(
        [gt],
        [semantic],
        params,
        sample_ids=["train"],
        log=True,
    )

    out = capsys.readouterr().out
    assert "running watershed" in out
    assert "running metrics" in out
    assert "watershed" in out
    assert "metrics" in out
    assert "PQ=" in out
    assert "DQ=" in out
    assert "pred=" in out


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
    from unet.extraction_tune_scoring import instance_map_for_watershed_params

    gt, semantic = _notched_two_grain_semantic()
    params_pq = WatershedParamSet(5, 0, 1, 0, False, None)
    params_aji = WatershedParamSet(2, 0, 1, 20, False, None)

    rows: list[dict[str, float | int]] = []
    for params in (params_aji, params_pq):
        mean_pq, _ = mean_train_pq_for_watershed_params([gt], [semantic], params)
        pred_instances = instance_map_for_watershed_params(semantic, params)
        legacy_aji = float(compute_aji(gt, pred_instances))
        rows.append(
            {
                "mean_pq": mean_pq["pq"],
                "legacy_aji": legacy_aji,
                "min_distance": params.min_distance,
                "min_area_px": params.min_area_px,
            }
        )

    pq_winner = max(rows, key=lambda row: float(row["mean_pq"]))
    aji_winner = max(rows, key=lambda row: float(row["legacy_aji"]))
    assert pq_winner["min_distance"] == 5
    assert aji_winner["min_distance"] == 2
    assert int(aji_winner["min_area_px"]) == 20
    assert pq_winner["mean_pq"] > aji_winner["mean_pq"]
    assert aji_winner["legacy_aji"] > pq_winner["legacy_aji"]

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


def test_watershed_best_json_summary_uses_merged_view_pq_not_bundle() -> None:
    params = WatershedParamSet(3, 0, 1, 0, False, None)
    mean_pq = {key: 0.5 for key in MERGED_VIEW_PQ_RESULT_KEYS}
    mean_pq["pq"] = 0.82
    row = watershed_tune_row(
        params,
        mean_pq,
        per_sample_pq=watershed_per_sample_columns(
            ["train"], [dict(mean_pq)], sanitize_sample_id=lambda s: s
        ),
    )

    summary = watershed_best_json_summary(
        row, params, ["train"], sanitize_sample_id=lambda s: s
    )

    assert summary["selection_objective"] == "pq"
    assert summary["best_mean_pq"] == pytest.approx(0.82)
    assert summary["best_per_sample_pq"] == {"train": pytest.approx(0.82)}
    assert isinstance(summary["best_mean_tp"], int)
    assert isinstance(summary["best_per_sample_tp"]["train"], int)
    assert "best_metric_bundle" not in summary
    assert "best_mean_aji" not in summary
    for key in MERGED_VIEW_PQ_RESULT_KEYS:
        assert f"best_mean_{key}" in summary
        assert f"best_per_sample_{key}" in summary


def test_watershed_tune_row_includes_merged_view_pq_fields_only() -> None:
    params = WatershedParamSet(3, 0, 1, 0, False, None)
    mean_pq = {key: 0.5 for key in MERGED_VIEW_PQ_RESULT_KEYS}
    mean_pq["pq"] = 0.82
    mean_pq["gt_instance_count"] = 2
    mean_pq["pred_instance_count"] = 2
    mean_pq["tp"] = 2
    per_sample = [dict(mean_pq)]

    row = watershed_tune_row(
        params,
        mean_pq,
        per_sample_pq=watershed_per_sample_columns(
            ["train"], per_sample, sanitize_sample_id=lambda s: s
        ),
    )

    fieldnames = watershed_tune_fieldnames(["train"], sanitize_sample_id=lambda s: s)
    assert set(row.keys()) == set(fieldnames)
    assert row["mean_pq"] == "0.82000000"
    assert row["mean_dq"] == "0.50000000"
    assert row["mean_tp"] == "2"
    assert row["pq__train"] == "0.82000000"
    assert "mean_aji" not in row
    assert "aji__train" not in row
    for bundle_key in ("iou75_precision", "aji_plus", "mean_precision"):
        assert f"mean_{bundle_key}" not in row


def test_extraction_method_selection_does_not_embed_sparse_pq_matching() -> None:
    """CC-vs-watershed selection compares eval bundles; tune-path PQ stays in extraction_tune_scoring."""
    source = (
        Path(__file__).resolve().parents[1] / "extraction_method_selection.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {
        "compute_merged_view_pq",
        "instance_overlap_stats",
        "greedy_one_to_one_matches_from_overlap",
    }
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert forbidden.isdisjoint(imported)
