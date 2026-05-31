"""Tests for YOLO inference profile train selection (ADR 0005)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from common.test_inference import (
    YoloInferenceProfileCandidate,
    load_test_inference_recipe,
)

from yolo.inference_profile_tune import (
    extract_mean_aji_from_report,
    iter_stage1_candidates,
    iter_stage2_candidates,
    load_stage_winner,
    load_tune_grid,
    mean_aji_across_variants,
    promote_profile_to_recipe,
    score_candidate_across_variants,
    select_best_candidate,
    write_stage_winner_json,
)


def _write_grid(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "stage1": {
                    "postprocess_type": ["GREEDYNMM", "NMM"],
                    "match_metric": ["IOS"],
                    "match_threshold": [0.4, 0.5],
                },
                "stage2": {
                    "conf": [0.2, 0.3],
                    "mask_threshold": [0.45, 0.55],
                },
            }
        ),
        encoding="utf-8",
    )


def test_load_tune_grid_without_stage1_fixed_uses_recipe_baseline(tmp_path: Path) -> None:
    grid_path = tmp_path / "grid.yaml"
    _write_grid(grid_path)
    recipe = load_test_inference_recipe()
    spec = load_tune_grid(grid_path)
    assert spec.stage1_fixed.conf == recipe.yolo.conf
    assert spec.stage1_fixed.mask_threshold == recipe.yolo.profile.mask_threshold


def test_load_tune_grid_reads_staged_search_space(tmp_path: Path) -> None:
    grid_path = tmp_path / "grid.yaml"
    _write_grid(grid_path)
    recipe = load_test_inference_recipe()
    spec = load_tune_grid(grid_path)
    assert spec.stage1_fixed.conf == recipe.yolo.conf
    assert spec.stage1_fixed.mask_threshold == recipe.yolo.profile.mask_threshold
    assert spec.stage1.postprocess_type == ("GREEDYNMM", "NMM")
    assert spec.stage2.conf == (0.2, 0.3)


def test_iter_stage1_candidates_cartesian_merge_knobs_only(tmp_path: Path) -> None:
    _write_grid(tmp_path / "grid.yaml")
    spec = load_tune_grid(tmp_path / "grid.yaml")
    recipe = load_test_inference_recipe()
    candidates = list(iter_stage1_candidates(spec))
    assert len(candidates) == 2 * 1 * 2  # postprocess × metric × threshold
    assert all(
        c.conf == recipe.yolo.conf and c.mask_threshold == recipe.yolo.profile.mask_threshold
        for c in candidates
    )
    assert candidates[0].postprocess_type == "GREEDYNMM"
    assert candidates[0].match_metric == "IOS"


def test_iter_stage2_candidates_uses_stage1_winner_merge_settings(tmp_path: Path) -> None:
    _write_grid(tmp_path / "grid.yaml")
    spec = load_tune_grid(tmp_path / "grid.yaml")
    winner = YoloInferenceProfileCandidate(
        postprocess_type="NMM",
        match_metric="IOS",
        match_threshold=0.4,
        conf=0.25,
        mask_threshold=0.5,
    )
    stage2 = list(iter_stage2_candidates(spec, winner))
    assert len(stage2) == 2 * 2
    assert all(
        c.postprocess_type == "NMM"
        and c.match_metric == "IOS"
        and c.match_threshold == 0.4
        for c in stage2
    )
    assert {(c.conf, c.mask_threshold) for c in stage2} == {
        (0.2, 0.45),
        (0.2, 0.55),
        (0.3, 0.45),
        (0.3, 0.55),
    }


def test_mean_aji_across_variants_equal_weights_per_variant() -> None:
    scores = {"PPL": 0.8, "PPL+AllPPX": 0.6}
    assert mean_aji_across_variants(scores) == pytest.approx(0.7)


def test_extract_mean_aji_from_single_sample_report() -> None:
    report = {
        "samples": [{"sample_id": "train", "aji": 0.42}],
    }
    assert extract_mean_aji_from_report(report) == pytest.approx(0.42)


def test_extract_mean_aji_prefers_report_mean_when_present() -> None:
    report = {
        "samples": [{"sample_id": "a", "aji": 0.1}, {"sample_id": "b", "aji": 0.9}],
        "mean": {"aji": 0.5},
    }
    assert extract_mean_aji_from_report(report) == pytest.approx(0.5)


def test_extract_mean_aji_averages_samples_when_mean_absent() -> None:
    report = {
        "samples": [
            {"sample_id": "a", "aji": 0.2},
            {"sample_id": "b", "aji": 0.8},
        ],
    }
    assert extract_mean_aji_from_report(report) == pytest.approx(0.5)


def test_select_best_candidate_maximizes_mean_train_aji() -> None:
    rows = [
        {"candidate_id": "a", "mean_aji": 0.71},
        {"candidate_id": "b", "mean_aji": 0.82},
        {"candidate_id": "c", "mean_aji": 0.80},
    ]
    best = select_best_candidate(rows)
    assert best["candidate_id"] == "b"


def test_promote_profile_preserves_recipe_comments_and_structure(tmp_path: Path) -> None:
    recipe_path = tmp_path / "test_inference.yaml"
    recipe_path.write_text(
        """# Shared test inference recipe (ADR 0003).
whole:
  window: 1024
  stride: 512
patch:
  imgsz: 1024
yolo:
  # YOLO inference profile (ADR 0005).
  conf: 0.25
  mask_threshold: 0.5
  postprocess_type: GREEDYNMM
  match_metric: IOS
  match_threshold: 0.5
  patch:
    batch: 16
  val:
    imgsz: 1024
    batch: 16
unet:
  whole:
    patch_size: 1024
    stride: 512
  patch:
    patch_size: 1024
    stride: 1024
    batch_size: 1
""",
        encoding="utf-8",
    )
    profile = YoloInferenceProfileCandidate(
        postprocess_type="NMM",
        match_metric="IOS",
        match_threshold=0.7,
        conf=0.2,
        mask_threshold=0.45,
    )
    promote_profile_to_recipe(profile, recipe_path)
    updated_text = recipe_path.read_text(encoding="utf-8")
    assert "# Shared test inference recipe (ADR 0003)." in updated_text
    assert "# YOLO inference profile (ADR 0005)." in updated_text
    assert "  conf: 0.2" in updated_text
    assert "  postprocess_type: NMM" in updated_text
    assert "  patch:\n    batch: 16" in updated_text


def test_promote_profile_to_recipe_updates_yolo_fields_only(tmp_path: Path) -> None:
    recipe_path = tmp_path / "test_inference.yaml"
    recipe_path.write_text(
        """whole:
  window: 1024
  stride: 512
patch:
  imgsz: 1024
yolo:
  conf: 0.25
  mask_threshold: 0.5
  postprocess_type: GREEDYNMM
  match_metric: IOS
  match_threshold: 0.5
  patch:
    batch: 16
  val:
    imgsz: 1024
    batch: 16
unet:
  whole:
    patch_size: 1024
    stride: 512
  patch:
    patch_size: 1024
    stride: 1024
    batch_size: 1
""",
        encoding="utf-8",
    )
    profile = YoloInferenceProfileCandidate(
        postprocess_type="NMM",
        match_metric="IOS",
        match_threshold=0.7,
        conf=0.2,
        mask_threshold=0.45,
    )
    promote_profile_to_recipe(profile, recipe_path)
    updated = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    assert updated["yolo"]["postprocess_type"] == "NMM"
    assert updated["yolo"]["match_threshold"] == 0.7
    assert updated["yolo"]["conf"] == 0.2
    assert updated["whole"]["window"] == 1024
    assert updated["unet"]["patch"]["batch_size"] == 1


def test_promote_profile_to_recipe_round_trips_through_recipe_loader(tmp_path: Path) -> None:
    recipe_path = tmp_path / "test_inference.yaml"
    recipe_path.write_text(
        """whole:
  window: 1024
  stride: 512
patch:
  imgsz: 1024
yolo:
  conf: 0.25
  mask_threshold: 0.5
  postprocess_type: GREEDYNMM
  match_metric: IOS
  match_threshold: 0.5
  patch:
    batch: 16
  val:
    imgsz: 1024
    batch: 16
unet:
  whole:
    patch_size: 1024
    stride: 512
  patch:
    patch_size: 1024
    stride: 1024
    batch_size: 1
""",
        encoding="utf-8",
    )
    profile = YoloInferenceProfileCandidate(
        postprocess_type="NMM",
        match_metric="IOS",
        match_threshold=0.7,
        conf=0.2,
        mask_threshold=0.45,
    )
    promote_profile_to_recipe(profile, recipe_path)
    loaded = load_test_inference_recipe(recipe_path)
    assert loaded.yolo.conf == profile.conf
    assert loaded.yolo.profile.mask_threshold == profile.mask_threshold
    assert loaded.yolo.profile.postprocess_type == profile.postprocess_type
    assert loaded.yolo.profile.match_metric == profile.match_metric
    assert loaded.yolo.profile.match_threshold == profile.match_threshold


def test_promote_profile_rejects_invalid_profile_and_preserves_recipe(tmp_path: Path) -> None:
    recipe_path = tmp_path / "test_inference.yaml"
    original = """whole:
  window: 1024
  stride: 512
patch:
  imgsz: 1024
yolo:
  conf: 0.25
  mask_threshold: 0.5
  postprocess_type: GREEDYNMM
  match_metric: IOS
  match_threshold: 0.5
  patch:
    batch: 16
  val:
    imgsz: 1024
    batch: 16
unet:
  whole:
    patch_size: 1024
    stride: 512
  patch:
    patch_size: 1024
    stride: 1024
    batch_size: 1
"""
    recipe_path.write_text(original, encoding="utf-8")
    profile = YoloInferenceProfileCandidate(
        postprocess_type="",
        match_metric="IOS",
        match_threshold=0.5,
        conf=0.25,
        mask_threshold=0.5,
    )
    with pytest.raises(ValueError, match="postprocess_type"):
        promote_profile_to_recipe(profile, recipe_path)
    assert recipe_path.read_text(encoding="utf-8") == original


def test_committed_tune_grid_loads() -> None:
    from common.variants import repo_root

    spec = load_tune_grid(repo_root() / "configs" / "yolo_inference_profile_tune.yaml")
    assert len(spec.stage1.postprocess_type) >= 2
    assert len(spec.stage2.conf) >= 2


def test_load_stage_winner_for_promote_requires_stage_two(tmp_path: Path) -> None:
    profile = YoloInferenceProfileCandidate(
        postprocess_type="GREEDYNMM",
        match_metric="IOS",
        match_threshold=0.5,
        conf=0.25,
        mask_threshold=0.5,
    )
    stage1_path = tmp_path / "stage1_winner.json"
    write_stage_winner_json(
        stage1_path, stage=1, candidate=profile, mean_aji=0.7, per_variant={"PPL": 0.7}
    )
    with pytest.raises(ValueError, match="stage 2"):
        load_stage_winner(stage1_path, expected_stage=2)


def test_score_candidate_across_variants_reads_variant_reports(tmp_path: Path) -> None:
    def _write_report(path: Path, aji: float) -> None:
        path.write_text(
            json.dumps({"samples": [{"sample_id": "train", "aji": aji}]}),
            encoding="utf-8",
        )

    ppl_report = tmp_path / "ppl.json"
    ppx_report = tmp_path / "ppx.json"
    _write_report(ppl_report, 0.8)
    _write_report(ppx_report, 0.6)
    mean_aji, per_variant = score_candidate_across_variants(
        {"PPL": ppl_report, "PPL+AllPPX": ppx_report}
    )
    assert mean_aji == pytest.approx(0.7)
    assert per_variant["PPL"] == pytest.approx(0.8)
    assert per_variant["PPL+AllPPX"] == pytest.approx(0.6)


def test_uv_run_cmd_uses_uv_directory_for_each_project() -> None:
    from yolo.tune_inference_profile import _uv_run_cmd

    yolo_cmd = _uv_run_cmd(Path("/repo/src/yolo"), "yolo.predict", "--unit", "whole")
    assert yolo_cmd[:4] == ["uv", "run", "--directory", "/repo/src/yolo"]
    assert yolo_cmd[4:7] == ["python", "-m", "yolo.predict"]

    common_cmd = _uv_run_cmd(
        Path("/repo/src/common"), "common.evaluate_instances", unbuffered=True
    )
    assert common_cmd[4:7] == ["python", "-u", "-m"]


def test_score_variant_on_cluster_runs_predict_write_eval_and_evaluate(
    tmp_path: Path,
) -> None:
    from yolo.tune_inference_profile import _score_variant_on_cluster

    repo = tmp_path / "repo"
    common_src = repo / "src" / "common"
    yolo_src = repo / "src" / "yolo"
    common_src.mkdir(parents=True)
    yolo_src.mkdir(parents=True)
    grainseg_root = tmp_path / "grainseg"
    run_root = grainseg_root / "runs" / "yolo26-seg"
    weights = run_root / "PPL" / "weights" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_text("", encoding="utf-8")
    work_root = tmp_path / "work"
    variant_out = tmp_path / "variant_out"
    candidate = YoloInferenceProfileCandidate(
        postprocess_type="NMM",
        match_metric="IOU",
        match_threshold=0.6,
        conf=0.3,
        mask_threshold=0.55,
    )
    invoked: list[list[str]] = []

    def _fake_run(cmd: list[str]) -> None:
        invoked.append(cmd)
        if "yolo.predict" in cmd:
            variant_out.mkdir(parents=True, exist_ok=True)
            (variant_out / "run_provenance.json").write_text("{}", encoding="utf-8")
        elif "write-eval" in cmd:
            (variant_out / "eval_manifest.json").write_text("{}", encoding="utf-8")
        elif "evaluate_instances" in cmd:
            metrics = variant_out / "instance_metrics.json"
            metrics.write_text(
                json.dumps({"samples": [{"sample_id": "train", "aji": 0.5}]}),
                encoding="utf-8",
            )

    staged_manifest = work_root / "PPL" / "staged" / "manifest.json"
    staged_manifest.parent.mkdir(parents=True)
    staged_manifest.write_text("{}", encoding="utf-8")

    with (
        patch("yolo.tune_inference_profile._run_subprocess", side_effect=_fake_run),
        patch(
            "yolo.tune_inference_profile._prepare_train_whole_manifest",
            return_value=grainseg_root / "canonical.json",
        ),
        patch(
            "yolo.tune_inference_profile._staged_manifest_path",
            return_value=staged_manifest,
        ),
    ):
        metrics_path = _score_variant_on_cluster(
            variant="PPL",
            candidate=candidate,
            variant_output_dir=variant_out,
            grainseg_root=grainseg_root,
            run_root=run_root,
            work_root=work_root,
            device="0",
            repo=repo,
        )

    assert metrics_path == variant_out / "instance_metrics.json"
    assert len(invoked) == 4
    modules = [cmd[cmd.index("-m") + 1] for cmd in invoked if "-m" in cmd]
    assert modules == [
        "common.stage_manifest",
        "yolo.predict",
        "common.stage_manifest",
        "common.evaluate_instances",
    ]
    predict_cmd = invoked[1]
    assert "yolo.predict" in predict_cmd
    assert "--conf" in predict_cmd and "0.3" in predict_cmd
    assert "--mask-threshold" in predict_cmd and "0.55" in predict_cmd
    assert "--postprocess-type" in predict_cmd and "NMM" in predict_cmd


def test_extract_mean_aji_from_evaluate_instances_report_shape() -> None:
    from common.reporting import build_instance_eval_report

    report = build_instance_eval_report(
        model_type="yolo",
        variant="PPL",
        unit="whole",
        samples=[
            {
                "sample_id": "train",
                "aji": 0.42,
                "f1_iou50": 0.5,
                "gt_instances": 1,
                "predicted_grain_count": 1,
                "empty_gt": False,
            }
        ],
    )
    assert extract_mean_aji_from_report(report) == pytest.approx(0.42)


def test_run_stage_search_dry_run_picks_best_across_all_registry_variants(
    tmp_path: Path,
) -> None:
    from common.variants import all_variant_names
    from yolo.tune_inference_profile import _dry_run_scorer, run_stage_search

    variants = all_variant_names()
    _write_grid(tmp_path / "grid.yaml")
    spec = load_tune_grid(tmp_path / "grid.yaml")
    candidates = list(iter_stage1_candidates(spec))[:2]
    better, worse = candidates[0], candidates[1]
    scores: dict[tuple[str, str], float] = {}
    for variant in variants:
        scores[(variant, better.candidate_id())] = 0.9
        scores[(variant, worse.candidate_id())] = 0.1
    winner, _ = run_stage_search(
        stage=1,
        candidates=[better, worse],
        variants=variants,
        output_dir=tmp_path / "tune_all",
        score_variant=_dry_run_scorer(scores),
    )
    assert winner == better


def test_run_stage_search_dry_run_writes_winner(tmp_path: Path) -> None:
    from yolo.tune_inference_profile import _dry_run_scorer, run_stage_search

    _write_grid(tmp_path / "grid.yaml")
    spec = load_tune_grid(tmp_path / "grid.yaml")
    candidates = list(iter_stage1_candidates(spec))
    assert len(candidates) >= 2
    better, worse = candidates[0], candidates[1]
    scorer = _dry_run_scorer(
        {
            ("PPL", better.candidate_id()): 0.9,
            ("PPL", worse.candidate_id()): 0.4,
        }
    )
    winner, _ = run_stage_search(
        stage=1,
        candidates=[better, worse],
        variants=("PPL",),
        output_dir=tmp_path / "tune",
        score_variant=scorer,
    )
    assert winner == better
    assert (tmp_path / "tune" / "stage1" / "winner.json").is_file()
    assert (tmp_path / "tune" / "stage1" / "results.csv").is_file()
