"""Tests for YOLO inference profile train selection (ADR 0005)."""

from __future__ import annotations

import json
from math import prod
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from common.test_inference import (
    YoloInferenceProfileCandidate,
    load_test_inference_recipe,
)

from yolo.inference_profile_tune import (
    GridResumeContext,
    append_grid_result_row,
    extract_mean_aji_from_report,
    iter_detector_jobs,
    iter_grid_candidates,
    load_grid_results_csv,
    load_grid_winner,
    load_tune_grid,
    mean_aji_across_variants,
    metrics_resume_valid,
    promote_profile_to_recipe,
    run_grid_search,
    score_candidate_across_variants,
    select_best_candidate,
    should_skip_variant_eval,
    variant_eval_fingerprint,
    write_grid_winner_json,
    write_variant_eval_resume_meta,
)
from yolo.profile_tune_dry_run import dry_run_scorer
from yolo.profile_tune_cli import validate_detector_caches


def _write_grid(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "grid": {
                    "postprocess_type": ["GREEDYNMM", "NMM"],
                    "match_metric": ["IOS"],
                    "match_threshold": [0.4, 0.5],
                    "conf": [0.2, 0.3],
                    "mask_threshold": [0.45, 0.55],
                },
            }
        ),
        encoding="utf-8",
    )


def test_load_tune_grid_reads_full_factorial_search_space(tmp_path: Path) -> None:
    grid_path = tmp_path / "grid.yaml"
    _write_grid(grid_path)
    spec = load_tune_grid(grid_path)
    assert spec.grid.postprocess_type == ("GREEDYNMM", "NMM")
    assert spec.grid.match_metric == ("IOS",)
    assert spec.grid.match_threshold == (0.4, 0.5)
    assert spec.grid.conf == (0.2, 0.3)
    assert spec.grid.mask_threshold == (0.45, 0.55)


def test_iter_detector_jobs_one_per_variant_and_detector_key(tmp_path: Path) -> None:
    _write_grid(tmp_path / "grid.yaml")
    spec = load_tune_grid(tmp_path / "grid.yaml")
    jobs = list(iter_detector_jobs(spec, ("PPL", "PPL+AllPPX")))
    assert len(jobs) == 2 * 2 * 2  # variants × conf × mask_threshold


def test_iter_grid_candidates_full_factorial_product(tmp_path: Path) -> None:
    _write_grid(tmp_path / "grid.yaml")
    spec = load_tune_grid(tmp_path / "grid.yaml")
    candidates = list(iter_grid_candidates(spec))
    assert len(candidates) == 2 * 1 * 2 * 2 * 2
    assert candidates[0] == YoloInferenceProfileCandidate(
        postprocess_type="GREEDYNMM",
        match_metric="IOS",
        match_threshold=0.4,
        conf=0.2,
        mask_threshold=0.45,
    )
    assert len({c.candidate_id() for c in candidates}) == len(candidates)


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


def test_committed_tune_grid_matches_yaml_factorial() -> None:
    from common.variants import repo_root

    spec = load_tune_grid(repo_root() / "configs" / "yolo_inference_profile_tune.yaml")
    grid = spec.grid
    expected = prod(
        (
            len(grid.postprocess_type),
            len(grid.match_metric),
            len(grid.match_threshold),
            len(grid.conf),
            len(grid.mask_threshold),
        )
    )
    candidates = list(iter_grid_candidates(spec))
    assert len(candidates) == expected
    assert grid.match_metric == ("IOS", "IOU")


def test_load_grid_winner_round_trips_profile(tmp_path: Path) -> None:
    profile = YoloInferenceProfileCandidate(
        postprocess_type="GREEDYNMM",
        match_metric="IOS",
        match_threshold=0.5,
        conf=0.25,
        mask_threshold=0.5,
    )
    winner_path = tmp_path / "grid" / "winner.json"
    write_grid_winner_json(
        winner_path, candidate=profile, mean_aji=0.7, per_variant={"PPL": 0.7}
    )
    loaded = load_grid_winner(winner_path)
    assert loaded == profile


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
    from yolo.profile_tune_work import uv_run_cmd

    yolo_cmd = uv_run_cmd(Path("/repo/src/yolo"), "yolo.predict", "--unit", "whole")
    assert yolo_cmd[:4] == ["uv", "run", "--directory", "/repo/src/yolo"]
    assert yolo_cmd[4:7] == ["python", "-m", "yolo.predict"]

    common_cmd = uv_run_cmd(
        Path("/repo/src/common"), "common.evaluate_instances", unbuffered=True
    )
    assert common_cmd[4:7] == ["python", "-u", "-m"]


def test_evaluate_variant_predictions_runs_write_eval_and_evaluate(
    tmp_path: Path,
) -> None:
    from yolo.profile_tune_work import evaluate_variant_predictions

    repo = tmp_path / "repo"
    common_src = repo / "src" / "common"
    common_src.mkdir(parents=True)
    variant_out = tmp_path / "variant_out"
    staged_manifest = tmp_path / "manifest.json"
    staged_manifest.write_text("{}", encoding="utf-8")
    invoked: list[str] = []

    def _fake_run(cmd: list[str]) -> None:
        invoked.append(cmd[cmd.index("-m") + 1])

    with patch("yolo.profile_tune_work.run_subprocess", side_effect=_fake_run):
        metrics_path = evaluate_variant_predictions(
            variant="PPL",
            variant_output_dir=variant_out,
            staged_manifest=staged_manifest,
            repo=repo,
        )

    assert metrics_path == variant_out / "instance_metrics.json"
    assert invoked == ["common.stage_manifest", "common.evaluate_instances"]



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


def test_run_grid_search_dry_run_picks_best_across_all_registry_variants(
    tmp_path: Path,
) -> None:
    from common.variants import all_variant_names
    variants = all_variant_names()
    _write_grid(tmp_path / "grid.yaml")
    spec = load_tune_grid(tmp_path / "grid.yaml")
    candidates = list(iter_grid_candidates(spec))[:2]
    better, worse = candidates[0], candidates[1]
    scores: dict[tuple[str, str], float] = {}
    for variant in variants:
        scores[(variant, better.candidate_id())] = 0.9
        scores[(variant, worse.candidate_id())] = 0.1
    winner, _ = run_grid_search(
        candidates=[better, worse],
        variants=variants,
        output_dir=tmp_path / "tune_all",
        score_variant=dry_run_scorer(scores),
    )
    assert winner == better


def test_run_grid_search_dry_run_writes_winner(tmp_path: Path) -> None:
    _write_grid(tmp_path / "grid.yaml")
    spec = load_tune_grid(tmp_path / "grid.yaml")
    candidates = list(iter_grid_candidates(spec))
    assert len(candidates) >= 2
    better, worse = candidates[0], candidates[1]
    scorer = dry_run_scorer(
        {
            ("PPL", better.candidate_id()): 0.9,
            ("PPL", worse.candidate_id()): 0.4,
        }
    )
    winner, _ = run_grid_search(
        candidates=[better, worse],
        variants=("PPL",),
        output_dir=tmp_path / "tune",
        score_variant=scorer,
    )
    assert winner == better
    assert (tmp_path / "tune" / "grid" / "winner.json").is_file()
    assert (tmp_path / "tune" / "grid" / "results.csv").is_file()


def test_should_skip_variant_eval_requires_matching_resume_fingerprint(
    tmp_path: Path,
) -> None:
    metrics = tmp_path / "instance_metrics.json"
    expected = {"candidate_id": "x", "conf": 0.25}
    assert not should_skip_variant_eval(
        metrics, resume=True, expected_fingerprint=expected
    )
    metrics.write_text("{}", encoding="utf-8")
    assert not should_skip_variant_eval(
        metrics, resume=True, expected_fingerprint=expected
    )
    write_variant_eval_resume_meta(metrics, expected)
    assert should_skip_variant_eval(metrics, resume=True, expected_fingerprint=expected)
    assert not should_skip_variant_eval(
        metrics, resume=True, expected_fingerprint={**expected, "conf": 0.99}
    )
    assert not should_skip_variant_eval(metrics, resume=False, expected_fingerprint=expected)


def test_run_grid_search_no_resume_does_not_delete_existing_csv_at_start(
    tmp_path: Path,
) -> None:
    grid_dir = tmp_path / "grid"
    grid_dir.mkdir(parents=True)
    csv_path = grid_dir / "results.csv"
    csv_path.write_text(
        "candidate_id,postprocess_type,match_metric,match_threshold,conf,mask_threshold,mean_aji,aji__PPL\n"
        "stale,GREEDYNMM,IOS,0.5,0.25,0.5,0.1,0.1\n",
        encoding="utf-8",
    )
    _write_grid(tmp_path / "grid.yaml")
    spec = load_tune_grid(tmp_path / "grid.yaml")
    candidate = list(iter_grid_candidates(spec))[0]

    def scorer(variant: str, cand: YoloInferenceProfileCandidate, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics = out_dir / "instance_metrics.json"
        metrics.write_text(
            json.dumps({"samples": [{"sample_id": "train", "aji": 0.55}]}),
            encoding="utf-8",
        )
        return metrics

    run_grid_search(
        candidates=[candidate],
        variants=("PPL",),
        output_dir=tmp_path,
        score_variant=scorer,
        resume=False,
        resume_context=None,
    )
    assert csv_path.is_file()
    loaded = load_grid_results_csv(csv_path)
    assert len(loaded) == 1
    assert loaded[0]["candidate_id"] == candidate.candidate_id()
    assert float(loaded[0]["mean_aji"]) == pytest.approx(0.55)


def test_append_grid_result_row_writes_incrementally(tmp_path: Path) -> None:
    csv_path = tmp_path / "grid" / "results.csv"
    row_a = {
        "candidate_id": "a",
        "postprocess_type": "GREEDYNMM",
        "match_metric": "IOS",
        "match_threshold": 0.5,
        "conf": 0.25,
        "mask_threshold": 0.5,
        "mean_aji": 0.7,
        "aji__PPL": 0.7,
    }
    append_grid_result_row(csv_path, row_a, variant_names=("PPL",))
    loaded = load_grid_results_csv(csv_path)
    assert len(loaded) == 1
    assert loaded[0]["candidate_id"] == "a"
    assert float(loaded[0]["mean_aji"]) == pytest.approx(0.7)
    row_b = {**row_a, "candidate_id": "b", "mean_aji": 0.8, "aji__PPL": 0.8}
    append_grid_result_row(csv_path, row_b, variant_names=("PPL",))
    assert len(load_grid_results_csv(csv_path)) == 2


def test_validate_detector_caches_raises_when_cache_missing(tmp_path: Path) -> None:
    _write_grid(tmp_path / "grid.yaml")
    spec = load_tune_grid(tmp_path / "grid.yaml")
    run_root = tmp_path / "grainseg" / "runs" / "yolo26-seg"
    weights = run_root / "PPL" / "weights" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"weights")
    with pytest.raises(FileNotFoundError, match="Missing or invalid"):
        validate_detector_caches(
            work_root=tmp_path / "_work",
            spec=spec,
            variants=("PPL",),
            grainseg_root=tmp_path / "grainseg",
            run_root=run_root,
        )


def test_run_grid_search_resume_skips_when_fingerprint_matches(tmp_path: Path) -> None:
    _write_grid(tmp_path / "grid.yaml")
    spec = load_tune_grid(tmp_path / "grid.yaml")
    candidates = list(iter_grid_candidates(spec))[:1]
    candidate = candidates[0]
    grainseg_root = tmp_path / "grainseg"
    run_root = grainseg_root / "runs" / "yolo26-seg"
    weights = run_root / "PPL" / "weights" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"weights")
    existing = tmp_path / "grid" / "candidates" / candidate.candidate_id() / "PPL"
    existing.mkdir(parents=True)
    metrics = existing / "instance_metrics.json"
    metrics.write_text(
        json.dumps({"samples": [{"sample_id": "train", "aji": 0.88}]}),
        encoding="utf-8",
    )
    resume_context = GridResumeContext(
        grainseg_root=grainseg_root,
        run_root=run_root,
        grid_config=tmp_path / "grid.yaml",
    )
    write_variant_eval_resume_meta(
        metrics,
        variant_eval_fingerprint(
            candidate=candidate, variant="PPL", context=resume_context
        ),
    )
    calls: list[tuple[str, str]] = []

    def scorer(variant: str, cand: YoloInferenceProfileCandidate, out_dir: Path) -> Path:
        calls.append((variant, cand.candidate_id()))
        raise AssertionError("scorer should not run when resume fingerprint matches")

    winner, _ = run_grid_search(
        candidates=[candidate],
        variants=("PPL",),
        output_dir=tmp_path,
        score_variant=scorer,
        resume=True,
        resume_context=resume_context,
    )
    assert winner == candidate
    assert calls == []
    assert metrics_resume_valid(
        metrics,
        expected=variant_eval_fingerprint(
            candidate=candidate, variant="PPL", context=resume_context
        ),
    )


def test_recompute_winner_from_csv_picks_highest_mean_aji(tmp_path: Path) -> None:
    from yolo.inference_profile_tune import load_grid_winner, recompute_winner_from_csv

    grid_dir = tmp_path / "grid"
    grid_dir.mkdir(parents=True)
    (grid_dir / "results.csv").write_text(
        "candidate_id,postprocess_type,match_metric,match_threshold,conf,mask_threshold,mean_aji,aji__PPL\n"
        "low,GREEDYNMM,IOS,0.5,0.25,0.5,0.4,0.4\n"
        "high,NMM,IOU,0.6,0.35,0.6,0.9,0.9\n",
        encoding="utf-8",
    )
    winner = recompute_winner_from_csv(tmp_path, variant_names=("PPL",))
    assert winner.postprocess_type == "NMM"
    assert winner.conf == pytest.approx(0.35)
    loaded = load_grid_winner(grid_dir / "winner.json")
    assert loaded == winner


def test_profile_tune_finalize_cli_recompute_winner_from_csv(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from yolo.profile_tune_finalize import main

    grid_dir = tmp_path / "grid"
    grid_dir.mkdir(parents=True)
    (grid_dir / "results.csv").write_text(
        "candidate_id,postprocess_type,match_metric,match_threshold,conf,mask_threshold,mean_aji,aji__PPL\n"
        "winner,GREEDYNMM,IOS,0.5,0.2,0.45,0.95,0.95\n",
        encoding="utf-8",
    )
    main(
        [
            "--output-dir",
            str(tmp_path),
            "--recompute-winner-from-csv",
            "--variants",
            "PPL",
        ]
    )
    captured = json.loads(capsys.readouterr().out.strip())
    assert captured["conf"] == pytest.approx(0.2)
    assert (grid_dir / "winner.json").is_file()


def test_grid_coordinator_cache_miss_fails_validation(tmp_path: Path) -> None:
    _write_grid(tmp_path / "grid.yaml")
    spec = load_tune_grid(tmp_path / "grid.yaml")
    run_root = tmp_path / "grainseg" / "runs" / "yolo26-seg"
    weights = run_root / "PPL" / "weights" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"weights")
    with pytest.raises(FileNotFoundError, match="Missing or invalid detector caches"):
        validate_detector_caches(
            work_root=tmp_path / "_work",
            spec=spec,
            variants=("PPL",),
            grainseg_root=tmp_path / "grainseg",
            run_root=run_root,
        )


def test_in_process_score_from_tiled_cache_matches_evaluate_instances(
    tmp_path: Path,
) -> None:
    """Disk tiled-proposal cache + GT cache → in-process AJI matches evaluate_instances."""
    from common.test_inference import load_test_inference_recipe
    from yolo.profile_tune_candidate import score_variant_train_aji_from_cache
    from yolo.profile_tune_gt_cache import (
        build_gt_fingerprint,
        gt_cache_dir,
        write_gt_instance_map_cache,
    )
    from yolo.tests.profile_tune_fixtures import disjoint_sahi_proposals, tiny_train_gt_map
    from yolo.tests.test_profile_tune_scoring import _train_aji_via_evaluate_instances
    from yolo.tiled_proposal_cache import (
        proposal_cache_dir,
        proposal_cache_record,
        recipe_whole_window_fingerprint,
        tiled_proposal_records_from_sahi_predictions,
        weights_sha256,
        write_tiled_proposals,
    )

    height, width = 16, 16
    candidate = YoloInferenceProfileCandidate(
        postprocess_type="GREEDYNMM",
        match_metric="IOS",
        match_threshold=0.4,
        conf=0.2,
        mask_threshold=0.45,
    )
    grainseg_root = tmp_path / "grainseg"
    run_root = grainseg_root / "runs" / "yolo26-seg"
    weights = run_root / "PPL" / "weights" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"weights")
    work_root = tmp_path / "_work"
    recipe = load_test_inference_recipe()
    proposals = disjoint_sahi_proposals(height, width)
    v2_records = tiled_proposal_records_from_sahi_predictions(
        proposals,
        height=height,
        width=width,
        mask_threshold=candidate.mask_threshold,
    )
    write_tiled_proposals(
        proposal_cache_dir(work_root / "PPL", conf=candidate.conf, mask_threshold=candidate.mask_threshold),
        v2_records,
        proposal_cache_record(
            variant="PPL",
            weights_sha256=weights_sha256(weights),
            recipe_window_fingerprint=recipe_whole_window_fingerprint(recipe),
            conf=candidate.conf,
            mask_threshold=candidate.mask_threshold,
            sample_id="train",
            height=height,
            width=width,
        ),
    )
    gt_map = tiny_train_gt_map(height, width)
    labels_gpkg = grainseg_root / "dataset" / "train" / "train_labels.gpkg"
    labels_gpkg.parent.mkdir(parents=True)
    labels_gpkg.write_bytes(b"labels")
    write_gt_instance_map_cache(
        gt_cache_dir(work_root, "PPL"),
        gt_map,
        fingerprint=build_gt_fingerprint(
            variant="PPL", sample_id="train", labels_gpkg=labels_gpkg
        ),
    )
    image_path = tmp_path / "train.tif"
    image_path.write_bytes(b"\x00")
    pred_path = tmp_path / "prediction_sets" / "train.json"

    fast_aji = score_variant_train_aji_from_cache(
        variant="PPL",
        candidate=candidate,
        grainseg_root=grainseg_root,
        run_root=run_root,
        work_root=work_root,
    )
    canonical_aji = _train_aji_via_evaluate_instances(
        gt_map,
        proposals,
        candidate=candidate,
        height=height,
        width=width,
        variant="PPL",
        image_path=image_path,
        prediction_set_path=pred_path,
    )
    assert fast_aji == pytest.approx(canonical_aji, rel=0.0, abs=1e-9)


def test_candidate_scoring_cache_fingerprint_mismatch(tmp_path: Path) -> None:
    from common.test_inference import load_test_inference_recipe
    from yolo.profile_tune_candidate import score_variant_train_aji_from_cache
    import numpy as np

    from yolo.tiled_proposal_cache import (
        proposal_cache_dir,
        proposal_cache_record,
        recipe_whole_window_fingerprint,
        tiled_proposal_record_from_binary_mask,
        weights_sha256,
        write_tiled_proposals,
    )

    grainseg_root = tmp_path / "grainseg"
    run_root = grainseg_root / "runs" / "yolo26-seg"
    weights = run_root / "PPL" / "weights" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"weights")
    work_root = tmp_path / "_work"
    recipe = load_test_inference_recipe()
    height, width = 16, 16
    mask = np.zeros((height, width), dtype=bool)
    mask[0:4, 0:4] = True
    write_tiled_proposals(
        proposal_cache_dir(work_root / "PPL", conf=0.2, mask_threshold=0.45),
        [tiled_proposal_record_from_binary_mask(mask, score=0.5)],
        proposal_cache_record(
            variant="PPL",
            weights_sha256=weights_sha256(weights),
            recipe_window_fingerprint=recipe_whole_window_fingerprint(recipe),
            conf=0.2,
            mask_threshold=0.45,
            sample_id="train",
            height=height,
            width=width,
        ),
    )
    candidate = YoloInferenceProfileCandidate(
        postprocess_type="GREEDYNMM",
        match_metric="IOS",
        match_threshold=0.5,
        conf=0.99,
        mask_threshold=0.45,
    )
    with pytest.raises(FileNotFoundError):
        score_variant_train_aji_from_cache(
            variant="PPL",
            candidate=candidate,
            grainseg_root=grainseg_root,
            run_root=run_root,
            work_root=work_root,
        )
