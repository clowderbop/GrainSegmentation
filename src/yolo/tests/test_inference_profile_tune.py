"""Tests for YOLO inference profile train selection (ADR 0005)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import yaml

from common.test_inference import (
    YoloInferenceProfileCandidate,
    load_test_inference_recipe,
    profile_tune_candidate_from_conf,
    profile_tune_fixed_mask_threshold,
)
from common.variants import repo_root

from common.instance_metric_bundle import INSTANCE_METRIC_BUNDLE_KEYS
from common.merged_view_pq import MERGED_VIEW_PQ_RESULT_KEYS
from yolo.inference_profile_tune import (
    GridResumeContext,
    PROFILE_SELECTION_OBJECTIVE,
    append_grid_result_row,
    count_detector_jobs,
    detector_keys_per_variant,
    count_grid_candidates,
    detector_job_at_index,
    variant_at_detector_array_index,
    extract_mean_pq_from_report,
    grid_result_row_from_candidate_scoring,
    grid_results_fieldnames,
    iter_detector_jobs,
    iter_grid_candidates,
    load_grid_results_csv,
    load_grid_winner,
    load_tune_grid,
    mean_pq_across_variants,
    metrics_resume_valid,
    promote_profile_to_recipe,
    run_grid_search,
    score_candidate_across_variants,
    select_best_candidate,
    should_skip_variant_eval,
    tune_grid_path,
    variant_eval_fingerprint,
    variant_metric_column,
    write_grid_winner_json,
    write_variant_eval_resume_meta,
)
from yolo.tests.profile_tune_fixtures import (
    constant_merged_view_pq_result,
    constant_metric_bundle,
    instance_metrics_report_for_pq,
)
from yolo.profile_tune_dry_run import dry_run_scorer
from yolo.profile_tune_cli import validate_detector_caches


def _write_grid(path: Path, *, conf: list[float] | None = None) -> None:
    path.write_text(
        yaml.safe_dump({"grid": {"conf": conf or [0.2, 0.3]}}),
        encoding="utf-8",
    )


def _write_legacy_grid(path: Path) -> None:
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


def test_load_tune_grid_reads_conf_only_search_space(tmp_path: Path) -> None:
    grid_path = tmp_path / "grid.yaml"
    _write_grid(grid_path, conf=[0.2, 0.3])
    spec = load_tune_grid(grid_path)
    assert spec.grid.conf == (0.2, 0.3)


def test_load_tune_grid_rejects_multi_value_mask_threshold(tmp_path: Path) -> None:
    grid_path = tmp_path / "grid.yaml"
    grid_path.write_text(
        yaml.safe_dump({"grid": {"conf": [0.2], "mask_threshold": [0.4, 0.5, 0.6]}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="mask_threshold must not list multiple"):
        load_tune_grid(grid_path)


def test_load_tune_grid_rejects_obsolete_sahi_axes(tmp_path: Path) -> None:
    grid_path = tmp_path / "grid.yaml"
    _write_legacy_grid(grid_path)
    with pytest.raises(ValueError, match="no longer a profile-selection axis"):
        load_tune_grid(grid_path)


def test_count_detector_jobs_is_variant_count(tmp_path: Path) -> None:
    _write_grid(tmp_path / "grid.yaml")
    spec = load_tune_grid(tmp_path / "grid.yaml")
    variants = ("PPL", "PPL+AllPPX")
    assert count_detector_jobs(spec, len(variants)) == len(variants)


def test_iter_detector_jobs_lists_flat_detector_grid(tmp_path: Path) -> None:
    _write_grid(tmp_path / "grid.yaml")
    spec = load_tune_grid(tmp_path / "grid.yaml")
    variants = ("PPL", "PPL+AllPPX")
    jobs = list(iter_detector_jobs(spec, variants))
    assert len(jobs) == len(variants) * detector_keys_per_variant(spec)


def test_variant_at_detector_array_index_selects_variant(tmp_path: Path) -> None:
    variants = ("PPL", "PPL+AllPPX")
    assert variant_at_detector_array_index(variants, 1) == "PPL"
    assert variant_at_detector_array_index(variants, 2) == "PPL+AllPPX"


def test_detector_job_at_index_matches_iter_detector_jobs_order(tmp_path: Path) -> None:
    _write_grid(tmp_path / "grid.yaml")
    spec = load_tune_grid(tmp_path / "grid.yaml")
    variants = ("PPL", "PPL+AllPPX")
    jobs = list(iter_detector_jobs(spec, variants))
    assert detector_job_at_index(spec, variants, 1) == jobs[0]
    assert detector_job_at_index(spec, variants, len(jobs)) == jobs[-1]


def test_detector_job_at_index_rejects_out_of_range(tmp_path: Path) -> None:
    _write_grid(tmp_path / "grid.yaml")
    spec = load_tune_grid(tmp_path / "grid.yaml")
    variants = ("PPL",)
    with pytest.raises(ValueError, match="must be >= 1"):
        detector_job_at_index(spec, variants, 0)
    with pytest.raises(ValueError, match="out of range"):
        detector_job_at_index(spec, variants, 999)


def test_iter_grid_candidates_one_per_conf(tmp_path: Path) -> None:
    _write_grid(tmp_path / "grid.yaml", conf=[0.2, 0.3])
    spec = load_tune_grid(tmp_path / "grid.yaml")
    candidates = list(iter_grid_candidates(spec))
    assert len(candidates) == count_grid_candidates(spec)
    assert candidates[0] == profile_tune_candidate_from_conf(0.2)
    assert len({c.candidate_id() for c in candidates}) == len(candidates)


def test_mean_pq_across_variants_equal_weights_per_variant() -> None:
    scores = {"PPL": 0.8, "PPL+AllPPX": 0.6}
    assert mean_pq_across_variants(scores) == pytest.approx(0.7)


def test_grid_results_fieldnames_use_merged_view_pq_only() -> None:
    fieldnames = grid_results_fieldnames(("PPL",))
    assert "mean_pq" in fieldnames
    assert "mean_tp" in fieldnames
    assert "aji_plus__PPL" not in fieldnames
    assert "f1_iou75__PPL" not in fieldnames
    assert "mF1_iou50_95__PPL" not in fieldnames
    for key in MERGED_VIEW_PQ_RESULT_KEYS:
        assert f"mean_{key}" in fieldnames
        assert variant_metric_column(key, "PPL") in fieldnames


def test_extract_mean_pq_from_single_sample_report() -> None:
    report = instance_metrics_report_for_pq(0.42)
    assert extract_mean_pq_from_report(report) == pytest.approx(0.42)


def test_extract_mean_pq_prefers_report_mean_when_present() -> None:
    report = {
        "samples": [
            {"sample_id": "a", **constant_metric_bundle(0.1)},
            {"sample_id": "b", **constant_metric_bundle(0.9)},
        ],
        "mean": constant_metric_bundle(0.5),
    }
    assert extract_mean_pq_from_report(report) == pytest.approx(0.5)


def test_extract_mean_pq_averages_samples_when_mean_absent() -> None:
    report = {
        "samples": [
            {"sample_id": "a", **constant_metric_bundle(0.2)},
            {"sample_id": "b", **constant_metric_bundle(0.8)},
        ],
    }
    assert extract_mean_pq_from_report(report) == pytest.approx(0.5)


def test_select_best_candidate_maximizes_mean_train_pq() -> None:
    rows = [
        {"candidate_id": "a", "mean_pq": 0.71},
        {"candidate_id": "b", "mean_pq": 0.82},
        {"candidate_id": "c", "mean_pq": 0.80},
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
    profile = profile_tune_candidate_from_conf(0.2)
    promote_profile_to_recipe(profile, recipe_path)
    updated_text = recipe_path.read_text(encoding="utf-8")
    assert "# Shared test inference recipe (ADR 0003)." in updated_text
    assert "# YOLO inference profile (ADR 0005)." in updated_text
    assert "  conf: 0.2" in updated_text
    assert "  postprocess_type: GREEDYNMM" in updated_text
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
    profile = profile_tune_candidate_from_conf(0.2)
    promote_profile_to_recipe(profile, recipe_path)
    updated = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    assert updated["yolo"]["conf"] == 0.2
    assert updated["yolo"]["mask_threshold"] == profile_tune_fixed_mask_threshold()
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
    profile = profile_tune_candidate_from_conf(0.2)
    promote_profile_to_recipe(profile, recipe_path)
    loaded = load_test_inference_recipe(recipe_path)
    assert loaded.yolo.conf == profile.conf
    assert loaded.yolo.profile.mask_threshold == profile_tune_fixed_mask_threshold()


def test_promote_profile_rejects_missing_conf_key_and_preserves_recipe(tmp_path: Path) -> None:
    recipe_path = tmp_path / "test_inference.yaml"
    original = """whole:
  window: 1024
  stride: 512
patch:
  imgsz: 1024
yolo:
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
    profile = profile_tune_candidate_from_conf(0.25)
    with pytest.raises(ValueError, match="missing yolo profile keys"):
        promote_profile_to_recipe(profile, recipe_path)
    assert recipe_path.read_text(encoding="utf-8") == original


def test_tune_grid_path_defaults_to_config_yolo_inference_profile_tune_yaml() -> None:
    assert tune_grid_path() == repo_root() / "config" / "yolo_inference_profile_tune.yaml"


def test_committed_tune_grid_is_conf_only_seven_candidates() -> None:
    """Default config/yolo_inference_profile_tune.yaml is conf-only (~7 candidates)."""
    spec = load_tune_grid(tune_grid_path())
    assert len(spec.grid.conf) == 7
    assert count_grid_candidates(spec) == 7

    candidates = list(iter_grid_candidates(spec))
    assert len(candidates) == 7
    assert len({c.candidate_id() for c in candidates}) == 7
    fixed_mask = profile_tune_fixed_mask_threshold()
    assert all(c.mask_threshold == fixed_mask for c in candidates)

    variants = ("PPL", "PPL+AllPPX")
    assert count_detector_jobs(spec, len(variants)) == len(variants)
    assert len(list(iter_detector_jobs(spec, variants))) == (
        len(variants) * detector_keys_per_variant(spec)
    )


def test_write_grid_winner_json_persists_per_variant_merged_view_pq_results(
    tmp_path: Path,
) -> None:
    profile = profile_tune_candidate_from_conf(0.25)
    ppl_result = constant_merged_view_pq_result(0.7)
    ppx_result = constant_merged_view_pq_result(0.5)
    winner_path = tmp_path / "grid" / "winner.json"
    write_grid_winner_json(
        winner_path,
        candidate=profile,
        mean_pq=0.6,
        per_variant_pq_results={"PPL": ppl_result, "PPL+AllPPX": ppx_result},
    )
    payload = json.loads(winner_path.read_text(encoding="utf-8"))
    assert "per_variant_pq" not in payload
    assert set(payload["per_variant_pq_results"].keys()) == {"PPL", "PPL+AllPPX"}
    for variant, expected in (
        ("PPL", ppl_result),
        ("PPL+AllPPX", ppx_result),
    ):
        stored = payload["per_variant_pq_results"][variant]
        assert tuple(stored.keys()) == MERGED_VIEW_PQ_RESULT_KEYS
        for key in MERGED_VIEW_PQ_RESULT_KEYS:
            assert stored[key] == expected[key]


def test_load_grid_winner_round_trips_profile(tmp_path: Path) -> None:
    profile = profile_tune_candidate_from_conf(0.25)
    winner_path = tmp_path / "grid" / "winner.json"
    write_grid_winner_json(
        winner_path,
        candidate=profile,
        mean_pq=0.7,
        per_variant_pq_results={"PPL": constant_merged_view_pq_result(0.7)},
    )
    payload = json.loads(winner_path.read_text(encoding="utf-8"))
    assert payload["selection_objective"] == PROFILE_SELECTION_OBJECTIVE
    assert payload["mean_pq"] == pytest.approx(0.7)
    assert payload["conf"] == pytest.approx(0.25)
    assert payload["fixed_mask_threshold"] == profile_tune_fixed_mask_threshold()
    assert "postprocess_type" in payload["removed_grid_axes"]
    loaded = load_grid_winner(winner_path)
    assert loaded == profile


def test_score_candidate_across_variants_reads_variant_reports(tmp_path: Path) -> None:
    def _write_report(path: Path, pq: float) -> None:
        path.write_text(
            json.dumps(instance_metrics_report_for_pq(pq)),
            encoding="utf-8",
        )

    ppl_report = tmp_path / "ppl.json"
    ppx_report = tmp_path / "ppx.json"
    _write_report(ppl_report, 0.8)
    _write_report(ppx_report, 0.6)
    mean_pq, per_variant_bundles = score_candidate_across_variants(
        {"PPL": ppl_report, "PPL+AllPPX": ppx_report}
    )
    assert mean_pq == pytest.approx(0.7)
    assert per_variant_bundles["PPL"]["pq"] == pytest.approx(0.8)
    assert per_variant_bundles["PPL+AllPPX"]["pq"] == pytest.approx(0.6)
    for key in INSTANCE_METRIC_BUNDLE_KEYS:
        assert key in per_variant_bundles["PPL"]


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



def test_extract_mean_pq_from_evaluate_instances_report_shape() -> None:
    from common.reporting import build_instance_eval_report

    report = build_instance_eval_report(
        model_type="yolo",
        variant="PPL",
        unit="whole",
        samples=[
            {
                "sample_id": "train",
                "pq": 0.42,
                "dq": 0.5,
                "sq": 0.84,
                "f1_iou50": 0.5,
                "gt_instance_count": 1,
                "pred_instance_count": 1,
                "empty_gt": False,
            }
        ],
    )
    assert extract_mean_pq_from_report(report) == pytest.approx(0.42)


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
        "candidate_id,conf,mask_threshold,mean_pq,pq__PPL\n"
        "stale,0.25,0.4,0.1,0.1\n",
        encoding="utf-8",
    )
    _write_grid(tmp_path / "grid.yaml")
    spec = load_tune_grid(tmp_path / "grid.yaml")
    candidate = list(iter_grid_candidates(spec))[0]

    def scorer(variant: str, cand: YoloInferenceProfileCandidate, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics = out_dir / "instance_metrics.json"
        metrics.write_text(
            json.dumps(instance_metrics_report_for_pq(0.55)),
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
    assert float(loaded[0]["mean_pq"]) == pytest.approx(0.55)


def test_append_grid_result_row_writes_incrementally(tmp_path: Path) -> None:
    csv_path = tmp_path / "grid" / "results.csv"
    profile = profile_tune_candidate_from_conf(0.25)
    row_a = grid_result_row_from_candidate_scoring(
        candidate=profile,
        per_variant_pq_results={"PPL": constant_merged_view_pq_result(0.7)},
    )
    row_a["candidate_id"] = "a"
    append_grid_result_row(csv_path, row_a, variant_names=("PPL",))
    loaded = load_grid_results_csv(csv_path)
    assert len(loaded) == 1
    assert loaded[0]["candidate_id"] == "a"
    assert float(loaded[0]["mean_pq"]) == pytest.approx(0.7)
    row_b = {
        **row_a,
        "candidate_id": "b",
        "mean_pq": 0.8,
        variant_metric_column("pq", "PPL"): 0.8,
    }
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
            work_root=tmp_path / ".cache",
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
        json.dumps(instance_metrics_report_for_pq(0.88)),
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


def test_finalize_grid_winner_writes_per_variant_pq_results_from_csv(tmp_path: Path) -> None:
    from yolo.inference_profile_tune import finalize_grid_winner, load_grid_winner

    grid_dir = tmp_path / "grid"
    grid_dir.mkdir(parents=True)
    row = grid_result_row_from_candidate_scoring(
        candidate=profile_tune_candidate_from_conf(0.25),
        per_variant_pq_results={
            "PPL": constant_merged_view_pq_result(0.82),
            "PPL+AllPPX": constant_merged_view_pq_result(0.58),
        },
    )
    winner = finalize_grid_winner(grid_dir, [row], variant_names=("PPL", "PPL+AllPPX"))
    assert winner.conf == pytest.approx(0.25)
    payload = json.loads((grid_dir / "winner.json").read_text(encoding="utf-8"))
    assert payload["mean_pq"] == pytest.approx(mean_pq_across_variants({"PPL": 0.82, "PPL+AllPPX": 0.58}))
    assert payload["per_variant_pq_results"]["PPL"]["pq"] == pytest.approx(0.82)
    assert load_grid_winner(grid_dir / "winner.json") == winner


def test_recompute_winner_from_csv_picks_highest_mean_pq(tmp_path: Path) -> None:
    from yolo.inference_profile_tune import load_grid_winner, recompute_winner_from_csv

    grid_dir = tmp_path / "grid"
    grid_dir.mkdir(parents=True)
    fieldnames = grid_results_fieldnames(("PPL",))
    mean_low = ",".join("0.4" for _ in MERGED_VIEW_PQ_RESULT_KEYS)
    per_variant_low = ",".join("0.4" for _ in MERGED_VIEW_PQ_RESULT_KEYS)
    mean_high = ",".join("0.9" for _ in MERGED_VIEW_PQ_RESULT_KEYS)
    per_variant_high = ",".join("0.9" for _ in MERGED_VIEW_PQ_RESULT_KEYS)
    (grid_dir / "results.csv").write_text(
        ",".join(fieldnames) + "\n"
        f"low,0.25,0.4,{mean_low},{per_variant_low}\n"
        f"high,0.35,0.4,{mean_high},{per_variant_high}\n",
        encoding="utf-8",
    )
    winner = recompute_winner_from_csv(tmp_path, variant_names=("PPL",))
    assert winner.conf == pytest.approx(0.35)
    loaded = load_grid_winner(grid_dir / "winner.json")
    assert loaded == winner


def test_profile_tune_finalize_cli_recompute_winner_from_csv(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from yolo.profile_tune_finalize import main

    grid_dir = tmp_path / "grid"
    grid_dir.mkdir(parents=True)
    fieldnames = grid_results_fieldnames(("PPL",))
    mean_values = ",".join("0.95" for _ in MERGED_VIEW_PQ_RESULT_KEYS)
    per_variant_values = ",".join("0.95" for _ in MERGED_VIEW_PQ_RESULT_KEYS)
    (grid_dir / "results.csv").write_text(
        ",".join(fieldnames) + "\n"
        f"winner,0.2,0.4,{mean_values},{per_variant_values}\n",
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
            work_root=tmp_path / ".cache",
            spec=spec,
            variants=("PPL",),
            grainseg_root=tmp_path / "grainseg",
            run_root=run_root,
        )


def test_in_process_score_from_tiled_cache_matches_evaluate_instances(
    tmp_path: Path,
) -> None:
    """Disk tiled-proposal cache + GT cache → in-process PQ matches evaluate_instances."""
    from common.test_inference import load_test_inference_recipe
    from common.profile_tune_gt_cache import (
        build_gt_fingerprint,
        gt_cache_dir,
        write_gt_instance_map_cache,
    )
    from yolo.profile_tune_candidate import score_variant_train_metrics_from_cache
    from yolo.tests.profile_tune_fixtures import (
        tiny_train_gt_map,
        tiled_proposal_records_disjoint_via_collector,
    )
    from yolo.tests.test_profile_tune_scoring import _train_pq_via_evaluate_instances
    from yolo.tiled_proposal_cache import (
        detector_cache_expected_record,
        proposal_cache_dir,
        proposal_cache_record,
        recipe_whole_window_fingerprint,
        weights_sha256,
        write_tiled_proposals,
    )

    height, width = 16, 16
    candidate = profile_tune_candidate_from_conf(0.2)
    fixed_mask = profile_tune_fixed_mask_threshold()
    grainseg_root = tmp_path / "grainseg"
    run_root = grainseg_root / "runs" / "yolo26-seg"
    weights = run_root / "PPL" / "weights" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"weights")
    work_root = tmp_path / ".cache"
    recipe = load_test_inference_recipe()
    tiled_records = tiled_proposal_records_disjoint_via_collector(
        height, width, mask_threshold=fixed_mask
    )
    write_tiled_proposals(
        proposal_cache_dir(work_root / "PPL", conf=candidate.conf),
        tiled_records,
        proposal_cache_record(
            variant="PPL",
            weights_sha256=weights_sha256(weights),
            recipe_window_fingerprint=recipe_whole_window_fingerprint(recipe),
            conf=candidate.conf,
            mask_threshold=fixed_mask,
            sample_id="train",
            height=height,
            width=width,
        ),
    )
    gt_map = tiny_train_gt_map(height, width)
    labels_gpkg = grainseg_root / "dataset" / "train" / "train_labels.gpkg"
    labels_gpkg.parent.mkdir(parents=True)
    labels_gpkg.write_bytes(b"labels")
    anchor = grainseg_root / "dataset" / "train" / "train_PPL.tif"
    import tifffile

    tifffile.imwrite(anchor, np.zeros((height, width, 3), dtype=np.uint8))
    write_gt_instance_map_cache(
        gt_cache_dir(work_root),
        gt_map,
        fingerprint=build_gt_fingerprint(
            sample_id="train",
            labels_gpkg=labels_gpkg,
            width=width,
            height=height,
        ),
    )
    image_path = tmp_path / "train.tif"
    image_path.write_bytes(b"\x00")
    pred_path = tmp_path / "prediction_sets" / "train.json"

    fast_bundle = score_variant_train_metrics_from_cache(
        variant="PPL",
        candidate=candidate,
        grainseg_root=grainseg_root,
        run_root=run_root,
        work_root=work_root,
        gt_map=gt_map,
    )
    canonical_pq = _train_pq_via_evaluate_instances(
        gt_map,
        tiled_records,
        candidate=candidate,
        height=height,
        width=width,
        variant="PPL",
        image_path=image_path,
        prediction_set_path=pred_path,
    )
    assert fast_bundle["pq"] == pytest.approx(canonical_pq, rel=0.0, abs=1e-9)


def test_build_train_gt_cache_loads_in_candidate_scoring(tmp_path: Path) -> None:
    """Micro GPKG fixture: CLI cache build → shared GT load in scoring path."""
    import os
    import shutil
    import subprocess
    import tifffile

    from common.profile_tune_gt_cache import (
        build_gt_fingerprint,
        gt_cache_dir,
        load_gt_instance_map_cache,
        train_labels_gpkg_path,
    )
    from yolo.profile_tune_candidate import score_variant_train_metrics_from_cache
    from yolo.tiled_proposal_cache import (
        proposal_cache_dir,
        proposal_cache_record,
        recipe_whole_window_fingerprint,
        tiled_proposal_record_from_binary_mask,
        weights_sha256,
        write_tiled_proposals,
    )

    fixtures = (
        Path(__file__).resolve().parents[2]
        / "common"
        / "tests"
        / "fixtures"
        / "gpkg_merged_instance_map"
    )
    height, width = 48, 64
    grainseg_root = tmp_path / "GrainSeg"
    labels_gpkg = grainseg_root / "dataset" / "train" / "train_labels.gpkg"
    labels_gpkg.parent.mkdir(parents=True)
    shutil.copy2(fixtures / "micro_labels.gpkg", labels_gpkg)
    anchor = grainseg_root / "dataset" / "train" / "train_PPL.tif"
    tifffile.imwrite(anchor, np.zeros((height, width, 3), dtype=np.uint8))

    output_dir = tmp_path / "run"
    work_root = output_dir / ".cache"
    common_src = Path(__file__).resolve().parents[2] / "common"
    tmpdir = tmp_path / "tmpdir"
    tmpdir.mkdir(parents=True)
    env = os.environ.copy()
    env["TMPDIR"] = str(tmpdir)
    subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-u",
            "-m",
            "common.profile_tune_gt_cache",
            "--output-dir",
            str(output_dir),
            "--grainseg-root",
            str(grainseg_root),
        ],
        cwd=common_src,
        check=True,
        env=env,
    )

    candidate = profile_tune_candidate_from_conf(0.2)
    fixed_mask = profile_tune_fixed_mask_threshold()
    run_root = grainseg_root / "runs" / "yolo26-seg"
    weights = run_root / "PPL" / "weights" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"weights")
    recipe = load_test_inference_recipe()
    mask = np.zeros((height, width), dtype=bool)
    mask[10:20, 10:20] = True
    write_tiled_proposals(
        proposal_cache_dir(work_root / "PPL", conf=candidate.conf),
        [tiled_proposal_record_from_binary_mask(mask, score=0.5)],
        proposal_cache_record(
            variant="PPL",
            weights_sha256=weights_sha256(weights),
            recipe_window_fingerprint=recipe_whole_window_fingerprint(recipe),
            conf=candidate.conf,
            mask_threshold=fixed_mask,
            sample_id="train",
            height=height,
            width=width,
        ),
    )

    labels = train_labels_gpkg_path(grainseg_root)
    gt_map, _ = load_gt_instance_map_cache(
        gt_cache_dir(work_root),
        expected=build_gt_fingerprint(
            sample_id="train",
            labels_gpkg=labels,
            width=width,
            height=height,
        ),
    )
    bundle = score_variant_train_metrics_from_cache(
        variant="PPL",
        candidate=candidate,
        grainseg_root=grainseg_root,
        run_root=run_root,
        work_root=work_root,
        gt_map=gt_map,
    )
    assert 0.0 <= bundle["pq"] <= 1.0


def test_on_disk_gt_and_v2_proposal_caches_score_within_sub_minute(
    tmp_path: Path,
) -> None:
    """ADR 0005: load both caches from disk and score in sub-minute walltime."""
    import os
    import shutil
    import subprocess
    import time
    import tifffile

    from yolo.profile_tune_candidate import (
        load_shared_train_gt_map,
        score_variant_train_metrics_from_cache,
    )
    from yolo.tiled_proposal_cache import (
        proposal_cache_dir,
        proposal_cache_record,
        recipe_whole_window_fingerprint,
        tiled_proposal_record_from_binary_mask,
        weights_sha256,
        write_tiled_proposals,
    )

    fixtures = (
        Path(__file__).resolve().parents[2]
        / "common"
        / "tests"
        / "fixtures"
        / "gpkg_merged_instance_map"
    )
    height, width = 48, 64
    grainseg_root = tmp_path / "GrainSeg"
    labels_gpkg = grainseg_root / "dataset" / "train" / "train_labels.gpkg"
    labels_gpkg.parent.mkdir(parents=True)
    shutil.copy2(fixtures / "micro_labels.gpkg", labels_gpkg)
    anchor = grainseg_root / "dataset" / "train" / "train_PPL.tif"
    tifffile.imwrite(anchor, np.zeros((height, width, 3), dtype=np.uint8))

    output_dir = tmp_path / "run"
    work_root = output_dir / ".cache"
    common_src = Path(__file__).resolve().parents[2] / "common"
    tmpdir = tmp_path / "tmpdir"
    tmpdir.mkdir(parents=True)
    env = os.environ.copy()
    env["TMPDIR"] = str(tmpdir)
    subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-u",
            "-m",
            "common.profile_tune_gt_cache",
            "--output-dir",
            str(output_dir),
            "--grainseg-root",
            str(grainseg_root),
        ],
        cwd=common_src,
        check=True,
        env=env,
    )

    candidate = profile_tune_candidate_from_conf(0.2)
    fixed_mask = profile_tune_fixed_mask_threshold()
    run_root = grainseg_root / "runs" / "yolo26-seg"
    weights = run_root / "PPL" / "weights" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"weights")
    recipe = load_test_inference_recipe()
    mask = np.zeros((height, width), dtype=bool)
    mask[10:20, 10:20] = True
    write_tiled_proposals(
        proposal_cache_dir(work_root / "PPL", conf=candidate.conf),
        [tiled_proposal_record_from_binary_mask(mask, score=0.5)],
        proposal_cache_record(
            variant="PPL",
            weights_sha256=weights_sha256(weights),
            recipe_window_fingerprint=recipe_whole_window_fingerprint(recipe),
            conf=candidate.conf,
            mask_threshold=fixed_mask,
            sample_id="train",
            height=height,
            width=width,
        ),
    )

    t0 = time.perf_counter()
    gt_map = load_shared_train_gt_map(work_root=work_root, grainseg_root=grainseg_root)
    bundle = score_variant_train_metrics_from_cache(
        variant="PPL",
        candidate=candidate,
        grainseg_root=grainseg_root,
        run_root=run_root,
        work_root=work_root,
        gt_map=gt_map,
    )
    elapsed = time.perf_counter() - t0
    assert elapsed < 60.0, (
        f"on-disk GT + v2 proposal cache scoring took {elapsed:.1f}s "
        "(expected sub-minute, not hours)"
    )
    assert 0.0 <= bundle["pq"] <= 1.0


def test_candidate_scoring_cache_fingerprint_mismatch(tmp_path: Path) -> None:
    from common.test_inference import load_test_inference_recipe
    from yolo.profile_tune_candidate import score_variant_train_metrics_from_cache
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
    work_root = tmp_path / ".cache"
    recipe = load_test_inference_recipe()
    height, width = 16, 16
    mask = np.zeros((height, width), dtype=bool)
    mask[0:4, 0:4] = True
    fixed_mask = profile_tune_fixed_mask_threshold()
    write_tiled_proposals(
        proposal_cache_dir(work_root / "PPL", conf=0.2),
        [tiled_proposal_record_from_binary_mask(mask, score=0.5)],
        proposal_cache_record(
            variant="PPL",
            weights_sha256=weights_sha256(weights),
            recipe_window_fingerprint=recipe_whole_window_fingerprint(recipe),
            conf=0.2,
            mask_threshold=fixed_mask,
            sample_id="train",
            height=height,
            width=width,
        ),
    )
    candidate = profile_tune_candidate_from_conf(0.99)
    with pytest.raises(FileNotFoundError):
        score_variant_train_metrics_from_cache(
            variant="PPL",
            candidate=candidate,
            grainseg_root=grainseg_root,
            run_root=run_root,
            work_root=work_root,
            gt_map=np.zeros((height, width), dtype=np.int32),
        )
