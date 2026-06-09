"""Tests for YOLO inference profile train selection (ADR 0005)."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest
import yaml

from common.test_inference import (
    load_test_inference_recipe,
    profile_tune_candidate_from_conf,
    profile_tune_fixed_mask_threshold,
)
from common.instance_eval_report import extract_metric_from_report
from common.merged_view_pq import MERGED_VIEW_PQ_RESULT_KEYS, merged_view_pq_column_name
from yolo.inference_profile_tune import (
    PROFILE_SELECTION_OBJECTIVE,
    append_grid_result_row,
    count_detector_jobs,
    detector_keys_per_variant,
    count_grid_candidates,
    grid_result_row_from_candidate_scoring,
    grid_results_fieldnames,
    iter_detector_jobs,
    iter_grid_candidates,
    load_grid_results_csv,
    load_grid_winner,
    load_tune_grid,
    mean_pq_across_variants,
    promote_profile_to_recipe,
    select_best_candidate,
    tune_grid_path,
    write_grid_winner_json,
)
from yolo.tests.profile_tune_fixtures import (
    constant_merged_view_pq_result,
    constant_metric_bundle,
    instance_metrics_report_for_pq,
)
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
    """INTENT: load_tune_grid parses a conf-only profile-selection search space from YAML."""
    grid_path = tmp_path / "grid.yaml"
    _write_grid(grid_path, conf=[0.2, 0.3])
    spec = load_tune_grid(grid_path)
    assert spec.grid.conf == (0.2, 0.3)


def test_load_tune_grid_rejects_multi_value_mask_threshold(tmp_path: Path) -> None:
    """INTENT: load_tune_grid rejects grids listing multiple mask_threshold values."""
    grid_path = tmp_path / "grid.yaml"
    grid_path.write_text(
        yaml.safe_dump({"grid": {"conf": [0.2], "mask_threshold": [0.4, 0.5, 0.6]}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="mask_threshold must not list multiple"):
        load_tune_grid(grid_path)


def test_load_tune_grid_rejects_obsolete_sahi_axes(tmp_path: Path) -> None:
    """INTENT: load_tune_grid rejects legacy SAHI postprocess grid axes."""
    grid_path = tmp_path / "grid.yaml"
    _write_legacy_grid(grid_path)
    with pytest.raises(ValueError, match="no longer a profile-selection axis"):
        load_tune_grid(grid_path)


def test_iter_detector_jobs_lists_flat_detector_grid(tmp_path: Path) -> None:
    """INTENT: iter_detector_jobs yields one job per variant times detector keys per variant."""
    _write_grid(tmp_path / "grid.yaml")
    spec = load_tune_grid(tmp_path / "grid.yaml")
    variants = ("PPL", "PPL+AllPPX")
    jobs = list(iter_detector_jobs(spec, variants))
    assert len(jobs) == len(variants) * detector_keys_per_variant(spec)


def test_iter_grid_candidates_one_per_conf(tmp_path: Path) -> None:
    """INTENT: iter_grid_candidates yields one unique candidate per conf grid value."""
    _write_grid(tmp_path / "grid.yaml", conf=[0.2, 0.3])
    spec = load_tune_grid(tmp_path / "grid.yaml")
    candidates = list(iter_grid_candidates(spec))
    assert len(candidates) == count_grid_candidates(spec)
    assert candidates[0] == profile_tune_candidate_from_conf(0.2)
    assert len({c.candidate_id() for c in candidates}) == len(candidates)


def test_grid_results_fieldnames_use_merged_view_pq_only() -> None:
    """INTENT: grid_results_fieldnames exposes merged-view PQ columns and omits legacy metric columns."""
    fieldnames = grid_results_fieldnames(("PPL",))
    assert "mean_pq" in fieldnames
    assert "mean_tp" in fieldnames
    assert "aji_plus__PPL" not in fieldnames
    assert "f1_iou75__PPL" not in fieldnames
    assert "mF1_iou50_95__PPL" not in fieldnames
    for key in MERGED_VIEW_PQ_RESULT_KEYS:
        assert f"mean_{key}" in fieldnames
        assert merged_view_pq_column_name(key, "PPL") in fieldnames


@pytest.mark.parametrize(
    ("report", "expected_pq"),
    [
        (instance_metrics_report_for_pq(0.42), 0.42),
        (
            {
                "samples": [
                    {"sample_id": "a", **constant_metric_bundle(0.1)},
                    {"sample_id": "b", **constant_metric_bundle(0.9)},
                ],
                "mean": constant_metric_bundle(0.5),
            },
            0.5,
        ),
        (
            {
                "samples": [
                    {"sample_id": "a", **constant_metric_bundle(0.2)},
                    {"sample_id": "b", **constant_metric_bundle(0.8)},
                ],
            },
            0.5,
        ),
    ],
)
def test_extract_mean_pq_from_report(report: dict, expected_pq: float) -> None:
    """INTENT: extract_metric_from_report returns PQ from report mean or averaged per-sample values."""
    assert extract_metric_from_report(
        report, PROFILE_SELECTION_OBJECTIVE
    ) == pytest.approx(expected_pq)


def test_select_best_candidate_maximizes_mean_train_pq() -> None:
    """INTENT: select_best_candidate chooses the row with the highest mean_pq."""
    rows = [
        {"candidate_id": "a", "mean_pq": 0.71},
        {"candidate_id": "b", "mean_pq": 0.82},
        {"candidate_id": "c", "mean_pq": 0.80},
    ]
    best = select_best_candidate(rows)
    assert best["candidate_id"] == "b"


def test_promote_profile_preserves_recipe_comments_and_structure(
    tmp_path: Path,
) -> None:
    """INTENT: promote_profile_to_recipe updates conf while preserving YAML comments and structure."""
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
    """INTENT: promote_profile_to_recipe updates YOLO profile scalars without changing unrelated recipe sections."""
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


def test_promote_profile_to_recipe_round_trips_through_recipe_loader(
    tmp_path: Path,
) -> None:
    """INTENT: promote_profile_to_recipe writes values loadable by load_test_inference_recipe."""
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


def test_promote_profile_rejects_missing_conf_key_and_preserves_recipe(
    tmp_path: Path,
) -> None:
    """INTENT: promote_profile_to_recipe rejects missing conf without mutating the recipe file."""
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


def test_committed_tune_grid_is_conf_only_seven_candidates() -> None:
    """INTENT: committed tune grid is conf-only with seven candidates and fixed mask threshold."""
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
    """INTENT: write_grid_winner_json persists per-variant merged-view PQ results, not legacy per_variant_pq."""
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
        from common.merged_view_pq import _merged_view_pq_value

        for key in MERGED_VIEW_PQ_RESULT_KEYS:
            assert _merged_view_pq_value(stored, key) == _merged_view_pq_value(
                expected, key
            )


def test_load_grid_winner_rejects_nested_profile_without_top_level_conf(
    tmp_path: Path,
) -> None:
    """INTENT: winner.json must use flat conf; nested profile-only payloads are invalid."""
    winner_path = tmp_path / "winner.json"
    winner_path.write_text(
        json.dumps({"profile": {"conf": 0.25}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="conf"):
        load_grid_winner(winner_path)


def test_load_grid_winner_round_trips_profile(tmp_path: Path) -> None:
    """INTENT: load_grid_winner round-trips a profile written by write_grid_winner_json."""
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


def test_append_grid_result_row_writes_incrementally(tmp_path: Path) -> None:
    """INTENT: append_grid_result_row appends rows incrementally to results.csv."""
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
        merged_view_pq_column_name("pq", "PPL"): 0.8,
    }
    append_grid_result_row(csv_path, row_b, variant_names=("PPL",))
    assert len(load_grid_results_csv(csv_path)) == 2


def test_finalize_grid_winner_writes_per_variant_pq_results_from_csv(
    tmp_path: Path,
) -> None:
    """INTENT: finalize_grid_winner writes winner.json with per-variant PQ results from grid rows."""
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
    assert payload["mean_pq"] == pytest.approx(
        mean_pq_across_variants({"PPL": 0.82, "PPL+AllPPX": 0.58})
    )
    assert payload["per_variant_pq_results"]["PPL"]["pq"] == pytest.approx(0.82)
    assert load_grid_winner(grid_dir / "winner.json") == winner


def test_recompute_winner_from_csv_picks_highest_mean_pq(tmp_path: Path) -> None:
    """INTENT: recompute_winner_from_csv selects the candidate with highest mean_pq from results.csv."""
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


def test_grid_coordinator_cache_miss_fails_validation(tmp_path: Path) -> None:
    """INTENT: validate_detector_caches fails fast when coordinator finds missing detector caches."""
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


def test_candidate_scoring_cache_fingerprint_mismatch(tmp_path: Path) -> None:
    """INTENT: score_variant_train_metrics_from_cache raises when proposal cache conf mismatches candidate."""
    from common.test_inference import load_test_inference_recipe
    from yolo.profile_tune_candidate import score_variant_train_metrics_from_cache

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
