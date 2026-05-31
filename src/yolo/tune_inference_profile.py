"""CLI: staged YOLO inference profile selection on train whole section."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from common.manifest_io import build_yolo_whole_manifest, write_dataset_manifest
from common.test_inference import (
    YoloInferenceProfileCandidate,
    load_test_inference_recipe,
    sahi_overlap_ratio,
)
from common.variants import all_variant_names, repo_root
from yolo.config import default_run_root, default_scratch_root
from yolo.inference_profile_tune import (
    iter_stage1_candidates,
    iter_stage2_candidates,
    load_tune_grid,
    score_candidate_across_variants,
    select_best_candidate,
    tune_grid_path,
    write_stage_results_csv,
    write_stage_winner_json,
    load_stage_winner,
)

VariantScorer = Callable[[str, YoloInferenceProfileCandidate, Path], Path]


def _uv_run_cmd(
    project_dir: Path,
    module: str,
    *args: str,
    unbuffered: bool = False,
) -> list[str]:
    python_cmd = ["python", "-u"] if unbuffered else ["python"]
    return [
        "uv",
        "run",
        "--directory",
        str(project_dir),
        *python_cmd,
        "-m",
        module,
        *args,
    ]


def _run_subprocess(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=cwd)


def _weights_path(grainseg_root: Path, variant: str, run_root: Path) -> Path:
    return run_root / variant / "weights" / "best.pt"


def _stage_manifest_path(work_root: Path, variant: str) -> Path:
    return work_root / variant / "yolo_whole_train.json"


def _staged_manifest_path(work_root: Path, variant: str) -> Path:
    return work_root / variant / "staged" / "manifest.json"


def _prepare_train_whole_manifest(
    grainseg_root: Path, variant: str, work_root: Path
) -> Path:
    manifest_path = _stage_manifest_path(work_root, variant)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_dataset_manifest(
        manifest_path,
        build_yolo_whole_manifest(
            split="train", variant=variant, grainseg_root=grainseg_root
        ),
    )
    return manifest_path


def _score_variant_on_cluster(
    *,
    variant: str,
    candidate: YoloInferenceProfileCandidate,
    variant_output_dir: Path,
    grainseg_root: Path,
    run_root: Path,
    work_root: Path,
    device: str,
    repo: Path,
) -> Path:
    variant_output_dir.mkdir(parents=True, exist_ok=True)
    weights = _weights_path(grainseg_root, variant, run_root)
    if not weights.is_file():
        raise FileNotFoundError(f"Missing YOLO weights for {variant}: {weights}")

    canonical_manifest = _prepare_train_whole_manifest(grainseg_root, variant, work_root)
    staged_root = work_root / variant / "staged"
    staged_root.mkdir(parents=True, exist_ok=True)

    _run_subprocess(
        _uv_run_cmd(
            repo / "src" / "common",
            "common.stage_manifest",
            "run",
            str(canonical_manifest),
            str(staged_root),
        )
    )
    staged_manifest = _staged_manifest_path(work_root, variant)
    if not staged_manifest.is_file():
        raise FileNotFoundError(f"Staged manifest missing: {staged_manifest}")

    recipe = load_test_inference_recipe()
    overlap = sahi_overlap_ratio(
        window=recipe.whole.window, stride=recipe.whole.stride
    )
    common_src = repo / "src" / "common"
    yolo_src = repo / "src" / "yolo"
    _run_subprocess(
        _uv_run_cmd(
            yolo_src,
            "yolo.predict",
            "--unit",
            "whole",
            "--weights",
            str(weights),
            "--variant",
            variant,
            "--manifest",
            str(staged_manifest),
            "--device",
            device,
            "--imgsz",
            str(recipe.whole.window),
            "--conf",
            str(candidate.conf),
            "--mask-threshold",
            str(candidate.mask_threshold),
            "--postprocess-type",
            candidate.postprocess_type,
            "--match-metric",
            candidate.match_metric,
            "--match-threshold",
            str(candidate.match_threshold),
            "--slice-height",
            str(recipe.whole.window),
            "--slice-width",
            str(recipe.whole.window),
            "--overlap-height-ratio",
            str(overlap),
            "--overlap-width-ratio",
            str(overlap),
            "--output-dir",
            str(variant_output_dir),
            unbuffered=True,
        )
    )

    eval_manifest = variant_output_dir / "eval_manifest.json"
    _run_subprocess(
        _uv_run_cmd(
            common_src,
            "common.stage_manifest",
            "write-eval",
            "--source",
            str(staged_manifest),
            "--prediction-set-dir",
            str(variant_output_dir),
            "--output",
            str(eval_manifest),
        )
    )

    metrics_path = variant_output_dir / "instance_metrics.json"
    _run_subprocess(
        _uv_run_cmd(
            common_src,
            "common.evaluate_instances",
            "--unit",
            "whole",
            "--model-type",
            "yolo",
            "--variant",
            variant,
            "--manifest",
            str(eval_manifest),
            "--output-json",
            str(metrics_path),
            unbuffered=True,
        )
    )
    return metrics_path


def _dry_run_scorer(
    variant_scores: dict[tuple[str, str], float],
) -> VariantScorer:
    def score(variant: str, candidate: YoloInferenceProfileCandidate, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        key = (variant, candidate.candidate_id())
        aji = variant_scores.get(key, 0.0)
        metrics_path = out_dir / "instance_metrics.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "samples": [{"sample_id": "train", "aji": aji}],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return metrics_path

    return score


def run_stage_search(
    *,
    stage: int,
    candidates: list[YoloInferenceProfileCandidate],
    variants: tuple[str, ...],
    output_dir: Path,
    score_variant: VariantScorer,
) -> tuple[YoloInferenceProfileCandidate, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    stage_dir = output_dir / f"stage{stage}"
    for candidate in candidates:
        candidate_dir = stage_dir / candidate.candidate_id()
        variant_reports: dict[str, Path] = {}
        for variant in variants:
            variant_out = candidate_dir / variant
            variant_reports[variant] = score_variant(variant, candidate, variant_out)
        mean_aji, per_variant = score_candidate_across_variants(variant_reports)
        row: dict[str, object] = {
            "candidate_id": candidate.candidate_id(),
            **candidate.to_dict(),
            "mean_aji": mean_aji,
        }
        for variant, aji in per_variant.items():
            row[f"aji__{variant}"] = aji
        rows.append(row)
        print(
            f"stage{stage} {candidate.candidate_id()}: mean_aji={mean_aji:.6f}",
            flush=True,
        )

    write_stage_results_csv(
        stage_dir / "results.csv",
        rows,
        variant_names=variants,
    )
    best_row = select_best_candidate(rows)
    winner = YoloInferenceProfileCandidate(
        postprocess_type=str(best_row["postprocess_type"]),
        match_metric=str(best_row["match_metric"]),
        match_threshold=float(best_row["match_threshold"]),
        conf=float(best_row["conf"]),
        mask_threshold=float(best_row["mask_threshold"]),
    )
    write_stage_winner_json(
        stage_dir / "winner.json",
        stage=stage,
        candidate=winner,
        mean_aji=float(best_row["mean_aji"]),
        per_variant={
            variant: float(best_row[f"aji__{variant}"]) for variant in variants
        },
    )
    return winner, rows


def _parse_variants(raw: str | None) -> tuple[str, ...]:
    if raw is None:
        return all_variant_names()
    names = tuple(v.strip() for v in raw.split(",") if v.strip())
    if not names:
        raise ValueError("--variants must list at least one registry variant")
    return names


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Scratch audit directory for stage tables and winners.",
    )
    parser.add_argument(
        "--grainseg-root",
        type=Path,
        default=None,
        help="GrainSeg scratch root (default: $SCRATCH/GrainSeg).",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="YOLO weights root (default: $SCRATCH/GrainSeg/runs/yolo26-seg).",
    )
    parser.add_argument(
        "--grid-config",
        type=Path,
        default=None,
        help="Search grid YAML (default: configs/yolo_inference_profile_tune.yaml).",
    )
    parser.add_argument(
        "--stage",
        choices=("1", "2", "all"),
        default="all",
        help="Run stage 1 only, stage 2 only (needs stage-1 winner), or both.",
    )
    parser.add_argument(
        "--stage1-winner",
        type=Path,
        default=None,
        help="stage1/winner.json path (required for --stage 2).",
    )
    parser.add_argument("--variants", default=None)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip predict/eval; write synthetic metrics (for tests).",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=None,
        help="Manifest staging directory (default: output-dir/_work).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    repo = repo_root()
    grainseg_root = args.grainseg_root or default_scratch_root() / "GrainSeg"
    run_root = args.run_root or default_run_root()
    work_root = args.work_root or (args.output_dir / "_work")
    variants = _parse_variants(args.variants)
    spec = load_tune_grid(args.grid_config)

    if args.dry_run:
        score_variant = _dry_run_scorer({})
    else:
        score_variant = lambda variant, candidate, out_dir: _score_variant_on_cluster(
            variant=variant,
            candidate=candidate,
            variant_output_dir=out_dir,
            grainseg_root=grainseg_root,
            run_root=run_root,
            work_root=work_root,
            device=args.device,
            repo=repo,
        )

    stage1_winner: YoloInferenceProfileCandidate | None = None
    if args.stage in ("1", "all"):
        stage1_winner, _ = run_stage_search(
            stage=1,
            candidates=list(iter_stage1_candidates(spec)),
            variants=variants,
            output_dir=args.output_dir,
            score_variant=score_variant,
        )

    if args.stage in ("2", "all"):
        if args.stage == "2":
            winner_path = args.stage1_winner or (
                args.output_dir / "stage1" / "winner.json"
            )
            if not winner_path.is_file():
                raise FileNotFoundError(
                    f"Stage 1 winner required for stage 2: {winner_path}"
                )
            stage1_winner = load_stage_winner(winner_path)
        assert stage1_winner is not None
        final_winner, _ = run_stage_search(
            stage=2,
            candidates=list(iter_stage2_candidates(spec, stage1_winner)),
            variants=variants,
            output_dir=args.output_dir,
            score_variant=score_variant,
        )
        print("\nFinal YOLO inference profile (promote with yolo.promote_inference_profile):")
        print(json.dumps(final_winner.to_dict(), indent=2))


if __name__ == "__main__":
    main()
