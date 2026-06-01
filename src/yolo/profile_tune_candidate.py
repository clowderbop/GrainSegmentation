"""CLI: score one profile-selection grid candidate (ADR 0005)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from common.reporting import count_instances
from common.test_inference import YoloInferenceProfileCandidate
from common.variants import repo_root
from yolo.inference_profile_tune import (
    TuneGridSpec,
    candidate_at_grid_index,
    iter_grid_candidates,
    load_profile_selection_row,
    load_tune_grid,
    profile_selection_row_path,
    tune_grid_fingerprint,
    write_profile_selection_row,
)
from yolo.profile_tune_cli import parse_profile_tune_variants
from yolo.profile_tune_gt_cache import (
    build_gt_fingerprint,
    gt_cache_dir,
    load_gt_instance_map_cache,
    train_labels_gpkg_path,
)
from yolo.profile_tune_scoring import compute_train_aji
from yolo.profile_tune_work import (
    default_grainseg_and_run_roots,
    weights_path,
)
from yolo.tiled_proposal_cache import (
    detector_cache_expected_record,
    load_tiled_proposals,
    proposal_cache_dir,
    sahi_predictions_from_tiled_proposal_records,
)


def _log(*parts: object) -> None:
    print(*parts, flush=True)


def validate_detector_caches_for_candidate(
    *,
    work_root: Path,
    candidate: YoloInferenceProfileCandidate,
    variants: tuple[str, ...],
    grainseg_root: Path,
    run_root: Path,
) -> None:
    """Ensure tiled-proposal caches exist for this candidate's (conf, mask_threshold)."""
    _log(
        f"Checking detector caches for conf={candidate.conf:g} "
        f"mask_threshold={candidate.mask_threshold:g} …"
    )
    missing: list[str] = []
    for variant in variants:
        weights = weights_path(grainseg_root, variant, run_root)
        cache_dir = proposal_cache_dir(
            work_root / variant,
            conf=candidate.conf,
            mask_threshold=candidate.mask_threshold,
        )
        expected = detector_cache_expected_record(
            variant=variant,
            weights_path=weights,
            conf=candidate.conf,
            mask_threshold=candidate.mask_threshold,
            sample_id="train",
        )
        try:
            proposals, meta = load_tiled_proposals(cache_dir, expected=expected)
            n_props = len(proposals)
            _log(
                f"  detector cache OK: {variant} → {cache_dir.name} "
                f"({n_props} proposals, weights_sha256={meta.get('weights_sha256', '')[:12]}…)"
            )
        except (FileNotFoundError, ValueError) as exc:
            missing.append(f"{variant}({exc})")
    if missing:
        raise FileNotFoundError(
            "Missing or invalid detector caches for candidate: " + "; ".join(missing)
        )


def candidate_row_fingerprint(
    *,
    candidate: YoloInferenceProfileCandidate,
    variants: tuple[str, ...],
    grainseg_root: Path,
    run_root: Path,
    work_root: Path,
    grid_config: Path | None,
) -> dict[str, Any]:
    labels_gpkg = train_labels_gpkg_path(grainseg_root)
    per_variant: dict[str, Any] = {}
    for variant in variants:
        weights = weights_path(grainseg_root, variant, run_root)
        from yolo.tiled_proposal_cache import weights_sha256

        gt_fp = build_gt_fingerprint(
            variant=variant,
            sample_id="train",
            labels_gpkg=labels_gpkg,
        )
        per_variant[variant] = {
            "weights_sha256": weights_sha256(weights),
            "gt_cache_fingerprint": gt_fp,
            "proposal_cache_key": (
                f"c{candidate.conf:g}_t{candidate.mask_threshold:g}"
            ),
        }
    return {
        "candidate_id": candidate.candidate_id(),
        "tune_grid_fingerprint": tune_grid_fingerprint(grid_config),
        "variants": per_variant,
    }


def row_fingerprint_matches(
    stored: dict[str, Any], expected: dict[str, Any]
) -> bool:
    return stored == expected


def score_variant_train_aji_from_cache(
    *,
    variant: str,
    candidate: YoloInferenceProfileCandidate,
    grainseg_root: Path,
    run_root: Path,
    work_root: Path,
    variant_index: int | None = None,
    variant_count: int | None = None,
) -> float:
    prefix = ""
    if variant_index is not None and variant_count is not None:
        prefix = f"[{variant_index}/{variant_count}] "
    _log(f"{prefix}Scoring {variant} …")
    t0 = time.perf_counter()

    weights = weights_path(grainseg_root, variant, run_root)
    cache_dir = proposal_cache_dir(
        work_root / variant, conf=candidate.conf, mask_threshold=candidate.mask_threshold
    )
    expected_proposals = detector_cache_expected_record(
        variant=variant,
        weights_path=weights,
        conf=candidate.conf,
        mask_threshold=candidate.mask_threshold,
        sample_id="train",
    )
    records, meta = load_tiled_proposals(cache_dir, expected=expected_proposals)
    cache_height = int(meta["height"])
    cache_width = int(meta["width"])
    proposals = sahi_predictions_from_tiled_proposal_records(
        records, height=cache_height, width=cache_width
    )
    labels_gpkg = train_labels_gpkg_path(grainseg_root)
    expected_gt = build_gt_fingerprint(
        variant=variant,
        sample_id="train",
        labels_gpkg=labels_gpkg,
    )
    gt_cache_path = gt_cache_dir(work_root, variant)
    gt_map, _gt_meta = load_gt_instance_map_cache(gt_cache_path, expected=expected_gt)
    height, width = int(gt_map.shape[0]), int(gt_map.shape[1])
    gt_n = count_instances(gt_map)
    _log(
        f"{prefix}  loaded {len(proposals)} proposals, GT {height}×{width} "
        f"({gt_n} instances) from {gt_cache_path.name}/"
    )
    aji = compute_train_aji(
        gt_map,
        proposals,
        candidate=candidate,
        height=height,
        width=width,
    )
    elapsed = time.perf_counter() - t0
    _log(f"{prefix}  {variant}: train AJI={aji:.6f} ({elapsed:.1f}s)")
    return aji


def build_profile_selection_row(
    *,
    candidate: YoloInferenceProfileCandidate,
    per_variant_aji: dict[str, float],
    fingerprint: dict[str, Any],
) -> dict[str, Any]:
    mean_aji = float(sum(per_variant_aji.values()) / len(per_variant_aji))
    row: dict[str, Any] = {
        "candidate_id": candidate.candidate_id(),
        **candidate.to_dict(),
        "mean_aji": mean_aji,
        "fingerprint": fingerprint,
    }
    for variant, aji in per_variant_aji.items():
        row[f"aji__{variant}"] = aji
    return row


def score_profile_selection_candidate(
    *,
    candidate: YoloInferenceProfileCandidate,
    variants: tuple[str, ...],
    output_dir: Path,
    grainseg_root: Path,
    run_root: Path,
    work_root: Path,
    grid_config: Path | None,
    resume: bool,
) -> Path:
    grid_dir = output_dir / "grid"
    row_path = profile_selection_row_path(grid_dir, candidate.candidate_id())
    fingerprint = candidate_row_fingerprint(
        candidate=candidate,
        variants=variants,
        grainseg_root=grainseg_root,
        run_root=run_root,
        work_root=work_root,
        grid_config=grid_config,
    )
    if resume and row_path.is_file():
        stored = load_profile_selection_row(row_path)
        if row_fingerprint_matches(stored.get("fingerprint", {}), fingerprint):
            _log(
                f"Resume: skipping score — row exists with matching fingerprint → {row_path}"
            )
            _log(
                f"  stored mean_aji={float(stored['mean_aji']):.6f} "
                f"({', '.join(f'{v}={float(stored[f'aji__{v}']):.4f}' for v in variants)})"
            )
            return row_path
        _log(
            f"Resume: fingerprint changed — re-scoring (row at {row_path} ignored)"
        )

    _log(
        f"Scoring {len(variants)} variants: "
        f"postprocess={candidate.postprocess_type} metric={candidate.match_metric} "
        f"match_threshold={candidate.match_threshold:g} conf={candidate.conf:g} "
        f"mask_threshold={candidate.mask_threshold:g}"
    )
    t_all = time.perf_counter()
    per_variant_aji: dict[str, float] = {}
    n_variants = len(variants)
    for idx, variant in enumerate(variants, start=1):
        per_variant_aji[variant] = score_variant_train_aji_from_cache(
            variant=variant,
            candidate=candidate,
            grainseg_root=grainseg_root,
            run_root=run_root,
            work_root=work_root,
            variant_index=idx,
            variant_count=n_variants,
        )
    mean_aji = float(sum(per_variant_aji.values()) / len(per_variant_aji))
    _log(
        f"Candidate {candidate.candidate_id()}: mean_aji={mean_aji:.6f} "
        f"(total {time.perf_counter() - t_all:.1f}s)"
    )
    row = build_profile_selection_row(
        candidate=candidate,
        per_variant_aji=per_variant_aji,
        fingerprint=fingerprint,
    )
    write_profile_selection_row(row_path, row)
    _log(f"Wrote profile selection row → {row_path}")
    return row_path


def resolve_candidate(
    *,
    spec: TuneGridSpec,
    candidate_id: str | None,
    array_index: int | None,
) -> YoloInferenceProfileCandidate:
    if candidate_id is not None and array_index is not None:
        raise ValueError("Specify only one of --candidate-id or --array-index")
    if candidate_id is not None:
        for candidate in iter_grid_candidates(spec):
            if candidate.candidate_id() == candidate_id:
                return candidate
        raise ValueError(f"Unknown candidate_id: {candidate_id}")
    if array_index is not None:
        return candidate_at_grid_index(spec, array_index)
    raise ValueError("One of --candidate-id or --array-index is required")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grainseg-root", type=Path, default=None)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--grid-config", type=Path, default=None)
    parser.add_argument("--candidate-id", default=None)
    parser.add_argument("--array-index", type=int, default=None)
    parser.add_argument("--variants", default=None)
    parser.add_argument("--work-root", type=Path, default=None)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip when grid/rows/{candidate_id}.json exists with matching fingerprint.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    variants = parse_profile_tune_variants(args.variants)
    spec = load_tune_grid(args.grid_config)
    candidate = resolve_candidate(
        spec=spec,
        candidate_id=args.candidate_id,
        array_index=args.array_index,
    )
    grainseg_root, run_root = default_grainseg_and_run_roots(
        args.grainseg_root, args.run_root
    )
    work_root = args.work_root or (args.output_dir / "_work")
    grid_dir = args.output_dir / "grid"
    _log("Profile selection candidate")
    _log(f"  output_dir={args.output_dir.resolve()}")
    _log(f"  work_root={work_root.resolve()}")
    _log(f"  resume={args.resume}")
    if args.array_index is not None:
        _log(f"  array_index={args.array_index}")
    _log(f"  candidate_id={candidate.candidate_id()}")
    if not args.resume:
        row_path = profile_selection_row_path(grid_dir, candidate.candidate_id())
        if row_path.is_file():
            _log(f"  --no-resume: removing stale row {row_path}")
            row_path.unlink()
    validate_detector_caches_for_candidate(
        work_root=work_root,
        candidate=candidate,
        variants=variants,
        grainseg_root=grainseg_root,
        run_root=run_root,
    )

    row_path = score_profile_selection_candidate(
        candidate=candidate,
        variants=variants,
        output_dir=args.output_dir,
        grainseg_root=grainseg_root,
        run_root=run_root,
        work_root=work_root,
        grid_config=args.grid_config,
        resume=args.resume,
    )
    row = load_profile_selection_row(row_path)
    _log("Result row (audit JSON):")
    print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
