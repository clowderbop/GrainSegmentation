Status: done
Category: enhancement
Labels: ready-for-agent, enhancement
Depends-on: .scratch/profile-tune-opencv-gt-fast-scoring/issues/05-fast-profile-selection-scoring-parity.md
Blocks: .scratch/profile-tune-opencv-gt-fast-scoring/issues/07-hitl-cluster-salvage-fresh-run.md

# SLURM profile tune resources and pipeline docs

## What to build

Align SLURM with ADR 0007 post-fix expectations: **profile tune candidate** jobs request **1 CPU**, **4 h** walltime, **32G** memory (drop 8 CPUs). Ensure GT-cache and submit scripts match ADR 0006/0007 (GT job **common**-only sync if not already done in slice 02). Update `SLURM/yolo/pipeline.md` and submit script help text for salvage: delete entire prior `runs/yolo_inference_profile_tune/<run_id>/`, submit new `RUN_ID`, full detector → GT → candidate → finalize pipeline (no `--skip-detectors` from old runs).

## Acceptance criteria

- [x] `run_profile_tune_candidate.sh` uses `#SBATCH --cpus-per-task=1` (4h, 32G unchanged unless justified)
- [x] Pipeline docs describe fresh `RUN_ID` salvage and v1 cache invalidation
- [x] Submit script / docs cross-link ADR 0006 and 0007 consequences
- [x] GT-cache SLURM script consistent with slice 02 (common-only, train layout)

## Blocked by

- [.scratch/profile-tune-opencv-gt-fast-scoring/issues/05-fast-profile-selection-scoring-parity.md](05-fast-profile-selection-scoring-parity.md)
