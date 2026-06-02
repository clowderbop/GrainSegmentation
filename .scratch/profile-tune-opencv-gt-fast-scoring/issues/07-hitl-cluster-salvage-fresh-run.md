Status: done
Category: enhancement
Labels: enhancement
Depends-on: .scratch/profile-tune-opencv-gt-fast-scoring/issues/06-slurm-profile-tune-resources-docs.md

# HITL: Cluster salvage — fresh profile selection grid

## What to build

On the cluster, execute the ADR 0006 + 0007 salvage: delete the entire in-flight `runs/yolo_inference_profile_tune/<old_run_id>/` directory, submit **profile selection** with a new `RUN_ID`, and let the full pipeline complete (detector array → GT cache → candidate array → finalize).

Confirm the fix worked: at least one **profile selection** candidate task finishes all registry variants within the 4 h walltime (order of minutes per variant, not hours stuck on variant 1/4); finalize writes `grid/results.csv` and `grid/winner.json`. Do not compare grid winners to pre-ADR runs.

## Acceptance criteria

- [x] Prior tune run directory removed; new `RUN_ID` submitted
- [x] Detector, GT-cache, candidate array, and finalize jobs succeed
- [x] Sample candidate log shows four variants scored with AJI lines and total time ≪ 4 h
- [x] Sample candidate log shows one GT cache load (not four) and per-phase timings (merge, score merge, AJI)
- [x] `grid/winner.json` exists and is internally consistent with `grid/results.csv`
- [x] No `--skip-detectors` from the deleted run

## Blocked by

- [.scratch/profile-tune-opencv-gt-fast-scoring/issues/06-slurm-profile-tune-resources-docs.md](06-slurm-profile-tune-resources-docs.md)

## Comments

**2026-06-02 — Cluster salvage verified**

- `RUN_ID=20260601_155942` → `$SCRATCH/GrainSeg/runs/yolo_inference_profile_tune/20260601_155942/`
- Prior run `20260531_215615` removed; full pipeline submitted (no `--skip-detectors`)
- SLURM: candidate array `29208383` (108/108 `COMPLETED`, 1 CPU); finalize `29208384` `COMPLETED`
- Per-task walltime ~4–22 min (max `29208383_10`); scoring total ~6 min after warm venv
- Sample log: `logs/yolo_prof_cand-1-29208393.log` — one `load GT`, `[1/4]`–`[4/4]` AJI, slice-merge / score-merge / AJI timings, v2 caches
- `grid/rows/` ×108, `grid/results.csv` + `grid/winner.json`; `profile_tune_finalize --recompute-winner-from-csv` unchanged winner
- Winner: `GREEDYNMM` / `IOU` / match `0.6` / conf `0.15` / mask `0.4` (mean train AJI 0.176)
