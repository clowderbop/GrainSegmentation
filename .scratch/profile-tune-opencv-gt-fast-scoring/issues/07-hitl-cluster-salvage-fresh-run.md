Status: ready-for-human
Category: enhancement
Labels: ready-for-human, enhancement
Depends-on: .scratch/profile-tune-opencv-gt-fast-scoring/issues/06-slurm-profile-tune-resources-docs.md

# HITL: Cluster salvage — fresh profile selection grid

## What to build

On the cluster, execute the ADR 0006 + 0007 salvage: delete the entire in-flight `runs/yolo_inference_profile_tune/<old_run_id>/` directory, submit **profile selection** with a new `RUN_ID`, and let the full pipeline complete (detector array → GT cache → candidate array → finalize).

Confirm the fix worked: at least one **profile selection** candidate task finishes all registry variants within the 4 h walltime (order of minutes per variant, not hours stuck on variant 1/4); finalize writes `grid/results.csv` and `grid/winner.json`. Do not compare grid winners to pre-ADR runs.

## Acceptance criteria

- [ ] Prior tune run directory removed; new `RUN_ID` submitted
- [ ] Detector, GT-cache, candidate array, and finalize jobs succeed
- [ ] Sample candidate log shows four variants scored with AJI lines and total time ≪ 4 h
- [ ] Sample candidate log shows one GT cache load (not four) and per-phase timings (merge, score merge, AJI)
- [ ] `grid/winner.json` exists and is internally consistent with `grid/results.csv`
- [ ] No `--skip-detectors` from the deleted run

## Blocked by

- [.scratch/profile-tune-opencv-gt-fast-scoring/issues/06-slurm-profile-tune-resources-docs.md](06-slurm-profile-tune-resources-docs.md)
