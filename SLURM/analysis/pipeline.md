# Post-eval reporting (SLURM)

CPU-only job that discovers finished test eval artifacts on scratch and writes the **reporting bundle** (`eval/reporting/`).

## Submit

From the repo root:

```bash
bash SLURM/analysis/submit_build_report.sh
```

Or directly:

```bash
sbatch SLURM/analysis/run_build_report.sh
```

## Resources

| Setting | Value |
|---------|--------|
| Memory | 8G |
| CPUs | 4 |
| Time | 15m |
| GPU | none |

## Optional environment

| Variable | Effect |
|----------|--------|
| `GRAINSEG_ROOT` | Override scratch root (default `$SCRATCH/GrainSeg`) |
| `OUTPUT_DIR` | Override bundle path (default `$GRAINSEG_ROOT/eval/reporting`) |
| `REPORT_STRICT=1` | Pass `--strict` to the CLI |
| `REPORT_NO_FIGURES=1` | Pass `--no-figures` (tables + summary only) |

## When to run

After YOLO and U-Net whole-section (and optional patch) test eval jobs have written `instance_metrics.json` under `eval/`. Logs: `logs/post_eval_report-<jobid>.log`.
