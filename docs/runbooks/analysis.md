# Post-eval reporting runbook

CPU-only job that discovers finished **test** eval artifacts on scratch and writes the **reporting bundle** under `eval/reporting/`. Not part of inference or per-job metric computation. See [`docs/metrics.md`](../metrics.md) and **Post-eval reporting** in [`CONTEXT.md`](../../CONTEXT.md).

## Prerequisites

- YOLO and U-Net whole-section test eval jobs have written `instance_metrics.json` under `eval/`.
- Optional patch eval outputs for supporting panels.

## Pipeline overview

```mermaid
flowchart LR
  E[YOLO + U-Net test eval] --> D[discover runs]
  D --> R[build_report]
  R --> B[eval/reporting/]
```

## Build report

**Submit:** `bash SLURM/analysis/submit_build_report.sh`

**Run:** `sbatch SLURM/analysis/run_build_report.sh`

### Resources

| Setting | Value |
|---------|--------|
| Memory | 8G |
| CPUs | 4 |
| Time | 15m |
| GPU | none |

### Optional environment

| Variable | Effect |
|----------|--------|
| `GRAINSEG_ROOT` | Override scratch root (default `$SCRATCH/GrainSeg`) |
| `OUTPUT_DIR` | Override bundle path (default `$GRAINSEG_ROOT/eval/reporting`) |
| `REPORT_STRICT=1` | Pass `--strict` to the CLI |
| `REPORT_NO_FIGURES=1` | Pass `--no-figures` (tables + summary only) |

### Local (login node)

```bash
uv sync --group analysis
uv run --group analysis python -m analysis.build_report \
  --grainseg-root "$SCRATCH/GrainSeg"
```

### Examples

```bash
bash SLURM/analysis/submit_build_report.sh
# or
sbatch SLURM/analysis/run_build_report.sh
```

Logs: `logs/post_eval_report-<jobid>.log`.

## Outputs

Under `$SCRATCH/GrainSeg/eval/reporting/` (or `OUTPUT_DIR`):

| Path | Contents |
|------|----------|
| `derived/` | Comparison tables |
| `figures/` | Thesis charts (AJI/F1 heatmap, model×configuration bars, PPL-relative delta; supporting YOLO patch-val panel) |
| `analysis_summary.json` | Run summary |

**Eval run discovery (v1):** `analysis.build_report` locates runs by path conventions per **producer**, registry variant key, and **sample unit** (whole vs patch). Implementation: `src/analysis/discover.py`. No catalog file in v1.

Variant axis labels use `display_name` in `config/variants.yaml` (thesis order: PPL → FullStack → PPL+XPLComp → FullComp).

## Upstream

| Inputs | From |
|--------|------|
| `eval/yolo_{variant}/`, `eval/yolo_patches/` | [yolo.md](yolo.md) |
| `eval/unet_test/`, `eval/unet_patches/` | [unet.md](unet.md) |
