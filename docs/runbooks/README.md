# Runbooks

Cluster procedures for the GrainSegmentation research pipeline. Run all submit scripts from the **repo root** (`GrainSegmentation/`).

**Submit scripts:** all `SLURM/<area>/submit_*.sh` files are **login-node launchers** — run them with `bash` from the repo root; they call `sbatch` on the corresponding `run_*.sh` job scripts internally. Do not `sbatch` a `submit_*.sh` script.

**Scratch root:** `$SCRATCH/GrainSeg` by default ([`docs/reference/scratch-layout.md`](../reference/scratch-layout.md)). **Manifests:** [`docs/manifests.md`](../manifests.md). **Staging:** [`docs/reference/staging.md`](../reference/staging.md).

## Runbook template

Each runbook includes:

1. Prerequisites and links
2. Pipeline overview (diagram)
3. One subsection per submit workflow (resources, outputs, examples)
4. Upstream / downstream

## Pipelines

| Order | Runbook | Purpose |
|-------|---------|---------|
| 1 | [preprocessing.md](preprocessing.md) | Download, labels, blends, rasterize, patches, manifests |
| 2 | [yolo.md](yolo.md) | YOLO tune/train, profile selection, test eval |
| 2 | [unet.md](unet.md) | U-Net tune/train, watershed tune, train/test eval |
| 3 | [analysis.md](analysis.md) | Post-eval tables and figures |

YOLO and U-Net training/eval can proceed in parallel after preprocessing. **Profile selection** runs after all YOLO variant weights exist. **Post-eval reporting** runs after YOLO and U-Net test eval jobs complete.

**Evaluation policy:** [`docs/metrics.md` § PQ-centered rerun policy](../metrics.md#pq-centered-rerun-policy) (ADR [0003](../adr/0003-test-evaluation-policy.md)).

## High-level experiment sequence

Per input variant (see root [`README.md`](../../README.md)):

1. Tune and train YOLO → profile selection (once, all variants) → YOLO test eval  
2. Tune and train U-Net → watershed tune → pick CC vs watershed on train → U-Net test eval  
3. Build reporting bundle
