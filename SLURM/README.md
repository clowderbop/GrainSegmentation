# SLURM submit scripts

Run all `submit_*.sh` scripts from the **repo root**. Operational steps live in [`docs/runbooks/`](../docs/runbooks/), not in this tree.

| Area | Scripts | Runbook |
|------|---------|---------|
| `preprocessing/` | Download, labels, blends, patches | [`docs/runbooks/preprocessing.md`](../docs/runbooks/preprocessing.md) |
| `yolo/` | Tune/train, profile selection, test eval | [`docs/runbooks/yolo.md`](../docs/runbooks/yolo.md) |
| `unet/` | Tune/train, watershed, test eval | [`docs/runbooks/unet.md`](../docs/runbooks/unet.md) |
| `analysis/` | Post-eval reporting | [`docs/runbooks/analysis.md`](../docs/runbooks/analysis.md) |

Shared helpers: `SLURM/utils/` (paths, manifests, venv entry). See [`docs/runbooks/README.md`](../docs/runbooks/README.md) for pipeline order and prerequisites.
