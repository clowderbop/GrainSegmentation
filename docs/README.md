# Documentation map

| Layer | Location | Purpose |
|-------|----------|---------|
| Glossary | [`CONTEXT.md`](../CONTEXT.md) | Domain terms, test policy, profile selection vocabulary |
| Decisions | [`docs/adr/`](adr/) | Why the system is shaped this way |
| Runbooks | [`docs/runbooks/`](runbooks/) | How to run cluster workflows (canonical ops docs) |
| Contracts | [`docs/manifests.md`](manifests.md) | Dataset manifest schema and paths |
| Reference | [`docs/reference/`](reference/) | Scratch layout, staging on compute nodes |
| Background | [`docs/dataset.md`](dataset.md) | Microscopy data, QGIS labels, overlap splitting, patches |
| Metrics | [`docs/metrics.md`](metrics.md) | Metric definitions; policy terms link to `CONTEXT.md` |
| Agents | [`docs/agents/`](agents/) | Issue tracker, triage, domain doc conventions |

## Runbooks

| Area | Runbook |
|------|---------|
| Index | [`runbooks/README.md`](runbooks/README.md) |
| Preprocessing | [`runbooks/preprocessing.md`](runbooks/preprocessing.md) |
| YOLO | [`runbooks/yolo.md`](runbooks/yolo.md) |
| U-Net | [`runbooks/unet.md`](runbooks/unet.md) |
| Post-eval reporting | [`runbooks/analysis.md`](runbooks/analysis.md) |

Submit scripts and `#SBATCH` defaults live under `SLURM/<area>/`. Operational steps belong in the runbooks above—not in script `--help` beyond flags and one-line pointers.

## Quick links

- Scratch root and directory tree: [`reference/scratch-layout.md`](reference/scratch-layout.md)
- Manifest staging on nodes: [`reference/staging.md`](reference/staging.md)
- Variant registry: `config/variants.yaml` (CLI: `uv run python -m common.variants`)
