[Hippocratic License HL3-FULL](https://firstdonoharm.dev/version/3/0/full.html)

# GrainSegmentation

**Research question:** How do different multi-modal microscopy input configurations affect instance grain segmentation accuracy in sandstone thin-section images, when using U-Net semantic segmentation with postprocessing-based instance extraction versus YOLO direct instance segmentation?

The project compares four microscopy input variants across two model families, giving eight experiment pipelines:

- `PPL`: single-input baseline.
- `PPLPPXblend`: single blended composite input.
- `PPL+PPXblend`: two-input PPL plus PPX-blend configuration.
- `PPL+AllPPX`: seven-input configuration using PPL and all PPX images.

In this codebase, `PPX` refers to cross-polarized light (`XPL`).

For each variant, the intended experiment sequence is:

1. Tune YOLO hyperparameters.
2. Train the final YOLO model using the selected hyperparameters.
3. Evaluate YOLO on the held-out test set, patch-wise and whole-image-wise using sliding-window.
4. Tune U-Net hyperparameters.
5. Train the final U-Net model using the selected hyperparameters.
6. Tune watershed instance-extraction hyperparameters for the U-Net semantic predictions.
7. Compare connected components and tuned watershed instance extraction for U-Net outputs on the train set.
8. Evaluate the U-Net (using the best postprocessing method) on the held-out test set, patch-wise and whole-image-wise using sliding-window.
9. Compare results across all model families and input variants.

## Scratch and data

Persistent artifacts live under **`$SCRATCH/GrainSeg`** by default (`SLURM/utils/paths.sh`). The variant registry is `config/variants.yaml`; dataset file lists are JSON manifests on scratch. See [`docs/reference/scratch-layout.md`](docs/reference/scratch-layout.md) and [`docs/manifests.md`](docs/manifests.md).

## Research pipeline (summary)

Two **producer** families: **U-Net** (semantic classes → instance extraction) and **YOLO** (direct instance segmentation). YOLO loads stacked TIFFs per variant; U-Net loads per-channel 3-channel TIFFs concatenated in the model (see [`docs/dataset.md`](docs/dataset.md)).

After all YOLO weights exist, run **profile selection** on the train whole section, then **profile promotion** into `configs/test_inference.yaml` before held-out YOLO test eval. Details: [`docs/runbooks/yolo.md`](docs/runbooks/yolo.md).

Metrics and thesis ranking policy: [`docs/metrics.md`](docs/metrics.md) and [`CONTEXT.md`](CONTEXT.md).

## Documentation

| What | Where |
|------|--------|
| **Doc map** | [`docs/README.md`](docs/README.md) |
| **Cluster runbooks** | [`docs/runbooks/`](docs/runbooks/) (preprocessing, YOLO, U-Net, analysis) |
| **Glossary & policy** | [`CONTEXT.md`](CONTEXT.md) |
| **ADRs** | [`docs/adr/`](docs/adr/) |
| **Dataset background** | [`docs/dataset.md`](docs/dataset.md) |
| **Staging on nodes** | [`docs/reference/staging.md`](docs/reference/staging.md) |

SLURM submit scripts live under `SLURM/<area>/` and point at [`docs/runbooks/`](docs/runbooks/) in `--help` or header comments.
