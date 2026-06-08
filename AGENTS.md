This codebase contains the work for a thesis with the following Research Qeustion: How do different multi-modal microscopy input configurations affect instance grain segmentation accuracy in sandstone thin-section images, when using U-Net semantic segmentation with postprocessing-based instance extraction versus YOLO direct instance segmentation?

This repo is a EARLY WIP. Proposing sweeping changes that improve long-term maintainability is encouraged. Long term maintainability is a core priority. If you add new functionality, first check if there is shared logic that can be extracted to a separate module. Duplicate logic across multiple files is a code smell and should be avoided. Don't be afraid to change existing code. Don't take shortcuts by just adding local logic to solve a problem.

- Don't code for backwards compatibility. Don't keep legacy code. Don't use thin wrappers to avoid breaking changes in consumers. If a breaking change is introduced, adapt the rest of the code. There are no out-of-repo consumers, this repo is used by only one person always on the latest version.
- This workplace is on a SLURM login node. Keep direct terminal commands lightweight: don't run long, intensive, scripts directly on the login node. For substantial work, use `srun` or `sbatch`
- Use brief conventional commits for git
- use `uv` for python

## Agent skills

### Issue tracker

Issues live as markdown under `.scratch/<feature-slug>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Five canonical triage roles with default label strings. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.