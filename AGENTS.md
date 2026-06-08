Codebase for thesis with RQ: how do multi-modal microscopy input configurations affect instance grain segmentation accuracy in sandstone thin-section images, comparing U-Net (semantic segmentation + instance extraction) vs YOLO (direct instance segmentation)?

**Environment:** SLURM login node — lightweight commands only; `srun`/`sbatch` for real work. Use `uv`, conventional commits.

This repo is EARLY WIP. Long term maintainability is a core priority: proposing sweeping changes that improve it is encouraged. If you add new functionality, first check if there is shared logic that can be extracted to a separate module. Duplicate logic across multiple files is a code smell and should be avoided. Don't be afraid to change existing code. Don't take shortcuts by just adding local logic to solve a problem.

## Standards

- Prefer Deep modules: hide complexity behind small interfaces
- Use Deletion test: no pass-through abstractions
- One implementation ≠ a seam; two adapters make a seam: don't over-abstract
- Patch at the seam
- Apply breaking changes freely (there's no out-of-repo consumers); migrate all caller in the same change
- Forbidden: Compat branches/fallbacks/shims; pass-through wrappers; unused parameters like `del x`; dead code; duplicate code; bare domain literals
- Test through the public interface; assert behavior, not internals
- Don't write tests for documentation
- Skip trivial tests. Before writing test, write their intent in a docstring starting with "INTENT:"
- Tests should not read source, shell, or markdown files to assert substrings or structure

## Agent skills

### Issue tracker

Issues live as markdown under `.scratch/<feature-slug>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Five canonical triage roles with default label strings. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.