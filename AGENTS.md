This repository is a VERY EARLY WIP. Proposing sweeping changes that improve long-term maintainability is encouraged.
- Long term maintainability is a core priority. If you add new functionality, first check if there is shared logic that can be extracted to a separate module. Duplicate logic across multiple files is a code smell and should be avoided. Don't be afraid to change existing code. Don't take shortcuts by just adding local logic to solve a problem.
- Don't code for backwards compatibility; if a breaking change is introduced, adapt the rest of the code.
- Delete dead code immediately (unused imports, functions, variables, commented code). If it's not running, it goes.
- Read the README before starting. 
- Use `uv` as the Python package manager and execution tool.
- Agents for this project run on a SLURM cluster node. Do not run long, intensive, or training jobs directly on the node. Keep direct terminal commands lightweight. For substantial work, use `srun` for interactive runs or `sbatch` for scheduled jobs.
- Use brief conventional commits for commit messages. Suggest committing if it seems appropriate (e.g. starting to work on different feature).

## Agent skills

### Issue tracker

Issues live as markdown under `.scratch/<feature-slug>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Five canonical triage roles with default label strings. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.