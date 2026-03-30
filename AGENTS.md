# AGENTS.md

These rules apply to the whole repository.

## Scope and ownership
- Treat `src/` as the source of truth for benchmark logic, contracts, and evaluation behavior.
- Treat `packaging/kaggle/` as downstream packaging only; it must not redefine benchmark semantics.
- Treat `scripts/` as operational helpers only; they must consume canonical state from `src/` and `packaging/kaggle/`.

## Product boundaries
- Keep the benchmark narrowly scoped to cognitive flexibility.
- Keep Binary as the only leaderboard-primary path.
- Keep Narrative supplemental to Binary, never a separate primary metric.
- Preserve strict public/private isolation. Never place private split artifacts in public repo paths or public packaging outputs.
- Do not keep parallel in-repo deploy trees or duplicate packaging surfaces for the MVP flow.

## Change policy
- Prefer small, auditable changes over broad refactors.
- Remove redundant docs instead of creating parallel documentation.
- Update docs only when they reflect the current code and workflow.
- Do not edit or commit generated artifacts or build residue (`build/`, `*.egg-info`, temporary staging outputs) unless the task explicitly requires it.

## Validation
- Run targeted tests for changed areas before finishing.
- If runtime execution, logging, or artifact generation changes, run:
  - `pytest tests/test_kaggle_execution.py tests/test_run_logging.py tests/test_diagnostics_summary.py tests/test_run_manifest.py tests/test_episode_ledger.py`
- If packaging or deploy code changes, run:
  - `pytest tests/test_cd_build.py tests/test_kbench_notebook.py tests/test_packaging.py`
- If split or packaging logic changes, verify public/private isolation before finishing.

## Commit messages
- Use commit messages in the format `type(scope): message`.