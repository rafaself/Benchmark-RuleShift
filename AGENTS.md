# AGENTS.md

These rules apply to the whole repository.

- Treat `src/` as the source of truth for benchmark logic, contracts, and evaluation behavior.
- Treat `packaging/kaggle/` as downstream packaging only; it must not redefine benchmark semantics.
- Keep the benchmark narrowly scoped to cognitive flexibility.
- Keep Binary as the only leaderboard-primary path.
- Keep Narrative supplemental to Binary, never a separate primary metric.
- Preserve strict public/private isolation. Never place private split artifacts in public repo paths or public packaging outputs.
- Prefer small, auditable changes over broad refactors.
- Remove redundant docs instead of creating parallel documentation.
- Update docs only when they reflect the current code and workflow.
- Run targeted tests for changed areas. If split or packaging logic changes, verify public/private isolation before finishing.
- Use commit messages in the format `type(scope): message`.
