# AGENTS.md

## Scope
These instructions apply to the entire repository.

## Source of truth
- Treat `src/` as the source of truth for benchmark logic, contracts, and evaluation behavior.
- Treat `packaging/kaggle/` as downstream packaging that must reflect, not redefine, benchmark behavior.

## Benchmark boundaries
- Keep the benchmark narrowly scoped to cognitive flexibility.
- Keep Binary as the only leaderboard-primary path.
- Keep Narrative strictly supplemental as an audit layer for Binary, never as the primary benchmark metric.

## Private/public isolation
- Never place private split artifacts in public repo paths or public packaging outputs.
- Preserve strict separation between public assets and authorized private leaderboard assets.

## Change discipline
- Prefer small, auditable changes over broad refactors.
- Do not introduce legacy compatibility unless explicitly required.
- Remove stale docs and redundant docs instead of adding parallel explanations.
- Update documentation only when it reflects the real current state of the codebase.
- Prefer test-first or test-aligned changes for benchmark logic, parsers, scoring, and split behavior.

## Validation
- Run targeted tests for the files you changed.
- When packaging or split logic changes, verify public/private isolation before finishing.

## Commit style
- Use commit messages in the format: `type(scope): message`.