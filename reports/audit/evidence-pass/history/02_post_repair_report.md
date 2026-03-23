# Post-Repair Evidence Pass Report

## Executive Summary

After the private split repair, the local RuleShift Benchmark is code-healthy, reproducible, and now clears the local benchmark-validity surface. The repaired private split passes the R13 anti-shortcut validity gate, the R15 deterministic re-audit no longer reports weak private slice separation as a blocker, and frozen artifact integrity remains clean. `hard` is still reserved and not emitted, and no real-model runs are bundled in-repo.

## Evidence Pass

Commands used:

```bash
.venv/bin/python -m pytest
.venv/bin/python scripts/ife.py evidence-pass -o /tmp/evidence_pass.json
```

Results:

- Full suite: `636 passed`
- R13 validity gate: `PASS`
- R15 re-audit: completed successfully
- Split overlap check: passed
- Frozen split audit issues: `0`
- Kaggle staging manifest validation: passed

## R13 Anti-Shortcut Gate

Gate outcome: `PASS`

Critical leaderboard checks all remained bounded at or below `0.55`.

Private-leaderboard subset separation:

- best template gap: `0.103175`
- best emitted-difficulty gap: `0.103175`
- strongest separating baseline: `template_position`

The gate now reports:

> The repaired benchmark clears the R13 anti-shortcut gate: `template_position` peaks at `0.500000` on `public_leaderboard`.

## R15 Re-Audit

The R15 note now reports:

- `last_evidence` peaks at `0.500000` on `public_leaderboard`
- recency no longer looks like the dominant shortcut failure mode
- the strongest critical heuristic remains `template_position=0.515625` on `dev`

Current remaining blockers in the re-audit surface:

- no bundled real-model runs in-repo
- `hard` remains reserved and un-emitted

## Frozen Artifact Integrity

Current frozen manifest summary:

| Partition | Manifest version | Seed bank version | Episode split | Episode count |
| --- | --- | --- | --- | ---: |
| `dev` | `R14` | `R14-dev-1` | `dev` | 16 |
| `public_leaderboard` | `R14` | `R14-public-1` | `public` | 16 |
| `private_leaderboard` | `R14` | `R14-private-2` | `private` | 16 |

Bundled evidence and packaging hashes were refreshed to match the repaired split and now validate cleanly.

## Bottom Line

Current local status: tests pass, frozen artifacts are consistent, R13 now passes, and the repaired benchmark is no longer blocked by weak private subset separation. The remaining gaps before real-model evaluation are unchanged: no bundled real-model evidence and no emitted `hard` slice.
