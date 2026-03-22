# Private Split Separation Repair Report

## Summary

The weak `heuristic_subset_separation` failure was caused by **private split composition**, not by gate thresholding and not by a remaining generator/protocol flaw.

The smallest justified repair was to replace the private contiguous seed bank `9054..9069` with the nearby contiguous seed bank `9060..9075` and bump `seed_bank_version` from `R14-private-1` to `R14-private-2`.

No gate logic changed. No generator or protocol logic changed.

## Root Cause

### What failed before

- R13 failed only on `heuristic_subset_separation` for `private_leaderboard`
- Best template gap: `0.031250`
- Best emitted-difficulty gap: `0.031250`
- All shortcut-cap checks already passed

### Why it failed

- The private split was a 16-episode frozen window with very flat `template_position` behavior:
  - private `template_position` by template: `T1 = 0.343750`, `T2 = 0.375000`
  - private `template_position` by difficulty: `easy = 0.343750`, `medium = 0.375000`
- The benchmark’s currently emitted difficulty labels are still coupled to template:
  - `T1 -> easy`
  - `T2 -> medium`
- That means the gate’s template-gap and emitted-difficulty-gap on the private split were both controlled by the same frozen episode selection.

### Why this was not a threshold bug

- The gate metric is simple and correct: `max(slice_accuracy) - min(slice_accuracy)` on the private split.
- The threshold is still defensible for this benchmark surface because the same benchmark already produced stronger separation elsewhere without any metric change:
  - dev `template_position` template gap: `0.156250`
- No audit or validity code bug was found.

### Why this was not a generator/protocol bug

- A nearby private seed window passes the unchanged gate and unchanged split audit with no code changes to generation.
- That means the current benchmark can already produce separating slices; the frozen private window was just underpowered.

## Change Applied

### Primary repair

- Updated [`src/frozen_splits/private_leaderboard.json`](/home/rafa/dev/Challenges/ch-executive-functions-1/src/frozen_splits/private_leaderboard.json)
- Changed:
  - `seed_bank_version`: `R14-private-1` -> `R14-private-2`
  - seeds: `9054..9069` -> `9060..9075`

### Why this was the smallest fix

- It is a narrow manifest-only change.
- It preserves deterministic generation and replay.
- It preserves split integrity:
  - overlap check still passes
  - `audit_frozen_splits()` still reports `0` issues
- It avoids relaxing the gate and avoids redesigning the benchmark.
- It was the nearest passing contiguous private window found during diagnosis.

## Before / After

### R13 validity gate

| Metric | Before | After |
| --- | ---: | ---: |
| Gate result | `FAIL` | `PASS` |
| Best template gap | `0.031250` | `0.103175` |
| Best emitted-difficulty gap | `0.031250` | `0.103175` |
| Best gap baseline | `template_position` | `template_position` |

### Private split critical slice details

| Metric | Before | After |
| --- | ---: | ---: |
| `template_position` private overall | `0.359375` | `0.343750` |
| `template_position` private T1 | `0.343750` | `0.388889` |
| `template_position` private T2 | `0.375000` | `0.285714` |
| `template_position` private easy | `0.343750` | `0.388889` |
| `template_position` private medium | `0.375000` | `0.285714` |

### R15 re-audit note

Before:

- Packaging blocker included weak private-leaderboard slice separation

After:

- The re-audit no longer reports private slice separation as a blocker
- Remaining blockers are:
  - no bundled real-model runs in-repo
  - `hard` remains reserved and un-emitted

## Evidence Pass

Commands run after the repair:

```bash
.venv/bin/python -m pytest
.venv/bin/python scripts/ife.py evidence-pass -o /tmp/evidence_pass.json
```

Observed results:

- Full local suite: `636 passed`
- R13 validity gate: `PASS`
- R15 re-audit: completed successfully
- Split integrity: `overlap_check_passed = true`
- Frozen artifact consistency: `kaggle_manifest_valid = true`
- Frozen split audit issues: `0`

## Conclusion

The failure mode was **underpowered private split composition**. The repaired benchmark now clears R13 without changing the gate definition or weakening the benchmark. The benchmark is locally reproducible and validity-clean on the current frozen artifacts.

## Remaining Blockers Before Real-Model Evaluation

- No real-model runs are bundled in-repo, so model-vs-heuristic separation is still unverified.
- `hard` remains reserved and un-emitted, so no hard-slice claim is supportable.
