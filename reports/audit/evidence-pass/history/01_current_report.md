# Current Evidence Pass Report

## Executive Summary

The current local Iron Find Electric benchmark is code-healthy and reproducible, but it is still blocked on benchmark-validity evidence. The full local automated suite passed (`629 passed, 0 failed, 0 skipped`), all frozen manifests and regression fixtures loaded cleanly, and the current R15 deterministic re-audit reproduced the bundled evidence exactly. The current R13 anti-shortcut gate still fails on one criterion: private-leaderboard subset separation remains too weak. `hard` is still reserved and not emitted.

## Scope and Execution

This pass used the existing implementation and frozen artifacts as-is.

Commands used:

```bash
.venv/bin/python -m pytest
PYTHONPATH=src .venv/bin/python - <<'PY'
from validate import run_benchmark_validity_report, R13_VALIDITY_GATE
from audit import run_release_r15_reaudit
from splits import load_all_frozen_splits, assert_no_partition_overlap, audit_frozen_splits
from kaggle import validate_kaggle_staging_manifest
PY
```

The raw `python` runs required `PYTHONPATH=src` because the repo is not installed as a package for direct interpreter entrypoints.

## 1. Code/Test Health

- Test suite result: `629 passed, 0 failed, 0 skipped` in `1.91s`
- Failing modules: none
- Flaky behavior observed in this pass: none

Because the suite was green, the evidence pass continued.

## 2. R13 Anti-Shortcut Validity Gate

### Gate Criteria Used

Release: `R13`

Evaluation splits:

- `public_leaderboard`
- `private_leaderboard`

Audit split:

- `private_leaderboard`

Critical baselines:

- `never_update`
- `last_evidence`
- `physics_prior`
- `template_position`

Exact criteria:

1. Each critical baseline must stay at or below `0.55` accuracy on `public_leaderboard` and `private_leaderboard`.
2. No critical baseline may dominate the evaluation splits above `0.55`.
3. On `private_leaderboard`, the best critical-baseline subset separation must show at least:
   - template gap `>= 0.10`
   - emitted-difficulty gap `>= 0.10`

### Gate Result

- Overall gate result: `FAIL`
- Blocking criterion: `heuristic_subset_separation`

### Criteria Passed

- `never_update_bounded`
- `last_evidence_bounded`
- `physics_prior_bounded`
- `template_position_bounded`
- `no_dominant_trivial_heuristic`

### Criteria Failed

- `heuristic_subset_separation`
  - Required on `private_leaderboard`: template gap `>= 0.10`, emitted-difficulty gap `>= 0.10`
  - Observed best template gap: `template_position = 0.031250`
  - Observed best emitted-difficulty gap: `template_position = 0.031250`

### Baseline Results Used by the Gate

Overall Post-shift Probe Accuracy:

| Baseline | Overall | Dev | Public | Private |
| --- | ---: | ---: | ---: | ---: |
| `random` | 0.526042 | 0.609375 | 0.484375 | 0.484375 |
| `never_update` | 0.500000 | 0.500000 | 0.500000 | 0.500000 |
| `last_evidence` | 0.500000 | 0.500000 | 0.500000 | 0.500000 |
| `physics_prior` | 0.500000 | 0.500000 | 0.500000 | 0.500000 |
| `template_position` | 0.458333 | 0.515625 | 0.500000 | 0.359375 |

Interpretation:

- The named shortcut baselines are materially bounded under the `0.55` cap.
- The gate still fails because the private split does not separate critical heuristics enough across template or emitted-difficulty slices.

## 3. R15 Empirical Re-Audit

### Inputs

- Release audit: `R15`
- Frozen manifests loaded from current local `R14` split manifests
- Episode counts: `16 dev`, `16 public_leaderboard`, `16 private_leaderboard`
- Difficulty labels present: `easy`, `medium`
- Difficulty labels missing: `hard`

### Overall Post-shift Probe Accuracy by Baseline

| Baseline | Accuracy | Parse-valid rate |
| --- | ---: | ---: |
| `random` | 0.526042 | 1.000000 |
| `never_update` | 0.500000 | 1.000000 |
| `last_evidence` | 0.500000 | 1.000000 |
| `physics_prior` | 0.500000 | 1.000000 |
| `template_position` | 0.458333 | 1.000000 |

### Scores by Split

| Baseline | Dev | Public | Private |
| --- | ---: | ---: | ---: |
| `random` | 0.609375 | 0.484375 | 0.484375 |
| `never_update` | 0.500000 | 0.500000 | 0.500000 |
| `last_evidence` | 0.500000 | 0.500000 | 0.500000 |
| `physics_prior` | 0.500000 | 0.500000 | 0.500000 |
| `template_position` | 0.515625 | 0.500000 | 0.359375 |

### Scores by Template

| Baseline | T1 | T2 |
| --- | ---: | ---: |
| `random` | 0.531250 | 0.520833 |
| `never_update` | 0.500000 | 0.500000 |
| `last_evidence` | 0.500000 | 0.500000 |
| `physics_prior` | 0.500000 | 0.500000 |
| `template_position` | 0.479167 | 0.437500 |

### Scores by Currently Emitted Difficulty Labels Only

`hard` is reserved and not emitted, so only `easy` and `medium` are auditable in the current build.

| Baseline | Easy | Medium |
| --- | ---: | ---: |
| `random` | 0.531250 | 0.520833 |
| `never_update` | 0.500000 | 0.500000 |
| `last_evidence` | 0.500000 | 0.500000 |
| `physics_prior` | 0.500000 | 0.500000 |
| `template_position` | 0.479167 | 0.437500 |

### Binary vs Narrative

- Available: no
- Matched Binary/Narrative comparisons: none
- Reason: no structured model runs were supplied, and no real-model adapters/runs are bundled or configured locally in-repo

### Parse-Valid Rate

- Deterministic baselines: `1.0` for every reported baseline and slice
- Model parse-valid rate: not applicable in this run because no model sources were provided

### Re-Audit Conclusion

- `last_evidence` peaks at `0.500000`, below the `0.55` shortcut cap
- Recency no longer appears to be the dominant shortcut threat in the current frozen benchmark
- The benchmark is still blocked because private-leaderboard slice separation is weak and there is no local real-model evidence

## 4. Frozen Artifact Integrity

### Frozen Split Load and Overlap

- `dev` manifest loaded: yes, `16/16` episodes regenerated successfully
- `public_leaderboard` manifest loaded: yes, `16/16` episodes regenerated successfully
- `private_leaderboard` manifest loaded: yes, `16/16` episodes regenerated successfully
- Cross-partition overlap check: passed
- `audit_frozen_splits()` issues: `0`

Current frozen manifest summary:

| Partition | Manifest version | Seed bank version | Episode split | Episode count |
| --- | --- | --- | --- | ---: |
| `dev` | `R14` | `R14-dev-1` | `dev` | 16 |
| `public_leaderboard` | `R14` | `R14-public-1` | `public` | 16 |
| `private_leaderboard` | `R14` | `R14-private-1` | `private` | 16 |

### Fixture and Regression Artifact Usability

- `tests/fixtures/validation_regression.json`: present and usable
- `tests/fixtures/release_r13_validity_report.json`: present and matches current local R13 output exactly
- `tests/fixtures/release_r15_reaudit_report.json`: present and matches current local R15 output exactly

### Manifest / Artifact / Docs Consistency

- `validate_kaggle_staging_manifest()` passed
- No mismatch was found between:
  - current frozen manifests
  - packaged artifact hashes
  - bundled R13/R15 evidence fixtures
  - current benchmark/docs claims about bounded shortcuts, missing real-model evidence, and non-emitted `hard`

## 5. Final Assessment

### What is the current benchmark state?

The implementation is stable and reproducible. Frozen assets load correctly and reproduce the bundled evidence. The benchmark remains validity-blocked, not code-blocked.

### Do all local tests pass?

Yes. `629 passed, 0 failed, 0 skipped`.

### Does the benchmark pass the anti-shortcut validity gate?

No. It fails the current R13 gate on private-leaderboard subset separation only.

### Is the recency shortcut still a serious threat?

Not by the current local evidence. `last_evidence` stays at `0.500000` on both leaderboard splits and is not the dominant heuristic. The broader shortcut-resistance story is still incomplete because slice separation remains weak.

### Which slices are weakest?

- Weakest split-level baseline slice: `template_position` on `private_leaderboard` at `0.359375`
- Weakest template slice overall: `template_position` on `T2` at `0.437500`
- Weakest emitted-difficulty slice overall: `template_position` on `medium` at `0.437500`
- Most important blocker slice: private-leaderboard template and emitted-difficulty separation, both only `0.031250` at best versus the required `0.100000`

### Are currently emitted difficulty labels behaving as expected?

Yes, for the currently emitted labels. Only `easy` and `medium` are present, and all current audit outputs are internally consistent with that. `hard` is still reserved and not emitted.

### Is the benchmark ready for Kaggle staging, or still blocked?

Still blocked.

Blocking reasons:

1. The current benchmark fails the R13 anti-shortcut validity gate.
2. Private-leaderboard subset separation is below threshold.
3. No real-model runs/adapters are bundled or configured locally, so model-vs-heuristic separation is unverified.
4. `hard` remains reserved and un-emitted, so no hard-slice claim is supportable.

### Exact Recommended Next Steps

1. Strengthen the frozen benchmark so the private-leaderboard critical heuristics achieve at least `0.10` template gap and `0.10` emitted-difficulty gap without letting any critical shortcut baseline exceed `0.55` on the leaderboard splits.
2. Re-run the exact same R13 gate and R15 re-audit after that change and refresh the frozen evidence artifacts only if the outputs change.
3. Add at least one locally runnable real-model evaluation path, or explicitly stage without model evidence and keep the benchmark card conservative about that limitation.
4. Continue to state clearly that `hard` is reserved and not emitted until the generator actually produces it and the frozen artifacts are refreshed.

## Bottom Line

Current local status: tests pass, artifacts are intact, recency is bounded, but the benchmark is not yet staging-ready because the anti-shortcut validity gate still fails on private-leaderboard subset separation and there is no local real-model evidence.
