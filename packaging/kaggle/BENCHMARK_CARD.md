# Iron Find Electric v1 Benchmark Card

## Summary

Iron Find Electric v1 is a narrow Executive Functions benchmark for cognitive flexibility. It uses electrostatics only as a controlled substrate for evaluating final post-shift rule application after sparse contradictory evidence.

A high v1 Binary score is evidence that a model correctly applied the post-shift rule to the final probes after sparse contradictory evidence in the frozen episodes. It is not evidence of physics skill, broad adaptation ability, or general reasoning ability.

This package is for Kaggle staging only. The implemented local benchmark under `src/`, the frozen split manifests under `src/frozen_splits/`, and the locally produced validation and audit evidence remain the source of truth.

## Task

- **Binary** is the only leaderboard-primary task. It is the scored evaluation path for the v1 claim.
- **Narrative** is required non-leaderboard robustness evidence. It uses the same frozen episodes and probe targets as Binary, and only the final four labels are scored.
- Electrostatics is only the controlled substrate. The benchmark is not intended to measure physics skill as the primary target.

Each episode contains:

- 5 labeled items
- 4 unlabeled probes
- a pre-shift segment governed by `rule_A`
- a post-shift segment governed by `rule_B`

Current rule family:

- `R_std`: same-sign charges repel, opposite-sign charges attract
- `R_inv`: same-sign charges attract, opposite-sign charges repel

## Metric

The primary metric is **Post-shift Probe Accuracy**: the fraction of final post-shift probe labels answered correctly under the post-shift rule in the Binary task.

Binary is the only headline report. Narrative is reviewed only as same-episode robustness evidence and does not change the headline score.

## Baselines

Current baseline references carried into the package:

- `random`
- `never_update`
- `last_evidence`
- `physics_prior`
- `template_position`

These are the benchmark sanity-check baselines used by the local validity and re-audit workflow. They are included to frame shortcut risk, not to claim any model leaderboard result.

## Frozen Artifacts And Reproducibility

This Kaggle package references frozen local artifacts rather than regenerating a new benchmark design:

- split manifests: `src/frozen_splits/dev.json`, `src/frozen_splits/public_leaderboard.json`, `src/frozen_splits/private_leaderboard.json`
- anti-shortcut gate evidence: `tests/fixtures/release_r13_validity_report.json`
- empirical re-audit evidence: `tests/fixtures/release_r15_reaudit_report.json`
- bundle index and integrity hashes: `packaging/kaggle/frozen_artifacts_manifest.json`

Version metadata currently frozen by the package:

- split manifest version: `R14`
- spec version: `v1`
- generator version: `R12`
- template set version: `v1`
- difficulty version: `R12`

## Current Implementation State

Current emitted difficulty labels are `easy` and `medium`. `hard` is reserved and not emitted by the current implementation, so no packaging claim depends on emitted `hard` slices.

The benchmark currently claims:

- a reproducible Binary benchmark and Narrative companion over the same frozen split manifests and probe targets
- a Binary-only primary metric of Post-shift Probe Accuracy
- deterministic local replay from the stored seed banks
- local validity and audit evidence tied to the current repaired implementation

The benchmark explicitly does **not** claim:

- physics skill as the primary measured ability
- full executive-function decomposition
- broad adaptation ability
- broad AGI capability
- general reasoning ability
- online detection latency
- switch cost measurement
- recovery length
- immediate post-shift drop
- emitted `hard` slices

## Current Evidence

### R13 anti-shortcut validity gate

The packaged anti-shortcut validity evidence is the local `R13` gate report. It now reports **PASS**: the best private-leaderboard critical-baseline subset gaps are `0.103175` for both template and emitted difficulty, above the required `0.100000`.

### R15 empirical re-audit

The packaged empirical re-audit is the local `R15` report over the refreshed `R14` frozen splits. It reports that the **recency shortcut was materially reduced** relative to the earlier blocker surface: `last_evidence` peaks at `0.500000` on `public_leaderboard`, so recency no longer looks like the dominant shortcut failure mode in that report.

The same re-audit also says the benchmark is still limited by:

- `hard` remaining reserved and un-emitted

### M1 live Gemini panel evidence

The current M1 live Gemini panel (gemini-2.5-flash, R18) provides the first real-model evidence:

- Binary accuracy: 0.781250 (vs best baseline random = 0.546875)
- Narrative accuracy: 0.458333
- Binary → Narrative delta: 0.322917
- Binary parse-valid: 1.000000
- Narrative parse-valid: 0.937500

Binary substantially exceeds all heuristic baselines. Narrative is meaningfully lower than Binary on the same frozen episodes, indicating a real surface-form robustness gap. A small Narrative provider/runtime contamination note (overall rate = 0.041667) was observed in the live run and must be disclosed separately from parse/format and adaptation outcomes.

### M2 staging dry-run readiness

M2 confirms that packaged frozen artifacts load, manifest validation passes, and the staging notebook completes end to end in both Binary and Narrative modes. M2 is packaging-validation evidence only, not live model-evaluation evidence.

## Limitations

- The package does not bundle model predictions or Kaggle submission outputs.
- M1 live Gemini panel evidence exists in the reports tree but is not part of the Kaggle staging package itself.
- Narrative is not a primary leaderboard task, and only the final four labels are scored.
- No claim should depend on explanation quality or formatting compliance.
- No claim here upgrades the local evidence beyond the bundled R13 and R15 reports.
- Local validation remains authoritative if any staging notebook output diverges from the frozen evidence.
