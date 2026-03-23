# RuleShift Benchmark v1 Benchmark Card

> **Status: SUPPORTING SUMMARY**
> This benchmark card is descriptive only.
> For binding benchmark terms, use [`../../KAGGLE_BENCHMARK_CONTRACT.md`](../../KAGGLE_BENCHMARK_CONTRACT.md).
> For Kaggle submission and staging steps, use [`README.md`](./README.md).

## Summary

RuleShift Benchmark v1 is a narrow Executive Functions benchmark for cognitive flexibility. It uses electrostatics only as a controlled substrate for evaluating final post-shift rule application after sparse contradictory evidence.

A high v1 Binary score is evidence that a model correctly applied the post-shift rule to the final probes after sparse contradictory evidence in the frozen episodes. It is not evidence of physics skill, broad adaptation ability, broad AGI capability, or general reasoning ability.

This package is a Kaggle packaging layer over the implemented local benchmark. The authoritative contract is `KAGGLE_BENCHMARK_CONTRACT.md` at the repository root. The implemented local benchmark under `src/` and the frozen split manifests under `src/frozen_splits/` remain the runtime source of truth. The single official Kaggle leaderboard notebook is `packaging/kaggle/ruleshift_benchmark_v1_kbench.ipynb`; `packaging/kaggle/staging/ruleshift_benchmark_v1_kaggle_staging.ipynb` is staging-only.

## Task Paths

- **Binary** is the only leaderboard-primary path. The Binary task is the scored evaluation path for the v1 claim.
- **Narrative** is the required same-episode robustness companion. It uses the same frozen episodes and probe targets as Binary, and only the final four labels are scored.
- Electrostatics is only the controlled substrate. The benchmark is not intended to measure physics skill as the primary target.

Each episode contains:

- 5 labeled items
- 4 unlabeled probes
- a pre-shift segment governed by `rule_A`
- a post-shift segment governed by `rule_B`

Current rule family:

- `R_std`: same-sign charges repel, opposite-sign charges attract
- `R_inv`: same-sign charges attract, opposite-sign charges repel

## Headline Metric

The sole headline metric is **Post-shift Probe Accuracy**: the fraction of final post-shift probe labels answered correctly under the post-shift rule in the Binary path.

Narrative is reviewed only as same-episode robustness evidence and does not change the headline score.

## Split Contract

Frozen split names are exactly:

- `dev`
- `public_leaderboard`
- `private_leaderboard`

## Baselines

Current baseline references carried into the package:

- `random`
- `never_update`
- `last_evidence`
- `physics_prior`
- `template_position`

These are benchmark sanity-check baselines used by the local validity and re-audit workflow. They are included to frame shortcut risk, not to claim any model leaderboard result.

## Frozen Artifacts And Reproducibility

This Kaggle package references frozen local artifacts rather than regenerating a new benchmark:

- split manifests: `src/frozen_splits/dev.json`, `src/frozen_splits/public_leaderboard.json`, `src/frozen_splits/private_leaderboard.json`
- anti-shortcut gate evidence: `tests/fixtures/release_r13_validity_report.json`
- empirical re-audit evidence: `tests/fixtures/release_r15_reaudit_report.json`
- single packaged Gemini readiness anchor: `reports/m1_binary_vs_narrative_robustness_report.md`
- Kaggle runtime-contract manifest and integrity hashes: `packaging/kaggle/frozen_artifacts_manifest.json`

Version metadata currently frozen by the package:

- split manifest version: `R14`
- spec version: `v1`
- generator version: `R12`
- template set version: `v1`
- difficulty version: `R12`

## Current Implementation State

Current emitted difficulty labels are `easy` and `medium`. `hard` is reserved and not emitted by the current implementation, so no benchmark claim depends on emitted `hard` slices.

The benchmark currently claims:

- a reproducible Binary benchmark and Narrative companion over the same frozen split manifests and probe targets
- Binary as the only leaderboard-primary path
- Post-shift Probe Accuracy as the sole headline metric
- deterministic local replay from the stored seed banks
- local validity and audit evidence tied to the current implemented benchmark

The benchmark explicitly does **not** claim:

- physics skill as the primary measured ability
- full executive-function decomposition
- broad adaptation ability
- broad AGI capability
- general reasoning ability
- human-level performance
- cross-provider readiness or cross-provider equivalence
- online detection latency
- switch cost measurement
- recovery length
- immediate post-shift drop
- emitted `hard` slices

## Current Readiness Status

- the active v1 readiness evidence path is Gemini;
- the single current packaged readiness anchor is `reports/m1_binary_vs_narrative_robustness_report.md`;
- that anchor is synced to the committed paired report preserving the requested model label `gemini-2.5-flash` at `reports/live/gemini-first-panel/binary-vs-narrative/history/report__20260323_120000.md`;
- the paired Gemini Flash-Lite `latest/` run and the direct Flash vs Flash-Lite comparison are preserved as supporting comparison material, not as a second active readiness anchor;
- Anthropic and OpenAI integrations already exist locally, but they are outside the current v1 readiness gate;
- current v1 readiness does not require cross-provider evidence.

## Deferred Work Boundary

Current v1 readiness remains the Gemini-only gate above. The following work is preserved but deferred beyond the current v1 readiness decision and beyond the current Kaggle staging claim:

- post-v1 empirical expansion: Anthropic live evidence, OpenAI live evidence, cross-provider comparison, and broader run-store expansion beyond the current provenance contract
- longer-term scientific-validity strengthening: human pilot, independent rerun, and protocol extensions needed for adaptation-lag or recovery claims

Anthropic and OpenAI integrations remain available as local-only in-repo execution surfaces. They are preserved assets for later empirical expansion, not blockers for the current v1 package.

## Current Evidence

### R13 anti-shortcut validity gate

The packaged anti-shortcut validity evidence is the local `R13` gate report. It reports **PASS**: the best `private_leaderboard` critical-baseline subset gaps are `0.103175` for both template and emitted difficulty, above the required `0.100000`.

### R15 empirical re-audit

The packaged empirical re-audit is the local `R15` report over the refreshed `R14` frozen splits. It reports that the **recency shortcut was materially reduced** relative to the earlier blocker surface: `last_evidence` peaks at `0.500000` on `public_leaderboard`, so recency no longer looks like the dominant shortcut failure mode in that report.

The same re-audit also says the benchmark is still limited by:

- `hard` remaining reserved and un-emitted

### M1 live Gemini evidence

The single current packaged readiness anchor is `reports/m1_binary_vs_narrative_robustness_report.md`, synced to the committed M1 Gemini panel report for requested model label `gemini-2.5-flash` at `reports/live/gemini-first-panel/binary-vs-narrative/history/report__20260323_120000.md`. This committed report was resynced in M6 from the original legacy capture; current local runners now require pinned model IDs, but the legacy capture did not record provider-served model-version, token-usage, or duration fields.

- Binary accuracy: 0.781250
- Narrative accuracy: 0.458333
- Binary -> Narrative delta: 0.322917
- Binary parse-valid: 1.000000
- Narrative parse-valid: 0.937500

Binary substantially exceeds all heuristic baselines. Narrative is meaningfully lower than Binary on the same frozen episodes, indicating a real surface-form robustness gap. A small Narrative provider/runtime contamination note (overall rate = 0.041667) was observed in the live run and must be disclosed separately from parse/format and adaptation outcomes.

### M2 staging dry-run readiness

M2 confirms that packaged frozen artifacts load, manifest validation passes, and the staging notebook completes end to end in both Binary and Narrative modes. M2 is packaging-validation evidence only, not live model-evaluation evidence.

## Limitations

- The package does not bundle model predictions or Kaggle submission outputs.
- Current readiness evidence is Gemini-only; the package does not require Anthropic or OpenAI evidence for v1 readiness.
- Narrative is not leaderboard-primary, and only the final four labels are scored.
- No claim should depend on explanation quality or formatting compliance.
- No claim here upgrades the local evidence beyond the bundled R13, R15, and committed Gemini evidence surfaces.
- Local validation remains authoritative if any staging notebook output diverges from the frozen evidence.
