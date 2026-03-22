# Iron Find Electric v1 Benchmark Card

## Summary

Iron Find Electric v1 is a small, frozen benchmark for **cognitive flexibility under hidden rule shift**. Each episode presents labeled evidence under one binary interaction rule, then switches rules mid-episode without announcing the switch, and finally asks the model to answer four post-shift probes.

This package is for Kaggle staging only. The implemented local benchmark under `src/`, the frozen split manifests under `src/frozen_splits/`, and the locally produced validation and audit evidence remain the source of truth.

## Task

- **Binary** is the leaderboard-primary task.
- **Narrative** is the robustness companion.
- The target construct is adaptation to a hidden rule change rather than recall of electrostatics facts or a single-step shortcut.

Each episode contains:

- 5 labeled items
- 4 unlabeled probes
- a pre-shift segment governed by `rule_A`
- a post-shift segment governed by `rule_B`

Current rule family:

- `R_std`: same-sign charges repel, opposite-sign charges attract
- `R_inv`: same-sign charges attract, opposite-sign charges repel

## Metric

The primary metric is **Post-shift Probe Accuracy**: the fraction of the four final probe labels that match the benchmark targets after the hidden rule shift.

Binary is the leaderboard-primary report. Narrative is staged as a robustness companion and should be reviewed alongside Binary, but it does not replace the primary Binary evaluation path.

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

- a reproducible Binary benchmark and Narrative companion over frozen split manifests
- a primary metric of Post-shift Probe Accuracy
- deterministic local replay from the stored seed banks
- local validity and audit evidence tied to the current repaired implementation

The benchmark explicitly does **not** claim:

- full executive-function decomposition
- online detection latency
- switch cost measurement
- emitted `hard` slices

## Current Evidence

### R13 anti-shortcut validity gate

The packaged anti-shortcut validity evidence is the local `R13` gate report. It still reports **FAIL** because the private leaderboard subset-separation requirement was not met: best template gap `0.031250` and best emitted-difficulty gap `0.031250`, both below the required `0.100000`.

### R15 empirical re-audit

The packaged empirical re-audit is the local `R15` report over the refreshed `R14` frozen splits. It reports that the **recency shortcut was materially reduced** relative to the earlier blocker surface: `last_evidence` peaks at `0.500000` on `public_leaderboard`, so recency no longer looks like the dominant shortcut failure mode in that report.

The same re-audit also says the benchmark is still limited by:

- weak private-leaderboard slice separation
- no bundled real-model runs in-repo
- `hard` remaining reserved and un-emitted

## Limitations

- The package does not bundle model predictions or Kaggle submission outputs.
- Narrative remains a robustness companion, not the primary leaderboard metric.
- No claim here upgrades the local evidence beyond the bundled R13 and R15 reports.
- Local validation remains authoritative if any staging notebook output diverges from the frozen evidence.
