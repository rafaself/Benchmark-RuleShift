# Gemini First Panel Report

> **Status: SUPPORTING LIVE REPORT MIRROR**
> This file mirrors the current public paired Gemini live report at `reports/live/gemini-first-panel/binary-vs-narrative/latest/report.md`.
> It is a preserved pre-hardening live-evidence artifact, not the source of truth for benchmark scope, leaderboard role, current emitted difficulty labels, or the current Narrative audit contract.

- Release: R18
- Provider: gemini
- Model: gemini-2.5-flash-lite
- Prompt modes run: binary, narrative
- Covered splits: dev, public_leaderboard, private_leaderboard

## Contract Hardening Note

- The tables below reflect the older Narrative contract that used long JSON fields and suffered substantial parse/format failure.
- The local benchmark now hardens Narrative into a short 4-line audit contract:
  - `rule_before: ...`
  - `shift_evidence: ...`
  - `rule_after: ...`
  - `final_decision: attract, repel, repel, attract`
- Binary remains the only leaderboard-primary path and headline metric.
- Narrative remains supplemental audit evidence only.
- The updated codebase now supports exact Binary/Narrative decision-agreement as an audit-consistency diagnostic, but this preserved report was not regenerated from a fresh live rerun and therefore does not claim improved post-hardening rates.

## Headline

- Binary-only headline metric: gemini-2.5-flash-lite Binary = 0.687500
- Binary parse-valid rate: 0.958333
- Best baseline: random (0.546875)

| Source | Accuracy | Gap vs model |
| --- | ---: | ---: |
| gemini-2.5-flash-lite Binary | 0.687500 | 0.000000 |
| random | 0.546875 | 0.140625 |
| never_update | 0.500000 | 0.187500 |
| last_evidence | 0.500000 | 0.187500 |
| physics_prior | 0.500000 | 0.187500 |
| template_position | 0.453125 | 0.234375 |

## Headline By Split

| Split | Model | random | never-update | last-evidence | physics-prior | template-position |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dev | 0.671875 | 0.609375 | 0.500000 | 0.500000 | 0.500000 | 0.515625 |
| public_leaderboard | 0.625000 | 0.484375 | 0.500000 | 0.500000 | 0.500000 | 0.500000 |
| private_leaderboard | 0.765625 | 0.546875 | 0.500000 | 0.500000 | 0.500000 | 0.343750 |

## Paired Robustness

| Scope | Binary accuracy | Narrative accuracy | Delta | Binary parse-valid | Narrative parse-valid |
| --- | ---: | ---: | ---: | ---: | ---: |
| overall | 0.687500 | 0.276042 | 0.411458 | 0.958333 | 0.520833 |
| dev | 0.671875 | 0.312500 | 0.359375 | 0.937500 | 0.750000 |
| public_leaderboard | 0.625000 | 0.281250 | 0.343750 | 0.937500 | 0.500000 |
| private_leaderboard | 0.765625 | 0.234375 | 0.531250 | 1.000000 | 0.312500 |

## Diagnostic Slices

| Slice type | Label | Binary accuracy | Narrative accuracy | Delta | Binary parse-valid | Narrative parse-valid |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| template | T1 | 0.690000 | 0.150000 | 0.540000 | 0.920000 | 0.360000 |
| template | T2 | 0.684783 | 0.413043 | 0.271739 | 1.000000 | 0.695652 |
| difficulty | easy | 0.690000 | 0.150000 | 0.540000 | 0.920000 | 0.360000 |
| difficulty | medium | 0.684783 | 0.413043 | 0.271739 | 1.000000 | 0.695652 |

## Failure Decomposition (diagnostic-only)

| Scope | Mode | Runtime | Parse/format | Parse-valid | Correct | Adaptation | Adaptation among parse-valid |
| --- | --- | --- | --- | --- | --- | --- | --- |
| overall | Binary | 2/48 (0.041667) | 0/48 (0.000000) | 46/48 (0.958333) | 11/48 (0.229167) | 35/48 (0.729167) | 35/46 (0.760870) |
| overall | Narrative | 9/48 (0.187500) | 14/48 (0.291667) | 25/48 (0.520833) | 2/48 (0.041667) | 23/48 (0.479167) | 23/25 (0.920000) |
| dev | Binary | 1/16 (0.062500) | 0/16 (0.000000) | 15/16 (0.937500) | 4/16 (0.250000) | 11/16 (0.687500) | 11/15 (0.733333) |
| dev | Narrative | 1/16 (0.062500) | 3/16 (0.187500) | 12/16 (0.750000) | 0/16 (0.000000) | 12/16 (0.750000) | 12/12 (1.000000) |
| public_leaderboard | Binary | 1/16 (0.062500) | 0/16 (0.000000) | 15/16 (0.937500) | 1/16 (0.062500) | 14/16 (0.875000) | 14/15 (0.933333) |
| public_leaderboard | Narrative | 4/16 (0.250000) | 4/16 (0.250000) | 8/16 (0.500000) | 1/16 (0.062500) | 7/16 (0.437500) | 7/8 (0.875000) |
| private_leaderboard | Binary | 0/16 (0.000000) | 0/16 (0.000000) | 16/16 (1.000000) | 6/16 (0.375000) | 10/16 (0.625000) | 10/16 (0.625000) |
| private_leaderboard | Narrative | 4/16 (0.250000) | 7/16 (0.437500) | 5/16 (0.312500) | 1/16 (0.062500) | 4/16 (0.250000) | 4/5 (0.800000) |

## Direct Disagreement Diagnostics (diagnostic-only)

| Scope | Mode | Exact global old-rule | Exact global recency | Old-rule-only episodes | Recency-only episodes | Mixed episodes | Old-rule error probes | Recency error probes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | Binary | 5/35 (0.142857) | 2/35 (0.057143) | 12/35 (0.342857) | 15/35 (0.428571) | 8/35 (0.228571) | 26/52 (0.500000) | 26/52 (0.500000) |
| overall | Narrative | 5/23 (0.217391) | 7/23 (0.304348) | 7/23 (0.304348) | 9/23 (0.391304) | 7/23 (0.304348) | 21/47 (0.446809) | 26/47 (0.553191) |
| dev | Binary | 2/11 (0.181818) | 1/11 (0.090909) | 4/11 (0.363636) | 4/11 (0.363636) | 3/11 (0.272727) | 9/17 (0.529412) | 8/17 (0.470588) |
| dev | Narrative | 2/12 (0.166667) | 4/12 (0.333333) | 2/12 (0.166667) | 4/12 (0.333333) | 6/12 (0.500000) | 11/28 (0.392857) | 17/28 (0.607143) |
| public_leaderboard | Binary | 3/14 (0.214286) | 1/14 (0.071429) | 6/14 (0.428571) | 6/14 (0.428571) | 2/14 (0.142857) | 11/20 (0.550000) | 9/20 (0.450000) |
| public_leaderboard | Narrative | 3/7 (0.428571) | 2/7 (0.285714) | 3/7 (0.428571) | 3/7 (0.428571) | 1/7 (0.142857) | 8/14 (0.571429) | 6/14 (0.428571) |
| private_leaderboard | Binary | 0/10 (0.000000) | 0/10 (0.000000) | 2/10 (0.200000) | 5/10 (0.500000) | 3/10 (0.300000) | 6/15 (0.400000) | 9/15 (0.600000) |
| private_leaderboard | Narrative | 0/4 (0.000000) | 1/4 (0.250000) | 2/4 (0.500000) | 2/4 (0.500000) | 0/4 (0.000000) | 2/5 (0.400000) | 3/5 (0.600000) |

Episode cells in this table are normalized by adaptation-failure episodes. Probe cells are normalized by wrong probes inside parse-valid adaptation failures.

## Failure Taxonomy (diagnostic-only)

| Scope | Mode | Provider/runtime error rate | Parse/format failure rate | Adaptation failure rate | Possible old-rule persistence rate | Possible recency overshoot rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| overall | Binary | 0.041667 | 0.000000 | 0.729167 | 0.104167 | 0.041667 |
| overall | Narrative | 0.187500 | 0.291667 | 0.479167 | 0.104167 | 0.145833 |
| dev | Binary | 0.062500 | 0.000000 | 0.687500 | 0.125000 | 0.062500 |
| dev | Narrative | 0.062500 | 0.187500 | 0.750000 | 0.125000 | 0.250000 |
| public_leaderboard | Binary | 0.062500 | 0.000000 | 0.875000 | 0.187500 | 0.062500 |
| public_leaderboard | Narrative | 0.250000 | 0.250000 | 0.437500 | 0.187500 | 0.125000 |
| private_leaderboard | Binary | 0.000000 | 0.000000 | 0.625000 | 0.000000 | 0.000000 |
| private_leaderboard | Narrative | 0.250000 | 0.437500 | 0.250000 | 0.000000 | 0.062500 |

Taxonomy rates are episode-level over scored outputs. Persistence and recency tags are diagnostic-only exact-match comparisons against `never_update` and `last_evidence`.

## Live Execution Review

Provider/runtime failures were observed in the live run. Review them separately from true parse/format failures before drawing benchmark conclusions.
At least one prompt mode has parse-valid outputs that still miss post-shift probes. Those misses are diagnostic-only adaptation evidence, not new benchmark scoring.

## Diagnostic Failure Slices (diagnostic-only)

| Slice type | Label | Mode | Episodes | Accuracy | Parse-valid | Adaptation among parse-valid | Old-rule error probes | Recency error probes |
| --- | --- | --- | --- | ---: | ---: | --- | --- | --- |
| template | T1 | Binary | 25 | 0.240000 | 0.920000 | 17/23 (0.739130) | 6/23 (0.260870) | 17/23 (0.739130) |
| template | T1 | Narrative | 25 | 0.000000 | 0.360000 | 9/9 (1.000000) | 8/21 (0.380952) | 13/21 (0.619048) |
| template | T2 | Binary | 23 | 0.217391 | 1.000000 | 18/23 (0.782609) | 20/29 (0.689655) | 9/29 (0.310345) |
| template | T2 | Narrative | 23 | 0.086957 | 0.695652 | 14/16 (0.875000) | 13/26 (0.500000) | 13/26 (0.500000) |
| difficulty | easy | Binary | 25 | 0.240000 | 0.920000 | 17/23 (0.739130) | 6/23 (0.260870) | 17/23 (0.739130) |
| difficulty | easy | Narrative | 25 | 0.000000 | 0.360000 | 9/9 (1.000000) | 8/21 (0.380952) | 13/21 (0.619048) |
| difficulty | medium | Binary | 23 | 0.217391 | 1.000000 | 18/23 (0.782609) | 20/29 (0.689655) | 9/29 (0.310345) |
| difficulty | medium | Narrative | 23 | 0.086957 | 0.695652 | 14/16 (0.875000) | 13/26 (0.500000) | 13/26 (0.500000) |
| transition | R_std_to_R_inv | Binary | 24 | 0.291667 | 0.958333 | 16/23 (0.695652) | 12/26 (0.461538) | 14/26 (0.538462) |
| transition | R_std_to_R_inv | Narrative | 24 | 0.041667 | 0.625000 | 14/15 (0.933333) | 18/28 (0.642857) | 10/28 (0.357143) |
| transition | R_inv_to_R_std | Binary | 24 | 0.166667 | 0.958333 | 19/23 (0.826087) | 14/26 (0.538462) | 12/26 (0.461538) |
| transition | R_inv_to_R_std | Narrative | 24 | 0.041667 | 0.416667 | 9/10 (0.900000) | 3/19 (0.157895) | 16/19 (0.842105) |

All views in this section are diagnostic-only. They use the frozen probe metadata already bundled with each episode and do not replace the Binary-only headline metric.

## Execution Provenance (diagnostic-only)

| Scope | Mode | Completed | Provider failures | Provider model versions | Mean duration (s) | Usage rows | Total tokens | Finish reasons |
| --- | --- | --- | --- | --- | ---: | --- | ---: | --- |
| overall | Binary | 46/48 (0.958333) | 2/48 (0.041667) | gemini-2.5-flash-lite | 1.106778 | 46/48 (0.958333) | 12852 | FinishReason.STOP=46 |
| overall | Narrative | 39/48 (0.812500) | 9/48 (0.187500) | gemini-2.5-flash-lite | 6.143193 | 39/48 (0.812500) | 47902 | FinishReason.STOP=39 |
| dev | Binary | 15/16 (0.937500) | 1/16 (0.062500) | gemini-2.5-flash-lite | 1.172060 | 15/16 (0.937500) | 4194 | FinishReason.STOP=15 |
| dev | Narrative | 15/16 (0.937500) | 1/16 (0.062500) | gemini-2.5-flash-lite | 8.459749 | 15/16 (0.937500) | 22357 | FinishReason.STOP=15 |
| private_leaderboard | Binary | 16/16 (1.000000) | 0/16 (0.000000) | gemini-2.5-flash-lite | 1.186689 | 16/16 (1.000000) | 4464 | FinishReason.STOP=16 |
| private_leaderboard | Narrative | 12/16 (0.750000) | 4/16 (0.250000) | gemini-2.5-flash-lite | 6.678454 | 12/16 (0.750000) | 12445 | FinishReason.STOP=12 |
| public_leaderboard | Binary | 15/16 (0.937500) | 1/16 (0.062500) | gemini-2.5-flash-lite | 0.961586 | 15/16 (0.937500) | 4194 | FinishReason.STOP=15 |
| public_leaderboard | Narrative | 12/16 (0.750000) | 4/16 (0.250000) | gemini-2.5-flash-lite | 3.291375 | 12/16 (0.750000) | 13100 | FinishReason.STOP=12 |

Execution provenance is diagnostic-only operational context. It does not change scoring or add a new benchmark metric.

## Binary Diagnostic Slices

| Template | Accuracy | Parse-valid rate |
| --- | ---: | ---: |
| T1 | 0.690000 | 0.920000 |
| T2 | 0.684783 | 1.000000 |

## Binary Difficulty Slices

| Difficulty | Accuracy | Parse-valid rate |
| --- | ---: | ---: |
| easy | 0.690000 | 0.920000 |
| medium | 0.684783 | 1.000000 |

## Notes

- This preserved run publishes difficulty tables for `easy` and `medium` only. Current benchmark manifests and bundled audit fixtures expose `easy`, `medium`, and `hard`; use the root `README.md`, `packaging/kaggle/BENCHMARK_CARD.md`, or `packaging/kaggle/frozen_artifacts_manifest.json` for current benchmark-state statements.
- Narrative remains required non-leaderboard robustness evidence on the same frozen episodes and probe targets as Binary; only the final four labels are scored, and it does not replace the primary Binary post-shift probe audit.
- Before hardening, Narrative failures were dominated by runtime plus parse/format issues rather than adaptation alone. The new 4-line contract is intended to make Narrative easier to validate automatically and more reliable as a Binary audit layer without changing leaderboard scoring.
