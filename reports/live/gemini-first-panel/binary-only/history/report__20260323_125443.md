# Gemini First Panel Report

- Release: R18
- Provider: gemini
- Model: gemini-2.5-flash
- Prompt modes run: binary
- Covered splits: dev, public_leaderboard, private_leaderboard

## Headline

- Binary-only headline metric: gemini-2.5-flash Binary = 0.781250
- Binary parse-valid rate: 1.000000
- Best baseline: random (0.546875)

| Source | Accuracy | Gap vs model |
| --- | ---: | ---: |
| gemini-2.5-flash Binary | 0.781250 | 0.000000 |
| random | 0.546875 | 0.234375 |
| never_update | 0.500000 | 0.281250 |
| last_evidence | 0.500000 | 0.281250 |
| physics_prior | 0.500000 | 0.281250 |
| template_position | 0.453125 | 0.328125 |

## Headline By Split

| Split | Model | random | never-update | last-evidence | physics-prior | template-position |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dev | 0.781250 | 0.609375 | 0.500000 | 0.500000 | 0.500000 | 0.515625 |
| public_leaderboard | 0.718750 | 0.484375 | 0.500000 | 0.500000 | 0.500000 | 0.500000 |
| private_leaderboard | 0.843750 | 0.546875 | 0.500000 | 0.500000 | 0.500000 | 0.343750 |

## Failure Decomposition (diagnostic-only)

| Scope | Mode | Runtime | Parse/format | Parse-valid | Correct | Adaptation | Adaptation among parse-valid |
| --- | --- | --- | --- | --- | --- | --- | --- |
| overall | Binary | 0/48 (0.000000) | 0/48 (0.000000) | 48/48 (1.000000) | 23/48 (0.479167) | 25/48 (0.520833) | 25/48 (0.520833) |
| dev | Binary | 0/16 (0.000000) | 0/16 (0.000000) | 16/16 (1.000000) | 8/16 (0.500000) | 8/16 (0.500000) | 8/16 (0.500000) |
| public_leaderboard | Binary | 0/16 (0.000000) | 0/16 (0.000000) | 16/16 (1.000000) | 5/16 (0.312500) | 11/16 (0.687500) | 11/16 (0.687500) |
| private_leaderboard | Binary | 0/16 (0.000000) | 0/16 (0.000000) | 16/16 (1.000000) | 10/16 (0.625000) | 6/16 (0.375000) | 6/16 (0.375000) |

## Direct Disagreement Diagnostics (diagnostic-only)

| Scope | Mode | Exact global old-rule | Exact global recency | Old-rule-only episodes | Recency-only episodes | Mixed episodes | Old-rule error probes | Recency error probes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | Binary | 0/25 (0.000000) | 6/25 (0.240000) | 2/25 (0.080000) | 14/25 (0.560000) | 9/25 (0.360000) | 13/42 (0.309524) | 29/42 (0.690476) |
| dev | Binary | 0/8 (0.000000) | 1/8 (0.125000) | 1/8 (0.125000) | 3/8 (0.375000) | 4/8 (0.500000) | 6/14 (0.428571) | 8/14 (0.571429) |
| public_leaderboard | Binary | 0/11 (0.000000) | 4/11 (0.363636) | 1/11 (0.090909) | 7/11 (0.636364) | 3/11 (0.272727) | 4/18 (0.222222) | 14/18 (0.777778) |
| private_leaderboard | Binary | 0/6 (0.000000) | 1/6 (0.166667) | 0/6 (0.000000) | 4/6 (0.666667) | 2/6 (0.333333) | 3/10 (0.300000) | 7/10 (0.700000) |

Episode cells in this table are normalized by adaptation-failure episodes. Probe cells are normalized by wrong probes inside parse-valid adaptation failures.

## Live Execution Review

At least one prompt mode has parse-valid outputs that still miss post-shift probes. Those misses are diagnostic-only adaptation evidence, not new benchmark scoring.

## Diagnostic Failure Slices (diagnostic-only)

| Slice type | Label | Mode | Episodes | Accuracy | Parse-valid | Adaptation among parse-valid | Old-rule error probes | Recency error probes |
| --- | --- | --- | --- | ---: | ---: | --- | --- | --- |
| template | T1 | Binary | 25 | 0.440000 | 1.000000 | 14/25 (0.560000) | 7/21 (0.333333) | 14/21 (0.666667) |
| template | T2 | Binary | 23 | 0.521739 | 1.000000 | 11/23 (0.478261) | 6/21 (0.285714) | 15/21 (0.714286) |
| difficulty | easy | Binary | 25 | 0.440000 | 1.000000 | 14/25 (0.560000) | 7/21 (0.333333) | 14/21 (0.666667) |
| difficulty | medium | Binary | 23 | 0.521739 | 1.000000 | 11/23 (0.478261) | 6/21 (0.285714) | 15/21 (0.714286) |
| transition | R_std_to_R_inv | Binary | 24 | 0.541667 | 1.000000 | 11/24 (0.458333) | 7/18 (0.388889) | 11/18 (0.611111) |
| transition | R_inv_to_R_std | Binary | 24 | 0.416667 | 1.000000 | 14/24 (0.583333) | 6/24 (0.250000) | 18/24 (0.750000) |

All views in this section are diagnostic-only. They use the frozen probe metadata already bundled with each episode and do not replace the Binary-only headline metric.

## Execution Provenance (diagnostic-only)

| Scope | Mode | Completed | Provider failures | Provider model versions | Mean duration (s) | Usage rows | Total tokens | Finish reasons |
| --- | --- | --- | --- | --- | ---: | --- | ---: | --- |
| overall | Binary | 48/48 (1.000000) | 0/48 (0.000000) | gemini-2.5-flash | 0.979831 | 48/48 (1.000000) | 12720 | FinishReason.STOP=48 |
| dev | Binary | 16/16 (1.000000) | 0/16 (0.000000) | gemini-2.5-flash | 0.981082 | 16/16 (1.000000) | 4240 | FinishReason.STOP=16 |
| private_leaderboard | Binary | 16/16 (1.000000) | 0/16 (0.000000) | gemini-2.5-flash | 0.970146 | 16/16 (1.000000) | 4240 | FinishReason.STOP=16 |
| public_leaderboard | Binary | 16/16 (1.000000) | 0/16 (0.000000) | gemini-2.5-flash | 0.988266 | 16/16 (1.000000) | 4240 | FinishReason.STOP=16 |

Execution provenance is diagnostic-only operational context. It does not change scoring or add a new benchmark metric.

## Binary Diagnostic Slices

| Template | Accuracy | Parse-valid rate |
| --- | ---: | ---: |
| T1 | 0.790000 | 1.000000 |
| T2 | 0.771739 | 1.000000 |

## Binary Difficulty Slices

| Difficulty | Accuracy | Parse-valid rate |
| --- | ---: | ---: |
| easy | 0.790000 | 1.000000 |
| medium | 0.771739 | 1.000000 |

## Notes

- `hard` remains reserved and is not emitted in the current frozen repaired benchmark, so no hard slice is reported.
- Narrative mode was not run in this first real-model panel, so Binary vs Narrative comparison is unavailable.
- Narrative remains required non-leaderboard robustness evidence on the same frozen episodes and probe targets as Binary; only the final four labels are scored, and it does not replace the primary Binary post-shift probe audit.
