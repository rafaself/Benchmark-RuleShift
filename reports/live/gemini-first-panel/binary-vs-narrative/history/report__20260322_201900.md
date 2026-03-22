# Gemini First Panel Report

- Release: R18
- Provider: gemini
- Model: gemini-2.5-flash
- Prompt modes run: binary, narrative
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

## Paired Robustness

| Scope | Binary accuracy | Narrative accuracy | Delta | Binary parse-valid | Narrative parse-valid |
| --- | ---: | ---: | ---: | ---: | ---: |
| overall | 0.781250 | 0.046875 | 0.734375 | 1.000000 | 0.083333 |
| dev | 0.781250 | 0.031250 | 0.750000 | 1.000000 | 0.062500 |
| public_leaderboard | 0.718750 | 0.000000 | 0.718750 | 1.000000 | 0.000000 |
| private_leaderboard | 0.843750 | 0.109375 | 0.734375 | 1.000000 | 0.187500 |

## Diagnostic Slices

| Slice type | Label | Binary accuracy | Narrative accuracy | Delta | Binary parse-valid | Narrative parse-valid |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| template | T1 | 0.790000 | 0.020000 | 0.770000 | 1.000000 | 0.040000 |
| template | T2 | 0.771739 | 0.076087 | 0.695652 | 1.000000 | 0.130435 |
| difficulty | easy | 0.790000 | 0.020000 | 0.770000 | 1.000000 | 0.040000 |
| difficulty | medium | 0.771739 | 0.076087 | 0.695652 | 1.000000 | 0.130435 |

## Failure Taxonomy

| Scope | Mode | Provider/runtime error rate | Parse/format failure rate | Adaptation failure rate | Possible old-rule persistence rate | Possible recency overshoot rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| overall | Binary | 0.000000 | 0.000000 | 0.520833 | 0.000000 | 0.125000 |
| overall | Narrative | 0.000000 | 0.916667 | 0.083333 | 0.000000 | 0.041667 |
| dev | Binary | 0.000000 | 0.000000 | 0.500000 | 0.000000 | 0.062500 |
| dev | Narrative | 0.000000 | 0.937500 | 0.062500 | 0.000000 | 0.062500 |
| public_leaderboard | Binary | 0.000000 | 0.000000 | 0.687500 | 0.000000 | 0.250000 |
| public_leaderboard | Narrative | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 |
| private_leaderboard | Binary | 0.000000 | 0.000000 | 0.375000 | 0.000000 | 0.062500 |
| private_leaderboard | Narrative | 0.000000 | 0.812500 | 0.187500 | 0.000000 | 0.062500 |

Taxonomy rates are episode-level over scored outputs. Persistence and recency tags are diagnostic-only exact-match comparisons against `never_update` and `last_evidence`.

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
- Narrative remains required non-leaderboard robustness evidence on the same frozen episodes and probe targets as Binary; only the final four labels are scored, and it does not replace the primary Binary post-shift probe audit.
