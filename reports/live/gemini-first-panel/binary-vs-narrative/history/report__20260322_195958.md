# Gemini First Panel Report

- Release: R18
- Provider: gemini
- Model: gemini-2.5-flash
- Prompt modes run: binary, narrative
- Covered splits: dev, public_leaderboard, private_leaderboard

## Headline

- Binary-only headline metric: gemini-2.5-flash Binary = 0.000000
- Binary parse-valid rate: 0.000000
- Best baseline: random (0.546875)

| Source | Accuracy | Gap vs model |
| --- | ---: | ---: |
| gemini-2.5-flash Binary | 0.000000 | 0.000000 |
| random | 0.546875 | -0.546875 |
| never_update | 0.500000 | -0.500000 |
| last_evidence | 0.500000 | -0.500000 |
| physics_prior | 0.500000 | -0.500000 |
| template_position | 0.453125 | -0.453125 |

## Headline By Split

| Split | Model | random | never-update | last-evidence | physics-prior | template-position |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dev | 0.000000 | 0.609375 | 0.500000 | 0.500000 | 0.500000 | 0.515625 |
| public_leaderboard | 0.000000 | 0.484375 | 0.500000 | 0.500000 | 0.500000 | 0.500000 |
| private_leaderboard | 0.000000 | 0.546875 | 0.500000 | 0.500000 | 0.500000 | 0.343750 |

## Paired Robustness

| Scope | Binary accuracy | Narrative accuracy | Delta | Binary parse-valid | Narrative parse-valid |
| --- | ---: | ---: | ---: | ---: | ---: |
| overall | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| dev | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| public_leaderboard | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| private_leaderboard | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## Diagnostic Slices

| Slice type | Label | Binary accuracy | Narrative accuracy | Delta | Binary parse-valid | Narrative parse-valid |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| template | T1 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| template | T2 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| difficulty | easy | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| difficulty | medium | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## Failure Taxonomy

| Scope | Mode | Parse/format failure rate | Adaptation failure rate | Possible old-rule persistence rate | Possible recency overshoot rate |
| --- | --- | ---: | ---: | ---: | ---: |
| overall | Binary | 1.000000 | 0.000000 | 0.000000 | 0.000000 |
| overall | Narrative | 1.000000 | 0.000000 | 0.000000 | 0.000000 |
| dev | Binary | 1.000000 | 0.000000 | 0.000000 | 0.000000 |
| dev | Narrative | 1.000000 | 0.000000 | 0.000000 | 0.000000 |
| public_leaderboard | Binary | 1.000000 | 0.000000 | 0.000000 | 0.000000 |
| public_leaderboard | Narrative | 1.000000 | 0.000000 | 0.000000 | 0.000000 |
| private_leaderboard | Binary | 1.000000 | 0.000000 | 0.000000 | 0.000000 |
| private_leaderboard | Narrative | 1.000000 | 0.000000 | 0.000000 | 0.000000 |

Taxonomy rates are episode-level over scored outputs. Persistence and recency tags are diagnostic-only exact-match comparisons against `never_update` and `last_evidence`.

## Binary Diagnostic Slices

| Template | Accuracy | Parse-valid rate |
| --- | ---: | ---: |
| T1 | 0.000000 | 0.000000 |
| T2 | 0.000000 | 0.000000 |

## Binary Difficulty Slices

| Difficulty | Accuracy | Parse-valid rate |
| --- | ---: | ---: |
| easy | 0.000000 | 0.000000 |
| medium | 0.000000 | 0.000000 |

## Notes

- `hard` remains reserved and is not emitted in the current frozen repaired benchmark, so no hard slice is reported.
- Narrative remains required non-leaderboard robustness evidence on the same frozen episodes and probe targets as Binary; only the final four labels are scored, and it does not replace the primary Binary post-shift probe audit.
