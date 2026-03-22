# Gemini First Panel Report

- Release: R18
- Provider: gemini
- Model: gemini-2.5-flash
- Prompt modes run: binary
- Covered splits: dev, public_leaderboard, private_leaderboard

## Overall

- Post-shift Probe Accuracy: 0.781250
- Parse-valid rate: 1.000000
- Best baseline: random (0.546875)

| Source | Accuracy | Gap vs model |
| --- | ---: | ---: |
| gemini-2.5-flash Binary | 0.781250 | 0.000000 |
| random | 0.546875 | 0.234375 |
| never_update | 0.500000 | 0.281250 |
| last_evidence | 0.500000 | 0.281250 |
| physics_prior | 0.500000 | 0.281250 |
| template_position | 0.453125 | 0.328125 |

## By Split

| Split | Model | random | never-update | last-evidence | physics-prior | template-position |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dev | 0.781250 | 0.609375 | 0.500000 | 0.500000 | 0.500000 | 0.515625 |
| public_leaderboard | 0.718750 | 0.484375 | 0.500000 | 0.500000 | 0.500000 | 0.500000 |
| private_leaderboard | 0.843750 | 0.546875 | 0.500000 | 0.500000 | 0.500000 | 0.343750 |

## By Template

| Template | Accuracy | Parse-valid rate |
| --- | ---: | ---: |
| T1 | 0.790000 | 1.000000 |
| T2 | 0.771739 | 1.000000 |

## By Difficulty

| Difficulty | Accuracy | Parse-valid rate |
| --- | ---: | ---: |
| easy | 0.790000 | 1.000000 |
| medium | 0.771739 | 1.000000 |

## Notes

- `hard` remains reserved and is not emitted in the current frozen repaired benchmark, so no hard slice is reported.
- Narrative mode was not run in this first real-model panel, so Binary vs Narrative comparison is unavailable.
- Narrative remains required non-leaderboard robustness evidence on the same frozen episodes and probe targets as Binary; only the final four labels are scored, and it does not replace the primary Binary post-shift probe audit.
