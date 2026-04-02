# Public Leaderboard Freeze

Date: 2026-04-01

Scope: freeze `public_leaderboard` to exactly 54 episodes for notebook-based auditing without changing the official Kaggle contract.

## Selection Policy

The public split was regenerated from one complete 54-seed structural cycle.

Policy used:

- keep benchmark semantics unchanged by using canonical `generate_episode()`
- require exact balance by construction across the generator's natural structure:
  - `difficulty`
  - `transition`
  - `template_family`
  - `template_id`
- search aligned 54-seed windows in a reserved public seed band
- reject any window with exact duplicate prompts or exact duplicate structural fingerprints
- rank candidate windows by:
  1. lowest maximum pairwise item-pair overlap
  2. lowest worst-case prompt similarity

Chosen frozen window:

- seed range: `16524..16577`
- seed bank version: `R14-public-5`
- public manifest path: `src/frozen_splits/public_leaderboard.json`

Reason this window was selected:

- it preserves perfect structural balance
- it had no exact duplicates
- within the audited candidate band it achieved the best observed worst-case prompt similarity among the windows tied for the best maximum item-pair overlap

## Balance Report

Public episode count: `54`

Distribution by difficulty:

| difficulty | episodes |
| --- | ---: |
| easy | 18 |
| medium | 18 |
| hard | 18 |

Distribution by transition:

| transition | episodes |
| --- | ---: |
| `R_std_to_R_inv` | 27 |
| `R_inv_to_R_std` | 27 |

Distribution by template family:

| template_family | episodes |
| --- | ---: |
| canonical | 18 |
| observation_log | 18 |
| case_ledger | 18 |

Distribution by template id:

| template_id | episodes |
| --- | ---: |
| T1 | 18 |
| T2 | 18 |
| T3 | 18 |

Combined structural coverage:

- unique `(difficulty, transition, template_family, template_id)` combinations: `54`
- count per combination: exactly `1`

## Similarity / Duplicate Report

Exact duplicate checks:

- duplicate episode fingerprints: `0`
- duplicate rendered prompts: `0`

Near-duplicate proxy:

- metric: overlap in the 9 `(r1, r2)` item pairs between any two episodes
- maximum observed overlap: `6 / 9`
- overlap histogram across all episode pairs:
  - `0`: 65 pairs
  - `1`: 303 pairs
  - `2`: 487 pairs
  - `3`: 380 pairs
  - `4`: 156 pairs
  - `5`: 36 pairs
  - `6`: 4 pairs

Most similar prompt pair in the chosen window:

- seeds: `16524` and `16558`
- prompt similarity ratio: `0.857645`
- item-pair overlap for that pair: `1 / 9`

Highest item-pair overlap examples:

- `16526` vs `16554`: overlap `6 / 9`, prompt similarity `0.854895`
- `16531` vs `16548`: overlap `6 / 9`, prompt similarity `0.191824`
- `16537` vs `16556`: overlap `6 / 9`, prompt similarity `0.061044`
- `16551` vs `16553`: overlap `6 / 9`, prompt similarity `0.794376`

Interpretation:

- the public set contains no exact duplicates
- template-family-specific phrasing can make some prompts look textually similar even when their underlying item sets differ sharply
- the selected window keeps structural overlap bounded while preserving exact balance, which is the primary auditability goal for this public surface
