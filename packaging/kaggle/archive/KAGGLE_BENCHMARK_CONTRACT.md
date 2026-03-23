# RuleShift Benchmark v1 — Kaggle Benchmark Contract (OBSOLETE)

> **Status: OBSOLETE — Phase 2 archive.**
> The authoritative benchmark contract is now [`/KAGGLE_BENCHMARK_CONTRACT.md`](../../../KAGGLE_BENCHMARK_CONTRACT.md) at the repository root.
> This file is retained as a historical record of the Phase 2 Kaggle integration contract. Do not update it or treat it as authoritative.

**Version:** v1
**Original status:** Frozen for Phase 2 Kaggle integration
**Source of truth:** `src/`, `src/frozen_splits/`, local validation and audit results
**This document covered:** the exact interface a Kaggle `@kbench.task` implementation must satisfy

---

## 1. Official Benchmark Identity

| Field | Value |
|---|---|
| Benchmark name | RuleShift Benchmark |
| Version | v1 |
| Track | Executive Functions |
| Construct | Cognitive flexibility — post-shift rule application after sparse contradictory evidence |
| Substrate | Simplified two-charge electrostatic interaction rules (not a physics test) |
| Spec version | `v1` |
| Generator version | `R12` |
| Template set version | `v1` |
| Difficulty version | `R12` |
| Manifest version | `R14` |

---

## 2. Tasks

### 2.1 Leaderboard-primary task: Binary

- **Only Binary is scored for the Kaggle leaderboard.**
- The `@kbench.task` decorator must select Binary as the main leaderboard task via `%choose`.
- Binary is the only path that contributes to the headline metric and leaderboard ranking.

### 2.2 Companion task: Narrative

- Narrative runs on the same frozen episodes and probe targets as Binary.
- Narrative is required same-episode robustness and interpretability evidence.
- Narrative does **not** contribute to the Kaggle headline score or leaderboard ranking.
- Only the final four predicted labels are scored in Narrative (any preceding reasoning text is unscored).

---

## 3. Official Metric

**Metric name:** Post-shift Probe Accuracy

**Definition:** The fraction of Binary post-shift probe labels answered correctly across all evaluated episodes.

**Formula:**

```
Post-shift Probe Accuracy = sum(correct_probe_predictions) / sum(total_probes)
```

where `total_probes = 4 × number_of_episodes` for any split.

**Per-episode scoring unit for Kaggle:**

Each episode contributes `(num_correct, 4)` to the aggregate — that is, numerator and denominator for the episode's four probes. This maps to the standard `tuple[int, int]` Kaggle return type.

- `num_correct` is the count of probe predictions that exactly match the episode's `probe_targets` (0–4).
- The denominator is always `4` (the fixed probe count per episode).
- Invalid or malformed outputs score `(0, 4)` for that episode.

**Aggregate metric:** Mean over episodes of `(num_correct / 4)`, equivalently `sum(num_correct) / (4 × N_episodes)`.

---

## 4. Episode Contents

Each episode is one benchmark row. An episode contains exactly nine items structured as follows:

```
positions 1–pre_count          : pre-shift labeled items (phase = pre, rule = rule_A)
positions pre_count+1–5        : post-shift labeled items (phase = post, rule = rule_B)
positions 6–9                  : post-shift probes (phase = post, unlabeled, scored under rule_B)
```

Constants:

| Constant | Value |
|---|---|
| `EPISODE_LENGTH` | 9 items total |
| `LABELED_ITEM_COUNT` | 5 (positions 1–5) |
| `PROBE_COUNT` | 4 (positions 6–9) |

Template family (frozen):

| Template | pre_count | post_labeled_count |
|---|---|---|
| T1 | 2 | 3 |
| T2 | 3 | 2 |

The hidden rule shift is never announced to the model.

### 4.1 What is scored

Only the four post-shift probes (positions 6–9) are scored. The pre-shift and post-shift labeled items are provided as context and are not scored.

### 4.2 Probe targets

Probe targets are the correct labels for positions 6–9. They are derived using a slice-local effective rule:

- The post-shift labeled items (positions `pre_count+1` through 5) cover exactly two distinct sign patterns: one same-sign (`++` or `--`) and one opposite-sign (`+-` or `-+`).
- For a probe, the active rule is `rule_B` if the probe's sign pattern was updated by the post-shift labeled items; otherwise it is `rule_A`.
- The full probe block covers all four sign patterns (`++`, `--`, `+-`, `-+`) exactly once.
- Targets never collapse to the global `rule_A` block or the global `rule_B` block for all four probes.

In practice, for all currently frozen episodes: the targets are the labels that a model applying `rule_B` consistently would produce for probes whose sign pattern appears in the post-shift labeled segment. For sign patterns not yet updated, the model applying `rule_A` would produce the target. The task is to infer which rule is active post-shift from the labeled context.

---

## 5. Binary Output Format

**Required model output:** exactly four interaction labels in order, separated by commas or newlines:

```
attract, repel, repel, attract
```

Labels are case-insensitive. The parser accepts comma-separated or newline-separated tokens. No other format variations are permitted for the strict Binary path.

**Structured representation for Kaggle schema-based prompting:**

```python
tuple[InteractionLabel, InteractionLabel, InteractionLabel, InteractionLabel]
```

where `InteractionLabel` is `"attract"` or `"repel"`. This maps to a 4-tuple of string literals in the schema, suitable for structured output APIs.

**Submission CSV format** (one row per episode):

```csv
episode_id,predictions
public_leaderboard_000001,"attract,repel,repel,attract"
```

---

## 6. Invalid and Malformed Output Handling

| Condition | Scoring result |
|---|---|
| Output parses to exactly 4 valid labels | Score each label against `probe_targets` |
| Output parses to fewer or more than 4 labels | `(0, 4)` — all four probes counted as incorrect |
| Output contains unrecognized label tokens | `(0, 4)` |
| Provider failure / no response | `(0, 4)` |
| Empty or blank output | `(0, 4)` |

`ParseStatus` values: `VALID`, `INVALID`, `SKIPPED_PROVIDER_FAILURE`. Only `VALID` predictions contribute non-zero probe scores. All other statuses score `(0, 4)`.

---

## 7. Split Semantics

The three splits are frozen and immutable. Split names are exact:

| Split name | Purpose |
|---|---|
| `dev` | Internal debugging, metric checks, baseline verification (16 episodes) |
| `public_leaderboard` | Public leaderboard evaluation (16 episodes) |
| `private_leaderboard` | Private leaderboard evaluation and contamination resistance (16 episodes) |

Each split uses a separate frozen seed bank. Episodes are not shared across splits. Seeds are deterministic: the same seed always regenerates the same episode.

**Split → episode_split mapping** (internal only):

| partition | episode_split |
|---|---|
| `dev` | `dev` |
| `public_leaderboard` | `public` |
| `private_leaderboard` | `private` |

---

## 8. Frozen Assets

The following assets are immutable benchmark inputs. They must not be regenerated or modified:

| Asset | Location | Version |
|---|---|---|
| Dev split manifest | `src/frozen_splits/dev.json` | manifest `R14`, seed bank `R14-dev-1` |
| Public leaderboard split manifest | `src/frozen_splits/public_leaderboard.json` | manifest `R14`, seed bank `R14-public-1` |
| Private leaderboard split manifest | `src/frozen_splits/private_leaderboard.json` | manifest `R14`, seed bank `R14-private-2` |
| Frozen artifacts index | `packaging/kaggle/frozen_artifacts_manifest.json` | SHA-256 hashes of all frozen inputs |
| Anti-shortcut gate evidence | `tests/fixtures/release_r13_validity_report.json` | R13 gate PASS |
| Empirical re-audit evidence | `tests/fixtures/release_r15_reaudit_report.json` | R15 re-audit |

Any divergence from the frozen manifests invalidates the benchmark claim. The local code and frozen splits remain authoritative if the Kaggle notebook produces different results.

---

## 9. Canonical Scoring Rule for One Episode

Given one episode `e` with `probe_targets = (t_1, t_2, t_3, t_4)` and a Binary model prediction parsed to `(p_1, p_2, p_3, p_4)`:

```
num_correct(e) = sum(1 if p_i == t_i else 0  for i in 1..4)
score(e)       = (num_correct(e), 4)
```

If the prediction is invalid (any parse failure): `score(e) = (0, 4)`.

Labels are compared as exact string matches after normalizing to lowercase (`attract` / `repel`).

---

## 10. Aggregate Metric Definition

Given `N` episodes in a split, each with `score(e_j) = (c_j, 4)`:

```
Post-shift Probe Accuracy = sum(c_j for j in 1..N) / (4 × N)
```

This is a flat average over all probe slots, not an average of per-episode averages (they are equivalent here because every episode has exactly 4 probes, but the flat formula is canonical).

---

## 11. Locked Decisions for Kaggle Integration

1. **Binary is the only Kaggle leaderboard task.** `%choose` must select Binary.
2. **Narrative is companion-only.** It must not appear as a leaderboard scoring path.
3. **Post-shift Probe Accuracy is the sole headline metric.** No other metric changes the score.
4. **The scoring unit is `(num_correct, 4)` per episode**, matching the `tuple[int, int]` Kaggle return type.
5. **Invalid outputs score `(0, 4)`.** No partial credit for malformed predictions.
6. **Probe targets use the slice-local effective rule**, not a uniform global `rule_B`. See Section 4.2. Targets are frozen in the manifests; they must not be recomputed by the notebook.
7. **Split names are exactly `dev`, `public_leaderboard`, `private_leaderboard`.** No aliases.
8. **All frozen assets are loaded from `src/frozen_splits/`.** The notebook does not regenerate episodes.
9. **PARSER_VERSION `v1` and METRIC_VERSION `v1` are frozen.** Any change that could alter scores requires a version bump and a new manifest.
10. **`hard` difficulty is reserved and not emitted.** No benchmark claim depends on `hard` slices.

---

## 12. What the `@kbench.task` Implementation Must Do

A correct Phase 2 implementation must:

1. Load frozen episode data from the split manifests under `src/frozen_splits/`.
2. Reconstruct episodes deterministically using the frozen seeds and the local generator (`GENERATOR_VERSION = R12`).
3. Render each episode as a Binary prompt using the canonical Binary format (Section 8.1 of `ruleshift_benchmark_implementation_spec.md`).
4. Call the model and collect the raw text output.
5. Parse the output with the v1 Binary parser (`parse_binary_output`).
6. Compute `(num_correct, 4)` per episode using `probe_targets` from the frozen episode.
7. Return `(num_correct, 4)` as the `tuple[int, int]` result.
8. Aggregate Post-shift Probe Accuracy across episodes as the leaderboard score.
9. Run Narrative on the same episodes as a companion task (not leaderboard-scored).
10. Not expose `rule_A`, `rule_B`, `probe_targets`, or `probe_metadata` in the model-facing prompt.

---

## 13. Out of Scope for Phase 2

The following are explicitly deferred and must not be partially implemented during Kaggle integration:

- Anthropic or OpenAI provider live evidence
- Cross-provider comparison claims
- `hard` difficulty slice emission
- Item-level switch cost or adaptation lag metrics
- Recovery length or immediate post-shift drop analysis
- Human pilot or independent rerun
- Explanation quality scoring
- Broad AGI or cross-provider capability claims
