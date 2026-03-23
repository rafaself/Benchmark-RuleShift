# RuleShift Benchmark v1 — Benchmark Contract

**Version:** v1
**Status:** Frozen
**Authoritative:** This is the single source of truth for all benchmark identity, metric, split, scoring, and claim decisions. All other docs must defer to this file on contract matters.

---

## 1. Target Cognitive Ability

**Cognitive flexibility** — specifically, post-shift rule application after sparse contradictory evidence.

The benchmark evaluates whether a model can infer a hidden rule shift from a small number of post-shift labeled examples and apply the updated rule to unlabeled probes. Electrostatics is used only as a controlled substrate; the benchmark does not target physics knowledge or skill.

---

## 2. Primary Benchmark Claim

A high v1 Binary score is evidence that a model correctly applied the post-shift rule to the final probes after sparse contradictory evidence in the frozen episodes.

---

## 3. Explicit Non-Claims

The benchmark does **not** claim to measure or provide evidence for:

- physics skill as the primary measured ability
- full executive-function decomposition
- broad adaptation ability
- broad AGI capability
- general reasoning ability
- human-level performance
- cross-provider readiness or cross-provider equivalence
- online change-detection latency
- switch cost measurement
- recovery length
- immediate post-shift drop
- emitted `hard` difficulty slices
- explanation quality or formatting compliance

---

## 4. Official Task Family

### 4.1 Leaderboard-primary task: Binary

- Binary is the **only** task scored for the leaderboard.
- The `@kbench.task` decorator must select Binary as the main leaderboard task via `%choose`.
- Binary is the only path that contributes to the headline metric and leaderboard ranking.

### 4.2 Companion task: Narrative

- Narrative runs on the same frozen episodes and probe targets as Binary.
- Narrative is required same-episode robustness and interpretability evidence.
- Narrative does **not** contribute to the headline score or leaderboard ranking.
- Only the final four predicted labels are scored in Narrative (any preceding reasoning text is unscored).
- Narrative is non-blocking: a Narrative failure must not prevent the Binary result from being reported.

---

## 5. Headline Metric

**Metric name:** Post-shift Probe Accuracy

**Definition:** The fraction of Binary post-shift probe labels answered correctly across all evaluated episodes.

**Formula:**

```
Post-shift Probe Accuracy = sum(correct_probe_predictions) / sum(total_probes)
```

where `total_probes = 4 × number_of_episodes` for any split.

**Per-episode scoring unit:**

Each episode contributes `(num_correct, 4)` — numerator and denominator for the episode's four probes. This maps to the `tuple[int, int]` Kaggle return type.

- `num_correct`: count of probe predictions exactly matching `probe_targets` (0–4).
- Denominator is always `4` (fixed probe count per episode).
- Invalid or malformed outputs score `(0, 4)`.

**Aggregate:** `sum(num_correct) / (4 × N_episodes)`, equivalently the mean of per-episode `(num_correct / 4)`.

No other metric changes the leaderboard score.

---

## 6. Split Policy

### 6.1 Split definitions

| Split name | Role | Episodes | Held-out? |
|---|---|---|---|
| `dev` | Internal debugging, metric checks, baseline verification | 16 | No |
| `public_leaderboard` | Public leaderboard evaluation | 16 | Yes (from model, not from developer) |
| `private_leaderboard` | Private leaderboard evaluation and contamination resistance | 16 | Yes (from model and from public results) |

Split names are exact. No aliases.

### 6.2 Frozen seed banks

Each split uses a separate frozen seed bank. Episodes are not shared across splits. Seeds are deterministic: the same seed always regenerates the same episode.

| Split | Manifest version | Seed bank version |
|---|---|---|
| `dev` | R14 | R14-dev-1 |
| `public_leaderboard` | R14 | R14-public-1 |
| `private_leaderboard` | R14 | R14-private-2 |

### 6.3 Contamination assumptions

- `private_leaderboard` episodes are never exposed in public results or model-facing prompts.
- `public_leaderboard` scores are visible on the public leaderboard but episodes are not disclosed to models during evaluation.
- `dev` is available for local iteration and is not held out.

### 6.4 Split-to-episode_split mapping (internal)

| Partition | `episode_split` field |
|---|---|
| `dev` | `dev` |
| `public_leaderboard` | `public` |
| `private_leaderboard` | `private` |

---

## 7. Prompt Invariants

### 7.1 Episode structure

Each episode contains exactly 9 items:

| Positions | Content | Phase | Labeled? |
|---|---|---|---|
| 1–`pre_count` | Pre-shift items | `pre` (rule_A) | Yes |
| `pre_count+1`–5 | Post-shift items | `post` (rule_B) | Yes |
| 6–9 | Post-shift probes | `post` | No (scored) |

Constants:

| Constant | Value |
|---|---|
| `EPISODE_LENGTH` | 9 |
| `LABELED_ITEM_COUNT` | 5 |
| `PROBE_COUNT` | 4 |

### 7.2 Template family

| Template | `pre_count` | `post_labeled_count` |
|---|---|---|
| T1 | 2 | 3 |
| T2 | 3 | 2 |

Template set version: `v1`.

### 7.3 Rule family

Two rules over charges from `{-3, -2, -1, +1, +2, +3}`:

- `R_std`: same-sign charges repel, opposite-sign charges attract.
- `R_inv`: same-sign charges attract, opposite-sign charges repel.

Labels depend only on charge sign, not magnitude. Pair order does not affect the outcome. The hidden rule shift is never announced to the model.

### 7.4 What must not appear in the model-facing prompt

`rule_A`, `rule_B`, `probe_targets`, and `probe_metadata` must never be exposed in the model-facing prompt.

### 7.5 Difficulty

- Currently emitted: `easy`, `medium`.
- Reserved (not emitted): `hard`.
- No benchmark claim depends on `hard` slices.

---

## 8. Scoring Invariants

### 8.1 What is scored

Only the four post-shift probes (positions 6–9). Pre-shift and post-shift labeled items are context, not scored.

### 8.2 Probe targets

Probe targets use the **slice-local effective rule**, not a uniform global `rule_B`:

- Post-shift labeled items cover exactly two distinct sign patterns (one same-sign, one opposite-sign).
- A probe uses `rule_B` if its sign pattern was updated by post-shift labeled items; otherwise `rule_A`.
- The full probe block covers all four sign patterns (`++`, `--`, `+-`, `-+`) exactly once.
- Targets never collapse to `rule_A` or `rule_B` for all four probes.

Targets are frozen in the manifests. They must not be recomputed by the notebook.

### 8.3 Canonical scoring rule (one episode)

Given `probe_targets = (t₁, t₂, t₃, t₄)` and parsed prediction `(p₁, p₂, p₃, p₄)`:

```
num_correct(e) = sum(1 if pᵢ == tᵢ else 0  for i in 1..4)
score(e)       = (num_correct(e), 4)
```

Labels compared as exact string matches after normalizing to lowercase (`attract` / `repel`).

### 8.4 Invalid output handling

| Condition | Score |
|---|---|
| Parses to exactly 4 valid labels | Score each against `probe_targets` |
| Fewer or more than 4 labels | `(0, 4)` |
| Unrecognized label tokens | `(0, 4)` |
| Provider failure / no response | `(0, 4)` |
| Empty or blank output | `(0, 4)` |

`ParseStatus` values: `VALID`, `INVALID`, `SKIPPED_PROVIDER_FAILURE`. Only `VALID` contributes non-zero scores.

### 8.5 Binary output format

Required: exactly four interaction labels in order, comma- or newline-separated:

```
attract, repel, repel, attract
```

Labels are case-insensitive. No other format variations are permitted for the strict Binary path.

### 8.6 Frozen versions

| Component | Version |
|---|---|
| Spec | `v1` |
| Generator | `R12` |
| Template set | `v1` |
| Difficulty | `R12` |
| Manifest | `R14` |
| Parser | `v1` |
| Metric | `v1` |

Any change that could alter scores requires a version bump and a new manifest.

---

## 9. Frozen Assets

The following are immutable benchmark inputs:

| Asset | Location | Version |
|---|---|---|
| Dev split | `src/frozen_splits/dev.json` | R14, seed bank R14-dev-1 |
| Public leaderboard split | `src/frozen_splits/public_leaderboard.json` | R14, seed bank R14-public-1 |
| Private leaderboard split | `src/frozen_splits/private_leaderboard.json` | R14, seed bank R14-private-2 |
| Kaggle runtime-contract manifest | `packaging/kaggle/frozen_artifacts_manifest.json` | SHA-256 hashes for the official notebook and frozen split inputs |
| Anti-shortcut gate evidence | `tests/fixtures/release_r13_validity_report.json` | R13 PASS |
| Empirical re-audit evidence | `tests/fixtures/release_r15_reaudit_report.json` | R15 |

Local code and frozen splits remain authoritative if any downstream notebook produces different results.

---

## 10. Locked Decisions

1. Binary is the only leaderboard task. `%choose` must select Binary.
2. Narrative is companion-only and non-blocking.
3. Post-shift Probe Accuracy is the sole headline metric.
4. Scoring unit is `(num_correct, 4)` per episode.
5. Invalid outputs score `(0, 4)`. No partial credit.
6. Probe targets use slice-local effective rule (Section 8.2). Targets are frozen, not recomputed.
7. Split names are exactly `dev`, `public_leaderboard`, `private_leaderboard`.
8. All frozen assets load from `src/frozen_splits/`. No regeneration in notebooks.
9. `PARSER_VERSION v1` and `METRIC_VERSION v1` are frozen.
10. `hard` difficulty is reserved, not emitted, and no claim depends on it.

---

## 11. Source of Truth Model

| Concern | Authoritative source | Role |
|---|---|---|
| Benchmark definition | This file (`KAGGLE_BENCHMARK_CONTRACT.md`) | Freezes identity, metric, splits, scoring, prompt invariants, and claim boundaries. All other docs defer to this file on contract matters. |
| Runtime behavior | Source package implementation (`src/`) | The code under `src/` and frozen assets under `src/frozen_splits/` are the executable truth. If any document (including this contract) diverges from implemented behavior, the code is authoritative for what the benchmark *does*; the contract is authoritative for what it *should* do — and the divergence is a bug to be resolved. |
| Kaggle execution entry point | Official leaderboard notebook only | `packaging/kaggle/ruleshift_benchmark_v1_kbench.ipynb`, wired by `packaging/kaggle/kernel-metadata.json` with `code_file = "ruleshift_benchmark_v1_kbench.ipynb"`, is the sole Kaggle execution surface. Staging notebooks, local panel runners, and audit scripts are development tools, not leaderboard entry points. |
| Supporting exposition | `BENCHMARK_CARD.md` and `README.md` | These describe the benchmark for human readers. They must not introduce, modify, or contradict any contract term. If they diverge from this contract, this contract governs. |

---

## 12. Deferred (Out of Scope for v1)

- Anthropic or OpenAI provider live evidence
- Cross-provider comparison claims
- `hard` difficulty slice emission
- Item-level switch cost or adaptation lag metrics
- Recovery length or immediate post-shift drop analysis
- Human pilot or independent rerun
- Explanation quality scoring
- Broad AGI or cross-provider capability claims
