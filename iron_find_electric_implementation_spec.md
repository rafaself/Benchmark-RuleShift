# Iron Find Electric

## Concrete Implementation Spec

> Status note: the repository already implements the local benchmark pipeline described here. The current frozen benchmark passes the R13 anti-shortcut validity gate, the R15 re-audit keeps `last_evidence` bounded, `hard` remains reserved and un-emitted, and current v1 readiness is anchored to Gemini evidence. Anthropic and OpenAI integrations exist locally but are outside the current v1 readiness gate.

## 1. Objective

Describe and maintain the implemented local benchmark pipeline for the Iron Find Electric v1 task, a targeted Executive Functions benchmark for cognitive flexibility. It uses electrostatics only as a controlled substrate for evaluating final post-shift rule application after sparse contradictory evidence in short two-charge binary episodes.

The implementation makes the following interpretation defensible:

> A strong v1 Binary score indicates that the model applied an updated latent rule to final post-shift probes after contradictory sparse evidence.

The implementation must prevent high scores from being explained mainly by:

- default electrostatics prior;
- majority-label behavior;
- constant-output behavior;
- recency-only behavior;
- structural template cues;
- formatting artifacts.

---

## 2. v1 Scope

### Included

- two charges per case;
- binary label: `attract` or `repel`;
- one hidden rule before shift;
- one hidden rule after shift;
- fixed total episode length;
- deterministic generation;
- one leaderboard-primary path;
- one required same-episode robustness companion.

### Excluded

- more than two charges;
- force magnitude prediction;
- continuous outputs;
- temporal dynamics;
- trajectory simulation;
- explanation-based primary scoring;
- item-level switch-cost or recovery claims;
- broad AGI, human-level, or cross-provider claims.

---

## 3. Rule System

## 3.1 Allowed charge values

Use:

```text
CHARGES = [-3, -2, -1, +1, +2, +3]
```

Zero is excluded.

## 3.2 Rule family for v1

Only two rules are allowed in the v1 benchmark:

### `R_std`
- same signs -> `repel`
- opposite signs -> `attract`

### `R_inv`
- same signs -> `attract`
- opposite signs -> `repel`

## 3.3 Label function

Let:

```text
same_sign(q1, q2) = sign(q1) == sign(q2)
```

Then:

```text
label(R_std, q1, q2) = repel   if same_sign(q1, q2) else attract
label(R_inv, q1, q2) = attract if same_sign(q1, q2) else repel
```

Magnitude must not affect the label.

---

## 4. Episode Definition

## 4.1 Template family

Each episode contains exactly 9 items and must use one of the following frozen templates:

### `T1`
```text
2 pre + 3 post-labeled + 4 probes
```

### `T2`
```text
3 pre + 2 post-labeled + 4 probes
```

The final 4 items are always probes. The first 5 items are always labeled examples. The hidden shift boundary varies by template.

The benchmark must not use any approved template whose shift position is recoverable solely by counting backward from the final probe block.

## 4.2 Hidden transition

For each episode:

- choose initial rule `rule_A ∈ {R_std, R_inv}`;
- set `rule_B` to the other rule;
- choose `template_id ∈ {T1, T2}`;
- derive `pre_count` and `post_labeled_count` from `template_id`;
- set `shift_after_position = pre_count`;
- first `pre_count` labeled examples use `rule_A`;
- next `post_labeled_count` labeled examples use `rule_B`;
- probe targets are derived using a slice-local effective rule: for each probe, the active rule is `rule_B` if the probe's charge sign pattern (`++`, `--`, `+-`, or `-+`) was covered by the post-shift labeled items; otherwise `rule_A`. Probe targets are frozen in the split manifests and must not be recomputed during evaluation.

The shift is never announced to the model.

Note: the post-shift labeled items cover exactly two distinct sign patterns (one same-sign and one opposite-sign); the probe block covers all four sign patterns exactly once. Targets never collapse to the uniform global `rule_A` or global `rule_B` output for all four probes.

## 4.3 No duplicate item tuples within an episode

Within a single episode, `(q1, q2)` pairs must be unique unless a later diagnostic variant explicitly allows repetition.

---

## 5. Core Generator Contract

## 5.1 Episode generation algorithm

For each episode:

1. sample `rule_A` uniformly from `{R_std, R_inv}`;
2. set `rule_B` as the opposite rule;
3. sample `template_id` from the frozen template set;
4. derive `pre_count`, `post_labeled_count`, and `shift_after_position`;
5. sample `pre_count` unique pre-shift labeled examples from the charge space;
6. sample `post_labeled_count` unique post-shift labeled examples;
7. sample 4 unique post-shift probes;
8. verify all validity constraints;
9. if any constraint fails, resample the episode.

Across each split, template usage must be approximately balanced between `T1` and `T2`.

## 5.2 Charge-space definition

Ordered pairs are allowed:

```text
CASE_SPACE = {(q1, q2) | q1 ∈ CHARGES, q2 ∈ CHARGES}
```

Since label is symmetric in `q1, q2`, ordered presentation is a surface-form choice, not a semantic distinction.

## 5.3 Required per-episode constraints

Every generated episode must satisfy all of the following.

### Constraint A - contradiction after shift
At least 1 post-shift labeled example must directly contradict the label predicted by `rule_A`.

Operationally:

```text
label(rule_B, q1, q2) != label(rule_A, q1, q2)
```

for at least one labeled item in the post-shift segment.

### Constraint B - nontrivial probe block
The 4 probes in an episode must not all have the same correct label under `rule_B`.

### Constraint C - disagreement metadata
The implementation should still compute and store whether each probe is a disagreement case:

```text
label(R_std, q1, q2) != label(R_inv, q1, q2)
```

This flag is for audit and analysis only, not for the headline metric.

### Constraint D - balanced probe labels at dataset level
Across the full dataset, probe answers under `rule_B` should be approximately balanced between `attract` and `repel`.

### Constraint E - balanced transition direction
Across the full dataset, approximately half of episodes must be:

- `R_std -> R_inv`

and half:

- `R_inv -> R_std`

### Constraint F - sign-pattern coverage
Across the full dataset, the following pair categories must be represented evenly enough for stable slice analysis:

- positive / positive
- negative / negative
- positive / negative
- negative / positive

### Constraint G - template validity
The approved template family must not expose the hidden shift boundary purely through end-position counting.

---

## 6. Difficulty Tiers

Difficulty is metadata attached to each episode.

## 6.1 Easy

Requirements:

- strong contradiction after shift;
- redundant post-shift evidence;
- mixed probe labels;
- low ambiguity after the labeled sequence.

## 6.2 Medium

Requirements:

- contradiction is present but less redundant;
- moderate ambiguity remains after the first post-shift labeled item;
- probe mix creates moderate temptation for shortcut behavior.

## 6.3 Hard

Requirements:

- minimal sufficient contradiction pattern;
- less redundant post-shift evidence;
- stronger temptation for persistence or recency shortcut heuristics such as `last_evidence`.

Difficulty assignment must be deterministic from generator metadata, not subjective post hoc judgment.

Current implementation note: emitted difficulties are `easy` and `medium`. `hard` remains reserved in the protocol but is not emitted by the current generator.

---

## 7. Episode Data Schema

Each episode is represented as one structured row in the source dataset.

## 7.1 Canonical row fields

```text
episode_id: string
split: string                  # dev | public_leaderboard | private_leaderboard
difficulty: string             # easy | medium | hard
template_id: string            # T1 | T2
pre_count: integer             # 2 | 3
post_labeled_count: integer    # 3 | 2
shift_after_position: integer  # equals pre_count
rule_A: string                 # stored internally; not shown to model
rule_B: string                 # stored internally; not shown to model
transition: string             # R_std_to_R_inv | R_inv_to_R_std
items: json/string
probe_targets: json/string
probe_metadata: json/string
```

## 7.2 `items` structure

`items` contains an ordered list of 9 dict objects. Example for `T1`:

```json
[
  {"position": 1, "phase": "pre",  "kind": "labeled", "q1": 2,  "q2": -3, "label": "attract"},
  {"position": 2, "phase": "pre",  "kind": "labeled", "q1": -1, "q2": -2, "label": "repel"},
  {"position": 3, "phase": "post", "kind": "labeled", "q1": 3,  "q2": 1,  "label": "attract"},
  {"position": 4, "phase": "post", "kind": "labeled", "q1": 1,  "q2": -1, "label": "repel"},
  {"position": 5, "phase": "post", "kind": "labeled", "q1": -2, "q2": -3, "label": "attract"},
  {"position": 6, "phase": "post", "kind": "probe",   "q1": 2,  "q2": 3},
  {"position": 7, "phase": "post", "kind": "probe",   "q1": -3, "q2": 1},
  {"position": 8, "phase": "post", "kind": "probe",   "q1": 1,  "q2": -2},
  {"position": 9, "phase": "post", "kind": "probe",   "q1": -1, "q2": -2}
]
```

## 7.3 `probe_targets` structure

```json
["attract", "repel", "repel", "attract"]
```

Order must match positions 6-9.

## 7.4 `probe_metadata` structure

One object per probe:

```json
[
  {"position": 6, "is_disagreement_probe": true, "old_rule_label": "repel",   "new_rule_label": "attract"},
  {"position": 7, "is_disagreement_probe": true, "old_rule_label": "attract", "new_rule_label": "repel"},
  {"position": 8, "is_disagreement_probe": true, "old_rule_label": "attract", "new_rule_label": "repel"},
  {"position": 9, "is_disagreement_probe": true, "old_rule_label": "repel",   "new_rule_label": "attract"}
]
```

This metadata is for evaluation and audit, not for model input.

---

## 8. Model Input Format

The benchmark exposes one prompt string or one structured serialized field per episode.

## 8.1 Canonical Binary rendering

Recommended textual rendering. Example for `T2`:

```text
You are given labeled interactions between two electric charges.
Each labeled line shows q1, q2, and the observed result.
Infer the interaction pattern supported by the full sequence, then answer the final unlabeled cases.

1. q1=+2, q2=-3 -> attract
2. q1=-1, q2=-2 -> repel
3. q1=+3, q2=+1 -> repel
4. q1=-2, q2=-3 -> attract
5. q1=+1, q2=-1 -> repel
6. q1=+2, q2=+3 -> ?
7. q1=-3, q2=+1 -> ?
8. q1=+1, q2=-2 -> ?
9. q1=-1, q2=-2 -> ?

Return exactly four labels in order, each either attract or repel.
```

## 8.2 Narrative companion rendering

Required same-episode robustness rendering using the same frozen episodes and probe targets as Binary:

```text
Two electric charges were observed interacting in the following sequence.
Use the pattern best supported by the labeled observations to answer the unlabeled observations at the end.

1. A +2 charge and a -3 charge were observed to attract.
2. A -1 charge and a -2 charge were observed to repel.
3. A +3 charge and a +1 charge were observed to repel.
4. A -2 charge and a -3 charge were observed to attract.
5. A +1 charge and a -1 charge were observed to repel.
6. A +2 charge and a +3 charge were observed to ?
7. A -3 charge and a +1 charge were observed to ?
8. A +1 charge and a -2 charge were observed to ?
9. A -1 charge and a -2 charge were observed to ?

Any reasoning is optional and unscored. On the final line, return exactly four labels in order, each either attract or repel.
```

The benchmark package includes both renderings. Only Binary is leaderboard-primary.

## 8.3 Required output format

Model output must parse into exactly four labels in order:

```text
attract, repel, repel, attract
```

Acceptable parser variants may allow newline-separated outputs, but the benchmark normalizes strictly. For Narrative, any preceding reasoning text is unscored; only the final four labels are evaluated.

---

## 9. Prediction Schema

Prediction file must contain:

```text
episode_id: string
predictions: string
```

Example:

```csv
episode_id,predictions
public_leaderboard_000001,"attract,repel,repel,attract"
public_leaderboard_000002,"repel,attract,repel,attract"
```

Parsed prediction length must equal 4. Invalid outputs count as incorrect for all four probes unless a more granular parser is explicitly versioned. Narrative reasoning text must not receive any score.

---

## 10. Metrics

## 10.1 Headline metric

### Post-shift Probe Accuracy

Compute accuracy over the final 4 probes under `rule_B`.

Formula:

```text
primary_score = correct_probe_predictions / total_probes
```

Post-shift Probe Accuracy is the sole headline metric.

## 10.2 Diagnostic reporting only

Any additional metrics or slices are diagnostic-only and must not extend the benchmark claim or change the Binary headline score.

### Rule Persistence Rate (diagnostic only)

Among probes where the old rule and new rule give different answers:

```text
rule_persistence_rate = predictions_matching_old_rule / total_old_new_disagreement_probes
```

High persistence is a diagnostic signal only; it is not a standalone v1 claim.

### Transition Slice Accuracy (diagnostic only)

Separate scores for:

- `R_std -> R_inv`
- `R_inv -> R_std`

### Slice Accuracy by Difficulty (diagnostic only)

Separate scores for emitted difficulties. Under the current generator this means:

- easy
- medium

### Format Robustness Comparison (diagnostic only)

Compare Binary and Narrative performance on the same frozen episode set and probe targets. Narrative reasoning is unscored, and only the final four labels count toward the Narrative companion score.

Metrics such as adaptation lag, immediate post-shift drop, and recovery length require a future stepwise protocol and are not part of v1 reporting.

## 10.3 Confidence intervals

Use bootstrap on episodes, not individual probes.

Recommended:

- 1000 bootstrap resamples
- report 95% interval for the headline metric

---

## 11. Baseline Implementations

All baselines must use the exact same model-facing episode format or equivalent structured access.

## 11.1 Physics-prior baseline

Always answer every probe using `R_std`, regardless of evidence.

## 11.2 Never-update baseline

Infer rule from the first `pre_count` labeled examples, then apply that inferred rule to all probes without updating.

If the initial segment is ambiguous, default deterministically to `R_std`.

## 11.3 Last-example baseline

Use only the final labeled example before the probes to infer the active rule.

## 11.4 Majority-label baseline

Predict the majority class observed in the 5 labeled examples.

Tie-break deterministically.

## 11.5 Random baseline

Uniformly sample `attract` or `repel` for each probe with fixed RNG seed.

---

## 12. Validation Checks

The generator pipeline runs validation before dataset freeze. Validation includes deterministic replay checks, property tests, and regression tests against frozen reference fixtures.

## 12.1 Per-episode checks

For every episode, verify:

- exactly 9 items exist;
- the first 5 items are labeled and the last 4 items are probes;
- `template_id` is valid;
- `pre_count` and `post_labeled_count` match `template_id`;
- `shift_after_position = pre_count`;
- at least 1 post-shift labeled item contradicts `rule_A`;
- no duplicate `(q1, q2)` pair within episode;
- probe target list length is 4;
- probe metadata length is 4;
- difficulty tier is present and reproducible from stored metadata;
- deterministic regeneration from the same seed reproduces the exact serialized episode.

## 12.2 Dataset-level checks

For every split, verify:

- transition directions are near-balanced;
- template usage is near-balanced between `T1` and `T2`;
- difficulty tiers match intended proportions;
- probe labels are near-balanced;
- sign categories have coverage;
- no duplicated episodes exist across splits;
- baseline performance pattern is sensible;
- generation logic is identical across `dev`, `public_leaderboard`, and `private_leaderboard` except for frozen seed banks.

## 12.3 Property and regression checks

The codebase includes:

- property tests for rule invariance, parser determinism, metric determinism, and seed replay;
- regression tests against frozen reference fixture datasets;
- schema drift checks so field names and serialized structures cannot silently change.

## 12.4 Anti-shortcut acceptance criteria

The benchmark is not accepted unless:

- physics-prior baseline is clearly below the target model on shift-sensitive slices;
- never-update baseline fails on shift-sensitive slices;
- majority-label baseline remains near chance or clearly inferior;
- the recency shortcut baseline `last_evidence` does not dominate emitted shift-sensitive subsets, and no acceptance claim depends on an unimplemented `hard` slice;
- approved templates do not expose the shift boundary solely through end-position counting.

---

## 13. Split Specification

## 13.1 Split names

Use exactly:

- `dev`
- `public_leaderboard`
- `private_leaderboard`

## 13.2 Split purpose

### dev
For internal debugging, metric checks, and baseline verification.

### public_leaderboard
For `public_leaderboard` evaluation.

### private_leaderboard
For `private_leaderboard` evaluation and contamination resistance.

## 13.3 Seed policy

Each split uses a separate frozen seed list and manifest payload.

---

## 14. Repository Layout

Implemented benchmark-facing layout:

```text
README.md
iron_find_electric_implementation_spec.md
iron_find_electric_improved_plan.md
benchmark_design_section_cognitive_flexibility.md
packaging/
  kaggle/
    BENCHMARK_CARD.md
    PACKAGING_NOTE.md
    README.md
    frozen_artifacts_manifest.json
    iron_find_electric_v1_kaggle_staging.ipynb
reports/
  audit/
  live/
src/
  core/
  frozen_splits/
    dev.json
    public_leaderboard.json
    private_leaderboard.json
  tasks/
    iron_find_electric/
tests/
```

This repository layout already exists and is part of the implemented benchmark surface.

---

## 15. Module Responsibilities

## `src/tasks/iron_find_electric/rules.py`

Implements:

- charge sign helper
- `label(rule, q1, q2)`
- disagreement detection

## `src/tasks/iron_find_electric/generator.py`

Implements:

- episode sampling
- template assignment
- difficulty assignment
- resampling on failed constraints

## `src/tasks/iron_find_electric/schema.py`

Defines:

- episode row schema
- item schema
- probe metadata schema

## `src/tasks/iron_find_electric/render.py`

Converts structured episodes into Binary and Narrative model-facing prompt strings.

## `src/core/parser.py`

Parses model outputs into 4-label predictions.

## `src/core/metrics.py`

Computes:

- Post-shift Probe Accuracy
- diagnostic summaries
- bootstrap confidence intervals

## `src/tasks/iron_find_electric/baselines.py`

Implements all benchmark baselines.

## `src/core/validate.py`

Runs per-episode and dataset-level validity checks.

## `src/core/splits.py`

Loads frozen datasets from split manifests and supports overlap checks and audits.

## `src/core/gemini_panel.py`

Runs the active Gemini readiness path and writes canonical live evidence under `reports/live/gemini-first-panel/`.

## `src/core/anthropic_panel.py` and `src/core/openai_panel.py`

Run local-only provider panels that exist in-repo but are outside the current v1 readiness gate.

---

## 16. Kaggle Staging Interface

## 16.1 Leaderboard-primary path

```text
Binary
```

## 16.2 Required companion path

```text
Narrative
```

Narrative is required same-episode robustness evidence, but it is not leaderboard-primary.

## 16.3 Notebook behavior

The staging notebook must:

1. load frozen evaluation data;
2. expose the Binary benchmark path;
3. run the required Narrative companion evaluation on the same underlying episodes;
4. parse outputs;
5. compute Post-shift Probe Accuracy;
6. return numeric evaluation results and robustness comparison.

The staging notebook is a replay and packaging surface over existing local benchmark artifacts. It is not the place where the benchmark is first created.

## 16.4 Benchmark card minimum contents

Must state:

- track: Executive Functions;
- construct: cognitive flexibility;
- substrate: simplified two-charge interaction rules;
- headline metric: Post-shift Probe Accuracy;
- required companion path: Narrative;
- main limitations;
- baseline results;
- reproducibility details.

---

## 17. Testing Requirements

## 17.1 Unit tests

Required tests:

- `R_std` labels correctly;
- `R_inv` labels correctly;
- disagreement flag is correct;
- parser handles exact format and common safe variants;
- invalid prediction length is penalized correctly;
- bootstrap code is deterministic given seed.

## 17.2 Property tests

At least these invariants:

- swapping `q1` and `q2` preserves label;
- magnitude changes preserving sign do not change label;
- `R_std` and `R_inv` always disagree on same-sign and opposite-sign cases;
- every accepted episode satisfies all generator constraints;
- approved templates never make the shift recoverable purely by counting backward from the probe block.

## 17.3 Regression tests

Freeze reference datasets and expected metric outputs so refactors cannot silently change benchmark logic.

---

## 18. Versioning Policy

Version all of the following independently:

- rule family version;
- episode template-set version;
- generator version;
- parser version;
- metric version;
- split seed bank version.

Any change that can alter leaderboard scores requires a version bump.

---

## 19. Acceptance Criteria for v1

The v1 benchmark is readiness-aligned only if all of the following are true:

1. generator is deterministic and validated;
2. dataset schema is frozen;
3. the only approved template family is the frozen fixed-length `T1` / `T2` set;
4. template usage is balanced across splits;
5. no approved template exposes the shift boundary purely through end-position counting;
6. Post-shift Probe Accuracy is the sole headline metric;
7. Binary is the only leaderboard-primary path;
8. Narrative is included as the required same-episode robustness companion;
9. `dev`, `public_leaderboard`, and `private_leaderboard` are frozen and separated by seed bank;
10. all baselines run end to end;
11. validity checks pass;
12. current readiness evidence is Gemini-only and does not require Anthropic or OpenAI;
13. if and when Kaggle staging is attempted, the notebook runs from frozen assets already present in the repo;
14. benchmark card states scope, limitations, and v1 protocol boundaries precisely.

Phase-level recovery and lag claims remain out of scope unless the protocol is expanded to collect intermediate predictions.

---

## 20. Active Consolidation Sequence

The repository is past the build phase. The active sequence is:

### Phase 1
Documentation and status synchronization.

### Phase 2
Gemini evidence progression, with current anchor `gemini-2.5-flash` and next target `gemini-2.5-flash-lite`.

### Phase 3
Packaging and release discipline over frozen local artifacts.

---

## 21. Implemented Local Deliverable

The core deliverable is already present:

> an implemented local benchmark pipeline that generates valid episodes, assigns deterministic difficulty tiers, renders Binary and Narrative prompts, parses 4-label outputs, computes Post-shift Probe Accuracy against frozen targets, and runs baseline sanity checks.

Current remaining work is documentation, evidence progression, and packaging discipline. It is not first-time benchmark implementation.
