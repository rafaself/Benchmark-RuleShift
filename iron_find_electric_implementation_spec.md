# Iron Find Electric

## Concrete Implementation Spec

## 1. Objective

Implement a Kaggle Community Benchmark that measures **cognitive flexibility** through **hidden rule updating** in short two-charge binary episodes.

The implementation must make the following interpretation defensible:

> A strong score indicates that the model updated from an initial latent rule to a new latent rule after contradictory sparse evidence.

The implementation must prevent high scores from being explained mainly by:

- default electrostatics prior;
- majority-label behavior;
- constant-output behavior;
- recency-only behavior;
- formatting artifacts.

---

## 2. MVP Scope

### Included

- two charges per case;
- binary label: `attract` or `repel`;
- one hidden rule before shift;
- one hidden rule after shift;
- structured/tabular evaluation;
- deterministic generation;
- one primary leaderboard task.

### Excluded

- more than two charges;
- force magnitude prediction;
- continuous outputs;
- temporal dynamics;
- trajectory simulation;
- explanation-based primary scoring.

---

## 3. Rule System

## 3.1 Allowed charge values

Use:

```text
CHARGES = [-3, -2, -1, +1, +2, +3]
```

Zero is excluded.

## 3.2 Rule family for MVP

Only two rules are allowed in the primary benchmark:

### `R_std`
- same signs → `repel`
- opposite signs → `attract`

### `R_inv`
- same signs → `attract`
- opposite signs → `repel`

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

## 4.1 Episode template

Each episode contains exactly 9 items in this order:

1. pre-shift labeled example 1
2. pre-shift labeled example 2
3. pre-shift labeled example 3
4. post-shift labeled example 1
5. post-shift labeled example 2
6. post-shift probe 1
7. post-shift probe 2
8. post-shift probe 3
9. post-shift probe 4

Template shorthand:

```text
3 pre + 2 post-labeled + 4 probes
```

## 4.2 Hidden transition

For each episode:

- choose initial rule `rule_A ∈ {R_std, R_inv}`
- set `rule_B` to the other rule
- first 3 labeled examples use `rule_A`
- next 2 labeled examples use `rule_B`
- all 4 probes must be answered under `rule_B`

The shift is not announced explicitly.

## 4.3 No duplicate item tuples within an episode

Within a single episode, `(q1, q2)` pairs must be unique unless a later diagnostic variant explicitly allows repetition.

---

## 5. Core Generator Contract

## 5.1 Episode generation algorithm

For each episode:

1. sample `rule_A` uniformly from `{R_std, R_inv}`
2. set `rule_B` as the opposite rule
3. sample 3 unique pre-shift examples from the charge space
4. sample 2 unique post-shift labeled examples
5. sample 4 unique post-shift probes
6. verify all validity constraints
7. if any constraint fails, resample the episode

## 5.2 Charge-space definition

Ordered pairs are allowed:

```text
CASE_SPACE = {(q1, q2) | q1 ∈ CHARGES, q2 ∈ CHARGES}
```

Since label is symmetric in `q1, q2`, ordered presentation is a surface-form choice, not a semantic distinction.

## 5.3 Required per-episode constraints

Every generated episode must satisfy all of the following.

### Constraint A — contradiction after shift
At least 1 of the 2 post-shift labeled examples must directly contradict the label predicted by `rule_A`.

Operationally:

```text
label(rule_B, q1, q2) != label(rule_A, q1, q2)
```

for at least one post-shift labeled item.

### Constraint B — disagreement probes
At least 2 of the 4 probes must be disagreement probes.

A disagreement probe is a pair for which:

```text
label(R_std, q1, q2) != label(R_inv, q1, q2)
```

In this benchmark, any non-degenerate sign pattern is a disagreement case, so the implementation should still check explicitly and store this flag.

### Constraint C — balanced probe labels at dataset level
Across the full dataset, probe answers under `rule_B` should be approximately balanced between `attract` and `repel`.

### Constraint D — balanced transition direction
Across the full dataset, approximately half of episodes must be:

- `R_std → R_inv`

and half:

- `R_inv → R_std`

### Constraint E — sign-pattern coverage
Across the full dataset, the following pair categories must be represented evenly enough for stable slice analysis:

- positive / positive
- negative / negative
- positive / negative
- negative / positive

### Constraint F — no trivial probe set
The 4 probes in an episode must not all have the same correct label under `rule_B`.

---

## 6. Difficulty Tiers

Difficulty is metadata attached to each episode.

## 6.1 Easy

Requirements:

- both post-shift labeled examples contradict the pre-shift rule;
- at least 3 of 4 probes are disagreement probes;
- probe label distribution is mixed.

## 6.2 Medium

Requirements:

- exactly 1 post-shift labeled example is decisive contradiction;
- at least 2 of 4 probes are disagreement probes;
- moderate ambiguity remains after first post-shift item.

## 6.3 Hard

Requirements:

- minimal sufficient contradiction pattern;
- exactly 2 disagreement probes;
- stronger temptation for persistence or recency heuristic.

Difficulty assignment must be deterministic from generator metadata, not subjective post hoc judgment.

---

## 7. Episode Data Schema

Each episode should be represented as one structured row in the source dataset.

## 7.1 Canonical row fields

```text
episode_id: string
split: string                  # dev | public | private
difficulty: string             # easy | medium | hard
rule_A: string                 # stored internally; not shown to model
rule_B: string                 # stored internally; not shown to model
transition: string             # R_std_to_R_inv | R_inv_to_R_std
items: json/string
probe_targets: json/string
probe_metadata: json/string
```

## 7.2 `items` structure

`items` contains an ordered list of 9 dict objects:

```json
[
  {"position": 1, "phase": "pre", "kind": "labeled", "q1": 2,  "q2": -3, "label": "attract"},
  {"position": 2, "phase": "pre", "kind": "labeled", "q1": -1, "q2": -2, "label": "repel"},
  {"position": 3, "phase": "pre", "kind": "labeled", "q1": 3,  "q2": 1,  "label": "repel"},
  {"position": 4, "phase": "post", "kind": "labeled", "q1": -2, "q2": -3, "label": "attract"},
  {"position": 5, "phase": "post", "kind": "labeled", "q1": 1,  "q2": -1, "label": "repel"},
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

Order must match positions 6–9.

## 7.4 `probe_metadata` structure

One object per probe:

```json
[
  {"position": 6, "is_disagreement_probe": true,  "old_rule_label": "repel",   "new_rule_label": "attract"},
  {"position": 7, "is_disagreement_probe": true,  "old_rule_label": "attract", "new_rule_label": "repel"},
  {"position": 8, "is_disagreement_probe": true,  "old_rule_label": "attract", "new_rule_label": "repel"},
  {"position": 9, "is_disagreement_probe": true,  "old_rule_label": "repel",   "new_rule_label": "attract"}
]
```

This metadata is for evaluation and audit, not for model input.

---

## 8. Model Input Format

The benchmark should expose one prompt string or one structured serialized field per episode.

## 8.1 Canonical model-facing format

Recommended textual rendering:

```text
You are given examples of interactions between two electric charges.
Each example shows charge_1, charge_2, and the observed result.
The hidden rule may change once without warning.
Use the latest valid rule to answer the final unlabeled cases.

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

## 8.2 Required output format

Model output must parse into exactly four labels in order:

```text
attract, repel, repel, attract
```

Acceptable parser variants may allow newline-separated outputs, but the benchmark must normalize strictly.

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
public_000001,"attract,repel,repel,attract"
public_000002,"repel,attract,repel,attract"
```

Parsed prediction length must equal 4. Invalid outputs count as incorrect for all four probes unless a more granular parser is explicitly versioned.

---

## 10. Metrics

## 10.1 Primary metric

### Disagreement Probe Accuracy

Compute accuracy only over probes where `is_disagreement_probe = true`.

Formula:

```text
primary_score = correct_disagreement_probe_predictions / total_disagreement_probes
```

This is the leaderboard metric.

## 10.2 Secondary metrics

Compute and report:

### Overall Post-shift Accuracy

```text
overall_probe_accuracy = correct_probe_predictions / total_probes
```

### Rule Persistence Rate

Among disagreement probes:

```text
rule_persistence_rate = predictions_matching_old_rule / total_disagreement_probes
```

High persistence is evidence of failed adaptation.

### Adaptation Success by Transition

Separate scores for:

- `R_std → R_inv`
- `R_inv → R_std`

### Slice Accuracy by Difficulty

Separate scores for:

- easy
- medium
- hard

## 10.3 Confidence intervals

Use bootstrap on episodes, not individual probes.

Recommended:

- 1000 bootstrap resamples
- report 95% interval for primary score

---

## 11. Baseline Implementations

All baselines must use the exact same model-facing episode format or equivalent structured access.

## 11.1 Physics-prior baseline

Always answer every probe using `R_std`, regardless of evidence.

## 11.2 Never-update baseline

Infer rule from the first 3 labeled examples, then apply that inferred rule to all probes without updating.

If the first 3 examples are ambiguous, default deterministically to `R_std`.

## 11.3 Last-example baseline

Use only the final labeled example before the probes to infer the active rule.

## 11.4 Majority-label baseline

Predict the majority class observed in the 5 labeled examples.

Tie-break deterministically.

## 11.5 Random baseline

Uniformly sample `attract` or `repel` for each probe with fixed RNG seed.

---

## 12. Validation Checks

The generator pipeline must run validation before dataset freeze.

## 12.1 Per-episode checks

For every episode, verify:

- exactly 9 items exist;
- first 5 items are labeled correctly;
- last 4 items are unlabeled probes;
- at least 1 post-shift labeled item contradicts `rule_A`;
- at least 2 probes are disagreement probes;
- no duplicate `(q1, q2)` pair within episode;
- probe target list length is 4;
- probe metadata length is 4.

## 12.2 Dataset-level checks

For every split, verify:

- transition directions are near-balanced;
- difficulty tiers match intended proportions;
- probe labels are near-balanced;
- sign categories have coverage;
- baseline performance pattern is sensible;
- public/private generation logic is identical except seed bank.

## 12.3 Anti-shortcut acceptance criteria

The benchmark is not accepted unless:

- physics-prior baseline is clearly below the target model on disagreement probes;
- never-update baseline fails on shift-sensitive slices;
- majority-label baseline remains near chance or clearly inferior;
- last-example baseline does not dominate hard subset.

---

## 13. Split Specification

## 13.1 Split names

Use:

- `dev`
- `public`
- `private`

## 13.2 Split purpose

### dev
For internal debugging, metric checks, and baseline verification.

### public
For public leaderboard evaluation.

### private
For held-out audit and contamination resistance.

## 13.3 Seed policy

Each split must use a separate frozen seed list.

Example:

```text
DEV_SEEDS     = [1001, 1002, ..., 1100]
PUBLIC_SEEDS  = [2001, 2002, ..., 2600]
PRIVATE_SEEDS = [9001, 9002, ..., 9600]
```

Exact values can vary, but must be frozen and versioned.

---

## 14. File Layout

Recommended repository layout:

```text
iron_find_electric/
  README.md
  benchmark_card.md
  pyproject.toml
  src/
    rules.py
    generator.py
    schema.py
    render.py
    parser.py
    metrics.py
    baselines.py
    validate.py
    splits.py
  data/
    dev.csv
    public.csv
    private.csv
  notebooks/
    benchmark.ipynb
    diagnostics.ipynb
  tests/
    test_rules.py
    test_generator.py
    test_metrics.py
    test_parser.py
    test_validation.py
```

---

## 15. Module Responsibilities

## `rules.py`

Implements:

- charge sign helper
- `label(rule, q1, q2)`
- disagreement detection

## `generator.py`

Implements:

- episode sampling
- difficulty assignment
- resampling on failed constraints

## `schema.py`

Defines:

- episode row schema
- item schema
- probe metadata schema

## `render.py`

Converts structured episodes into model-facing prompt strings.

## `parser.py`

Parses model outputs into 4-label predictions.

## `metrics.py`

Computes:

- primary score
- secondary scores
- bootstrap confidence intervals

## `baselines.py`

Implements all benchmark baselines.

## `validate.py`

Runs per-episode and dataset-level validity checks.

## `splits.py`

Builds frozen datasets from seed lists.

---

## 16. Kaggle Benchmark Interface

## 16.1 Primary benchmark task name

```text
Adaptive Rule Updating — Binary
```

## 16.2 Notebook behavior

Main benchmark notebook must:

1. load frozen evaluation data;
2. expose one benchmark task through `%choose`;
3. call the participant model on each episode prompt;
4. parse outputs;
5. compute primary score and confidence interval;
6. return numeric evaluation results.

## 16.3 Benchmark card minimum contents

Must state:

- track: Executive Functions;
- construct: cognitive flexibility;
- substrate: simplified two-charge interaction rules;
- primary metric: Disagreement Probe Accuracy;
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
- every accepted episode satisfies all generator constraints.

## 17.3 Regression tests

Freeze a small reference dataset and expected metric outputs so refactors cannot silently change benchmark logic.

---

## 18. Versioning Policy

Version all of the following independently:

- rule family version;
- episode template version;
- generator version;
- parser version;
- metric version;
- split seed bank version.

Suggested tag format:

```text
IFE-v1.0.0
```

Any change that can alter leaderboard scores requires a version bump.

---

## 19. Acceptance Criteria for MVP

The MVP is ready only if all of the following are true:

1. generator is deterministic and validated;
2. dataset schema is frozen;
3. primary metric is disagreement-probe accuracy;
4. public/private splits are frozen and separated by seed bank;
5. all baselines run end-to-end;
6. validity checks pass;
7. benchmark notebook runs from frozen assets;
8. benchmark card states scope and limitations precisely.

---

## 20. Immediate Build Order

Implement in this exact order:

### Step 1
Build `rules.py` and unit tests.

### Step 2
Build `generator.py` with per-episode constraints.

### Step 3
Build `schema.py` and serialize sample episodes.

### Step 4
Build `render.py` and canonical prompt format.

### Step 5
Build `parser.py` and invalid-output handling.

### Step 6
Build `metrics.py` with primary and secondary scores.

### Step 7
Build `baselines.py` and run first sanity checks.

### Step 8
Build `validate.py` and dataset freeze pipeline.

### Step 9
Generate dev/public/private splits.

### Step 10
Wrap everything in the Kaggle benchmark notebook.

---

## 21. First Deliverable

The first concrete deliverable is not the full Kaggle notebook.

It is:

> a deterministic local prototype that generates valid episodes, renders prompts, parses 4-label outputs, and computes Disagreement Probe Accuracy against frozen targets.

That prototype should be completed before any benchmark packaging work.

