# Iron Find Electric

## Improved Project Plan

## 1. Purpose

**Iron Find Electric** is a Kaggle Community Benchmark for the **Measuring Progress Toward AGI – Cognitive Abilities** challenge.

It targets **cognitive flexibility** in the **Executive Functions** track.

Core question:

> Can a model infer a latent binary rule from sparse labeled evidence, detect contradiction, revise its active rule, and apply the updated rule to final probe cases?

The electrostatics setting is only a **controlled substrate**. The benchmark is **not** intended to measure physics knowledge, simulation ability, or broad scientific reasoning.

---

## 2. Benchmark Claim

The benchmark should support the following claim and no stronger one:

> A high v1 score indicates successful application of an updated hidden rule to final post-shift probes after sparse contradictory evidence in a simple controlled binary environment.

The benchmark should **not** claim that a high score demonstrates broad physical reasoning, broad AGI capability, or item-level recovery dynamics.

---

## 3. Challenge Alignment

This project is aligned if it satisfies four conditions:

1. It measures a clearly named cognitive ability inside one official track.
2. It is implementable as a Kaggle Community Benchmark.
3. It yields reproducible scores from frozen benchmark logic.
4. A strong score is interpretable as evidence of **rule updating**, not shortcut exploitation.

Alignment test for every design decision:

> If a model scores well, is that only plausibly explained by adaptation to a hidden rule change, rather than by electrostatics priors, majority-label guessing, last-example copying, prompt artifacts, or structural template cues?

---

## 4. Construct Definition

### Primary construct

**Cognitive flexibility** operationalized as:

> Inferring a latent binary rule from few examples, revising that rule after contradiction, and applying the updated rule to new probes.

### What the benchmark must measure

A successful model must:

1. infer the active rule from sparse evidence;
2. detect contradiction against the earlier hypothesis;
3. inhibit the old response pattern;
4. switch to the new rule;
5. answer final probes according to the new rule.

### What the benchmark must not accidentally measure instead

- standard electrostatics prior only;
- majority-label guessing;
- last-example copying;
- fixed shift-position detection;
- prompt-template memorization;
- order artifacts.

---

## 5. Scope

### In scope for v1

- two charges only;
- binary output: `attract` or `repel`;
- one hidden rule before shift;
- one hidden rule after shift;
- fixed total episode length;
- few-shot labeled evidence;
- one primary leaderboard task;
- one required non-leaderboard robustness task.

### Out of scope for v1

- 3D trajectories;
- temporal simulation;
- more than two charges;
- realistic force magnitudes;
- explanation scoring;
- item-level switch-cost or recovery measurement;
- multi-ability benchmark design.

These extensions add complexity faster than they add construct validity in the first release.

---

## 6. Rule Family

### v1 rule family

Use only two relational rules in v1:

- **R_std**: same signs repel, opposite signs attract
- **R_inv**: same signs attract, opposite signs repel

This keeps the task relational while preventing constant-output solutions.

### Diagnostic-only rules

The following rules may be used later in diagnostics, but should **not** be part of the first benchmark package:

- **R_allA**: always attract
- **R_allR**: always repel

These are useful for stress-testing heuristics, but they weaken construct purity if included in the core task.

---

## 7. Case and Episode Format

### Atomic case

Each atomic case consists of:

- `q1 ∈ {-3, -2, -1, +1, +2, +3}`
- `q2 ∈ {-3, -2, -1, +1, +2, +3}`
- label in `{attract, repel}`

Magnitude is present only to keep the representation from collapsing into a trivial token rule. Magnitudes do not affect the correct label.

### Episode format

Each evaluation unit is an **episode** with four segments:

1. **Pre-shift labeled examples** under Regime A
2. **Unsignaled shift** from Regime A to Regime B
3. **Post-shift labeled examples** under Regime B
4. **Post-shift unlabeled probes** to be answered under Regime B

### Frozen v1 template family

Use a small frozen, versioned template set with fixed total length:

- **T1**: 2 pre-shift labeled + 3 post-shift labeled + 4 probes
- **T2**: 3 pre-shift labeled + 2 post-shift labeled + 4 probes

Total: **9 items per episode**

The benchmark must not use any template whose shift boundary can be recovered solely by counting backward from the final probe block.

---

## 8. Generator Requirements

The generator must be deterministic, auditable, and explicitly designed to block shortcuts.

### 8.1 Episode generation rules

For every episode:

- choose `rule_A` uniformly from `{R_std, R_inv}`;
- choose `rule_B` as the opposite rule;
- choose `template_id` from `{T1, T2}`;
- derive `pre_count` and `post_labeled_count` from the template;
- set `shift_after_position = pre_count`;
- sample pre-shift, post-shift, and probe items from the same charge space;
- avoid duplicate examples unless a later diagnostic variant explicitly allows repetition.

### 8.2 Conflict requirement

Every episode must contain at least:

- **1 decisive post-shift labeled example** that directly contradicts the pre-shift rule;
- a nontrivial probe block whose correct answers are not all identical.

Without contradiction after the shift, a model can score well without showing real adaptation.

### 8.3 Balance requirements

Across the dataset:

- balance transition direction (`R_std → R_inv` and `R_inv → R_std`);
- balance template usage (`T1` and `T2`);
- balance `attract` and `repel` on probes;
- balance sign patterns across episodes;
- balance difficulty tiers.

### 8.4 Invariance and anti-shortcut requirements

- swapping `q1` and `q2` must never change the label;
- surface formatting must not alter the answer;
- template design must not expose the shift boundary by end-position counting;
- equivalent episodes should remain equivalent under harmless re-renderings.

---

## 9. Difficulty Tiers

Difficulty should be defined operationally, not rhetorically.

### Easy

- strong contradiction after shift;
- redundant post-shift evidence;
- probe labels are mixed and easy to separate from shortcut behavior.

### Medium

- contradiction is present but less redundant;
- moderate ambiguity remains after the first post-shift labeled item;
- probe mix creates meaningful temptation for prior-following behavior.

### Hard

- minimal but sufficient contradiction pattern;
- less redundant post-shift evidence;
- stronger temptation for old-rule persistence or last-example heuristics.

Difficulty metadata should be attached per episode and used in slice reporting.

---

## 10. Metrics

## 10.1 Primary leaderboard metric

Use:

### **Post-shift Probe Accuracy**

Accuracy computed over the final four probes under `rule_B`.

Why this should be primary in v1:

- it directly reflects success on the scored post-shift task;
- it is honest about what the current protocol actually observes;
- it keeps the benchmark claim defensible without overstating disagreement-only scoring.

## 10.2 Secondary metrics

Report but do not optimize the leaderboard around:

- **Rule Persistence Rate**: fraction of probes answered according to the old rule when old-rule and new-rule answers differ;
- **Transition Slice Accuracy** by `R_std → R_inv` and `R_inv → R_std`;
- **Difficulty Slice Accuracy** by `easy`, `medium`, and `hard`;
- **Format Robustness Comparison** between Binary and Narrative renderings.

### Future diagnostic metrics

Metrics such as adaptation lag, recovery length, immediate post-shift drop, and switch cost require a later stepwise protocol that collects intermediate predictions. They are not part of the v1 leaderboard claim.

## 10.3 Uncertainty reporting

Report bootstrap confidence intervals for the primary metric.

This improves reproducibility and fits the challenge’s emphasis on uncertainty-aware measurement.

---

## 11. Baselines

The benchmark should explicitly test rival explanations.

Required baselines:

1. **Physics Prior Baseline**  
   Always apply `R_std`.

2. **Never-Update Baseline**  
   Infer one initial rule from the first `pre_count` labeled examples and continue using it after the shift.

3. **Last-Example Baseline**  
   Overweight only the final labeled example before the probe block.

4. **Majority Label Baseline**  
   Always output the majority class observed in the five labeled examples.

5. **Random Baseline**  
   Uniform random over `{attract, repel}`.

Expected pattern:

- baselines may perform non-trivially on easy subsets;
- they should fail clearly on shift-sensitive slices;
- a strong model should beat them with margin on the main metric.

---

## 12. Anti-shortcut Design

### Threat model

The benchmark must assume the model may exploit:

- electrostatics prior;
- recency;
- majority labels;
- constant output behavior;
- fixed template structure;
- positional artifacts;
- format-specific prompt cues.

### Required controls

- contradiction after the hidden shift in every episode;
- balanced label marginals;
- variable sign combinations;
- no constant-rule shortcuts in the primary task;
- adversarial subsets where standard-prior and never-update fail;
- frozen and versioned template sets;
- a required narrative-format robustness companion task.

### Heuristics-fail subsets

At minimum, include evaluation slices where:

- standard-prior fails;
- never-update fails;
- last-example fails;
- majority-label fails.

These subsets are part of the validity argument, not optional extras.

---

## 13. Splits and Reproducibility

### Split structure

Use three splits:

- **Dev**: generator debugging and internal checks
- **Public Eval**: leaderboard-facing benchmark set
- **Private Audit**: held-out validity set

### Reproducibility protocol

Freeze and record:

- generator version
- rule-family version
- template-set version
- dataset seed list
- difficulty assignment logic
- metric code

### Contamination control

The private audit set should be generated from frozen logic and held outside normal prompt iteration and benchmark tuning.

---

## 14. Failure Analysis

The benchmark should support diagnosis of at least four v1 failure modes:

1. **Initial inference failure**  
   Model never infers the starting rule from the early labeled items.

2. **Change-detection failure**  
   Model does not revise after contradictory post-shift evidence.

3. **Old-rule persistence**  
   Model continues using the original rule on probes where old-rule and new-rule predictions diverge.

4. **Format fragility**  
   Model succeeds in the Binary rendering but degrades materially in the Narrative rendering.

Item-level recovery and lag analysis require a future stepwise protocol rather than the current final-probe-only outputs.

---

## 15. Validity Extensions

### Required in v1

#### Alternate format

Include a semantically equivalent **Adaptive Rule Updating — Narrative** task in the first benchmark package.

Purpose: test whether the benchmark measures rule updating rather than performance on one specific surface template.

This companion task is required for validity evidence, but it is **not** leaderboard-primary.

### Phase 2–3 extensions

#### Human baseline

Run a small adult pilot using the exact same episodes and response format.

Use it to support calibration, not grand claims.

#### Stepwise diagnostics

Add a protocol variant that collects intermediate predictions if switch cost, recovery length, or adaptation lag become required benchmark claims.

#### Independent verification

Obtain external review or independent rerun of:

- episode generator
- labeling logic
- metric code

---

## 16. Kaggle Implementation Plan

### Primary task

**Adaptive Rule Updating — Binary**

This is the only leaderboard-primary task in the first submission.

Its score should be based on **Post-shift Probe Accuracy**.

### Required companion task

**Adaptive Rule Updating — Narrative**

This uses the same underlying episodes and targets with a narrative-format rendering. It is required in v1 for robustness evidence, but it is not leaderboard-primary.

### Packaging

- one main notebook with `%choose` pointing to the primary task;
- one required companion evaluation for the narrative rendering;
- benchmark card explaining construct, scope, limitations, and v1 protocol boundaries;
- versioned dataset artifacts;
- baseline results table.

---

## 17. Step-by-Step Roadmap

## Phase 0 — Freeze the framing

Deliverables:

- final construct statement;
- track declaration: **Executive Functions**;
- benchmark claim;
- explicit non-goals;
- frozen v1 template set and metric definitions.

Exit condition:

- no wording suggests this is mainly a physics benchmark;
- no wording overclaims beyond final post-shift rule updating under hidden shift.

## Phase 1 — Build the v1 benchmark

Deliverables:

- deterministic generator;
- deterministic labeler;
- frozen `T1` / `T2` episode template family;
- deterministic difficulty assignment implemented in generator logic;
- canonical frozen schema and metadata fields;
- `Post-shift Probe Accuracy` primary metric;
- Binary task;
- Narrative companion task;
- baseline suite;
- adversarial evaluation slices;
- validation scripts plus property/regression tests;
- frozen split seed banks and version metadata;
- reproducible dev/public/private split logic.

Exit condition:

- a high score is not plausibly explained by standard-prior, majority-label, never-update, end-counting, or other simple template heuristics;
- deterministic regeneration, validation, and regression checks pass before split freeze.

## Phase 2 — Add deeper diagnostics

Deliverables:

- stepwise prediction protocol;
- adaptation-lag diagnostics;
- stronger slice reporting by difficulty and transition direction;
- richer failure analysis.

Exit condition:

- new diagnostics strengthen the same construct instead of broadening the task into a different benchmark.

## Phase 3 — Strengthen scientific validity

Deliverables:

- small human pilot;
- independent audit or rerun;
- confidence interval reporting in benchmark card;
- contamination statement.

Exit condition:

- stronger claims are made only when supporting evidence exists.

---

## 18. Non-goals

Do not add the following to v1 unless they solve a clear validity problem:

- more charges because they appear more sophisticated;
- 3D or temporal motion;
- realistic electrostatic simulation;
- explanation-based main scoring;
- multiple cognitive abilities in one benchmark.

---

## 19. Final Formulation

### Name

**Iron Find Electric**

### Track

**Executive Functions**

### One-sentence description

A Kaggle Community Benchmark that measures cognitive flexibility through hidden rule updating in short two-charge binary episodes.

### Submission-safe description

Iron Find Electric evaluates whether a model can revise an inferred binary interaction rule after contradictory labeled evidence and apply the updated rule to final post-shift probes in a controlled two-charge environment, using one leaderboard-primary binary task and one required non-leaderboard narrative robustness task.

### Scope statement

This is a targeted benchmark of cognitive flexibility within the challenge framework, not a standalone measure of AGI progress or a direct measure of item-level recovery dynamics.

---

## 20. Immediate Execution Order

Implement the first executable version in this order:

1. `rules.py`
   - encode the two-rule family;
   - add invariance and unit tests;
   - verify label consistency.

2. `generator.py`
   - generate episodes deterministically from seed;
   - enforce `T1` / `T2` template constraints;
   - assign deterministic difficulty tiers from explicit metadata;
   - ensure pre-shift and post-shift structure is valid.

3. `schema.py`
   - define the canonical episode format;
   - freeze field names, metadata, and version tags.

4. `render.py`
   - render Binary task prompts;
   - render Narrative robustness prompts.

5. `parser.py`
   - parse model outputs safely;
   - handle malformed answers deterministically.

6. `metrics.py`
   - compute `Post-shift Probe Accuracy`;
   - compute subset and diagnostic metrics.

7. `baselines.py`
   - implement never-update, always-update / last-evidence, physics-prior, template-position heuristic, and random baselines;
   - verify shortcut baselines fail on intended hard slices.

8. `validate.py` and tests
   - check no invalid episodes, no schema drift, no leakage, no split duplication, and balanced metadata distributions;
   - add property/regression tests and a frozen reference fixture so refactors cannot silently change behavior.

9. `splits.py`
   - freeze dev, public, and private splits;
   - freeze separate seed banks and version identifiers for generator, templates, metrics, parser, and difficulty logic.

10. Kaggle packaging
   - prepare the benchmark notebook and benchmark card only after the local prototype is stable.

### Definition of done for the next milestone

The next milestone is complete when:

- generator outputs valid frozen-format episodes;
- deterministic difficulty assignment is attached to every episode;
- renderer and parser work end-to-end;
- metric computation is stable;
- baseline behaviors are measurable;
- at least one hard slice clearly defeats shortcut baselines;
- validation, property tests, and regression tests pass from frozen seeds.

