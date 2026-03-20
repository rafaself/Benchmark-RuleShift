# Iron Find Electric

## Improved Project Plan

## 1. Purpose

**Iron Find Electric** is a Kaggle Community Benchmark for the **Measuring Progress Toward AGI – Cognitive Abilities** challenge.

It targets **cognitive flexibility** in the **Executive Functions** track.

Core question:

> Can a model infer a latent rule from sparse evidence, detect an unsignaled rule change, inhibit the previous response policy, and apply the new rule on conflict cases?

The electrostatics setting is only a **controlled substrate**. The benchmark is **not** intended to measure physics knowledge, simulation ability, or general scientific reasoning.

---

## 2. Benchmark Claim

The benchmark should support the following claim and no stronger one:

> A high score indicates successful adaptation to a hidden rule shift from sparse labeled evidence in a simple, controlled binary environment.

The benchmark should **not** claim that a high score demonstrates broad physical reasoning or broad AGI capability.

---

## 3. Challenge Alignment

This project is aligned if it satisfies four conditions:

1. It measures a clearly named cognitive ability inside one official track.
2. It is implementable as a Kaggle Community Benchmark.
3. It yields reproducible scores from frozen benchmark logic.
4. A strong score is interpretable as evidence of **rule updating**, not shortcut exploitation.

Alignment test for every design decision:

> If a model scores well, is that only plausibly explained by adaptation to a hidden rule change, rather than by electrostatics priors, recency, majority-label guessing, or benchmark artifacts?

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
5. answer new probes according to the new rule.

### What the benchmark must not accidentally measure instead

- standard electrostatics prior only;
- majority-label guessing;
- last-example copying;
- fixed shift-position detection;
- prompt-template memorization;
- order artifacts.

---

## 5. Scope

### In scope for MVP

- two charges only;
- binary output: `attract` or `repel`;
- hidden rule shift;
- few-shot labeled evidence;
- short episodes;
- tabular / structured evaluation format;
- one primary leaderboard task.

### Out of scope for MVP

- 3D trajectories;
- temporal simulation;
- more than two charges;
- realistic force magnitudes;
- open-ended explanation scoring;
- multi-ability benchmark design.

These extensions add complexity faster than they add construct validity.

---

## 6. Rule Family

### MVP leaderboard rule family

Use only two relational rules in the primary task:

- **R_std**: same signs repel, opposite signs attract
- **R_inv**: same signs attract, opposite signs repel

This is the cleanest MVP because both rules depend on the **relation between the two charges**. A model cannot solve the task by following a constant output policy.

### Diagnostic-only rules

The following rules may be used later in diagnostics, but should **not** be part of the first leaderboard task:

- **R_allA**: always attract
- **R_allR**: always repel

These are useful for stress-testing heuristics, but they weaken construct purity if included in the core leaderboard because they permit non-relational shortcut behavior.

---

## 7. Case and Episode Format

### Atomic case

Each atomic case consists of:

- `q1 ∈ {-3, -2, -1, +1, +2, +3}`
- `q2 ∈ {-3, -2, -1, +1, +2, +3}`
- label in `{attract, repel}`

Magnitude is present only to prevent the benchmark from collapsing into a trivial visual token rule, but magnitudes do not affect the correct label.

### Episode format

Each evaluation unit is an **episode** with four segments:

1. **Pre-shift labeled examples** under Regime A
2. **Unsignaled shift** from Regime A to Regime B
3. **Post-shift labeled examples** under Regime B
4. **Post-shift unlabeled probes** to be answered under Regime B

### Fixed MVP episode template

For the first benchmark version, use a frozen template:

- 3 pre-shift labeled examples
- 2 post-shift labeled examples
- 4 post-shift probes

Total: **9 items per episode**

This is enough to create sparse evidence without making episodes long or noisy.

---

## 8. Generator Requirements

The generator must be deterministic, auditable, and explicitly designed to block shortcuts.

### 8.1 Episode generation rules

For every episode:

- choose Regime A uniformly from `{R_std, R_inv}`
- choose Regime B as the opposite rule
- sample pre-shift and post-shift examples from the same charge space
- vary shift position only across benchmark versions or episode templates, not within the item list semantics
- avoid duplicate examples unless explicitly needed for diagnostics

### 8.2 Conflict requirement

Every episode must contain at least:

- **2 disagreement probes** where `R_std` and `R_inv` predict different answers
- **1 decisive post-shift labeled example** that directly contradicts the pre-shift rule

This is the most important generator constraint. Without it, a model can score well without showing real adaptation.

### 8.3 Balance requirements

Across the dataset:

- balance `attract` and `repel`
- balance initial regime direction (`R_std → R_inv` and `R_inv → R_std`)
- balance sign patterns across episodes
- balance probe difficulty tiers

### 8.4 Invariance requirements

- swapping `q1` and `q2` must never change the label
- surface formatting must not alter the answer
- equivalent episodes should remain equivalent under harmless re-orderings where appropriate

---

## 9. Difficulty Tiers

Difficulty should be defined operationally, not rhetorically.

### Easy

- strong contradiction after shift
- disagreement probes dominate post-shift probes
- post-shift evidence is clean and redundant

### Medium

- contradiction is present but less redundant
- mix of disagreement and non-disagreement probes
- moderate temptation for prior-following behavior

### Hard

- minimal but sufficient shift evidence
- disagreement probes still present but fewer
- stronger temptation for old-rule persistence or recency heuristics

Difficulty metadata should be attached per episode and used in slice reporting.

---

## 10. Metrics

## 10.1 Primary leaderboard metric

Use:

### **Disagreement Probe Accuracy**

Accuracy computed only on post-shift probes for which the pre-shift and post-shift rules would give different answers.

Why this should be primary:

- it directly measures successful rule updating;
- it minimizes inflation from probes that both rules answer the same way;
- it makes the benchmark claim more defensible.

## 10.2 Secondary metrics

Report but do not optimize the leaderboard around:

- **Overall Post-shift Accuracy**
- **Rule Persistence Rate**: fraction of disagreement probes answered according to the old rule
- **Adaptation Lag**: number of post-shift labeled examples needed before predictions align with the new rule
- **Slice Accuracy** by transition direction and difficulty tier

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
   Infer one initial rule and continue using it after the shift.

3. **Last-Example Baseline**  
   Overweight only the latest labeled example.

4. **Majority Label Baseline**  
   Always output the majority class.

5. **Random Baseline**  
   Uniform random over `{attract, repel}`.

Expected pattern:

- baselines may perform non-trivially on easy subsets;
- they should fail clearly on disagreement-probe slices;
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
- positional artifacts.

### Required controls

- disagreement probes in every episode;
- balanced label marginals;
- variable sign combinations;
- no constant-rule shortcuts in the primary task;
- adversarial subsets where standard-prior and never-update fail;
- frozen and versioned templates.

### Heuristics-fail subsets

At minimum, include evaluation slices where:

- standard-prior fails;
- never-update fails;
- last-example fails;
- majority-label fails.

These subsets are not optional diagnostics. They are part of the validity argument.

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
- template version
- dataset seed list
- difficulty assignment logic
- metric code

### Contamination control

The private audit set should be generated from frozen logic and held outside normal prompt iteration and benchmark tuning.

---

## 14. Failure Analysis

The benchmark should support diagnosis of at least five failure modes:

1. **Initial inference failure**  
   Model fails before the shift.

2. **Change-detection failure**  
   Model does not react to contradictory post-shift evidence.

3. **Old-rule persistence**  
   Model continues using the original rule on disagreement probes.

4. **Recency overshoot**  
   Model follows the latest example without stable rule revision.

5. **Format fragility**  
   Model succeeds only in one prompt surface form.

This is why disagreement probes, lag, and sliced reporting are necessary.

---

## 15. Validity Extensions

These are important, but they are **Phase 2–3 items**, not blockers for the first submission.

### Human baseline

Run a small adult pilot using the exact same episodes and response format.

Use it to support calibration, not grand claims.

### Alternate format

Add a semantically equivalent alternate presentation format, such as a lightly verbalized version of the same structured task.

Purpose: test whether the benchmark measures rule updating rather than one specific surface template.

### Independent verification

Obtain external review or independent rerun of:

- episode generator
- labeling logic
- metric code

---

## 16. Kaggle Implementation Plan

### Primary task

**Adaptive Rule Updating — Binary**

This is the only leaderboard task in the first submission.

Its score should be based on **Disagreement Probe Accuracy**.

### Optional companion tasks

1. **Shift Detection & Rule Switching**  
   Diagnostic task to separate detection failure from application failure.

2. **Adaptive Rule Updating — Alternate Format**  
   Robustness task for surface variation.

### Packaging

- one main notebook with `%choose` pointing to the primary task
- benchmark card explaining construct, scope, and limitations
- versioned dataset artifacts
- baseline results table

---

## 17. Step-by-Step Roadmap

## Phase 0 — Freeze the framing

Deliverables:

- final construct statement
- track declaration: **Executive Functions**
- benchmark claim
- explicit non-goals

Exit condition:

- no wording suggests this is mainly a physics benchmark
- no wording overclaims beyond cognitive flexibility under hidden rule shift

## Phase 1 — Build the MVP benchmark

Deliverables:

- deterministic generator
- deterministic labeler
- fixed episode template
- disagreement-probe primary metric
- baseline suite
- adversarial evaluation slices
- reproducible public/dev split logic

Exit condition:

- a high score is not plausibly explained by standard-prior, majority-label, or never-update shortcuts

## Phase 2 — Add robustness

Deliverables:

- alternate-format companion task
- adaptation-lag diagnostics
- private audit split
- stronger slice reporting by difficulty and transition direction

Exit condition:

- robustness additions strengthen the same construct instead of changing the task into something broader

## Phase 3 — Strengthen scientific validity

Deliverables:

- small human pilot
- independent audit or rerun
- confidence interval reporting in benchmark card
- contamination statement

Exit condition:

- stronger claims are made only when supporting evidence exists

---

## 18. Non-goals

Do not add the following to the MVP unless they solve a clear validity problem:

- more charges because it appears more sophisticated;
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

Iron Find Electric evaluates whether a model can revise an inferred binary interaction rule after an unsignaled rule shift from sparse labeled evidence, using a controlled two-charge environment and a primary score based on post-shift disagreement probes.

### Scope statement

This is a targeted benchmark of cognitive flexibility within the challenge framework, not a standalone measure of AGI progress.

