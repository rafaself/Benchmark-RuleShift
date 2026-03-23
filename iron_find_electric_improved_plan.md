# Iron Find Electric

## Improved Project Plan

> Status note: the repository already implements the local benchmark infrastructure for the Iron Find Electric v1 task. Phase 1 is documentation and status synchronization: align benchmark-facing documents to the implemented repo state, the current Gemini readiness path, and the actual v1 claim boundaries. This is a consolidation pass, not a protocol-build phase.

## 1. Purpose

**Iron Find Electric** is benchmark infrastructure for the implemented Iron Find Electric v1 task in the **Executive Functions** track of the **Measuring Progress Toward AGI – Cognitive Abilities** challenge.

The repository's current role is to maintain the implemented local benchmark environment, frozen split assets, Kaggle staging layer, and benchmark evidence surfaces. Kaggle is a downstream packaging layer, not the benchmark source of truth.

Core question:

> Can a model infer a latent binary rule from sparse labeled evidence, detect contradiction, revise its active rule, and apply the updated rule to final probe cases?

The electrostatics setting is only a **controlled substrate**. The benchmark is **not** intended to measure physics knowledge, simulation ability, or broad scientific reasoning.

---

## 2. Benchmark Claim

The benchmark supports the following claim and no stronger one:

> A high v1 Binary score is evidence that a model correctly applied the post-shift rule to the final probes after sparse contradictory evidence in the frozen episodes. It is not evidence of physics skill, broad adaptation ability, broad AGI capability, or general reasoning ability.

The benchmark does **not** claim that a high score demonstrates broad physical reasoning, broad executive-function coverage, human-level performance, cross-provider generality, or item-level recovery dynamics.

---

## 3. Current Readiness Surface

Current v1 readiness policy:

- the active v1 readiness evidence path is Gemini;
- the current anchor evidence is the committed Gemini panel report for requested model label `gemini-2.5-flash`;
- the current paired Gemini Flash-Lite run is canonical at `reports/live/gemini-first-panel/binary-vs-narrative/latest/`;
- the direct Flash vs Flash-Lite comparison is canonical at `reports/live/gemini-first-panel/comparison/latest/`;
- Anthropic and OpenAI integrations already exist locally, but they are outside the current v1 readiness gate;
- current v1 readiness does not require cross-provider evidence.

Evidence and packaging surfaces already present in the repo:

- implemented benchmark code under `src/`;
- frozen split manifests under `src/frozen_splits/`;
- Kaggle staging assets under `packaging/kaggle/`;
- audit and live-evidence outputs under `reports/`.

---

## 4. Construct Definition

### Primary construct

**Cognitive flexibility** operationalized as:

> Inferring a latent binary rule from few examples, revising that rule after contradiction, and applying the updated rule to final probes.

### What the benchmark measures

A successful model must:

1. infer the active rule from sparse evidence;
2. detect contradiction against the earlier hypothesis;
3. inhibit the old response pattern;
4. switch to the new rule;
5. answer final probes according to the new rule.

### What the benchmark must not accidentally measure instead

- standard electrostatics prior only;
- majority-label guessing;
- recency shortcut behavior such as `last_evidence`;
- fixed shift-position detection;
- prompt-template memorization;
- order artifacts.

---

## 5. Scope

### In scope for v1

- two charges only;
- Binary output: `attract` or `repel`;
- one hidden rule before shift;
- one hidden rule after shift;
- fixed total episode length;
- few-shot labeled evidence;
- one leaderboard-primary path;
- one required same-episode robustness companion.

### Out of scope for v1

- 3D trajectories;
- temporal simulation;
- more than two charges;
- realistic force magnitudes;
- explanation scoring;
- item-level switch-cost or recovery measurement;
- multi-ability benchmark design;
- cross-provider readiness claims.

These extensions add complexity faster than they add construct validity in the current v1 task.

---

## 6. Rule Family

### v1 rule family

Use only two relational rules in v1:

- **R_std**: same signs repel, opposite signs attract
- **R_inv**: same signs attract, opposite signs repel

This keeps the task relational while preventing constant-output solutions.

### Diagnostic-only rules

The following rules may be used later in diagnostics, but should **not** be part of the benchmark contract:

- **R_allA**: always attract
- **R_allR**: always repel

---

## 7. Episode and Split Format

### Episode format

Each evaluation unit is an **episode** with four segments:

1. **Pre-shift labeled examples** under Regime A
2. **Unsignaled shift** from Regime A to Regime B
3. **Post-shift labeled examples** under Regime B
4. **Post-shift unlabeled probes** to be answered under Regime B

Each episode contains exactly 9 items:

- 5 labeled items;
- 4 unlabeled probes.

### Split contract

Use exactly three frozen split names:

- `dev`
- `public_leaderboard`
- `private_leaderboard`

These names should be used consistently across docs, reports, manifests, and packaging surfaces.

---

## 8. Task Paths and Metric

### Leaderboard-primary path

**Binary** is the only leaderboard-primary path.

### Required robustness companion

**Narrative** is required same-episode robustness evidence. It uses the same frozen episodes and probe targets as Binary. Any reasoning is unscored. Only the final four labels are evaluated.

### Headline metric

**Post-shift Probe Accuracy** is the sole headline metric.

Diagnostic slices, disagreement analysis, parse-valid rates, and Binary-vs-Narrative comparisons remain supporting evidence only.

---

## 9. Baselines and Validity

Required baseline families:

1. **Physics Prior Baseline**
2. **Never-Update Baseline**
3. **Recency Shortcut Baseline (`last_evidence`)**
4. **Majority Label Baseline**
5. **Random Baseline**

Current validity status:

- the R13 anti-shortcut validity gate passes on the current frozen artifacts;
- the R15 re-audit keeps `last_evidence` materially bounded on leaderboard splits;
- `hard` remains reserved and un-emitted, and no acceptance claim depends on an emitted `hard` slice.

---

## 10. Repo Surfaces That Must Stay In Sync

Benchmark-facing docs should describe the existing repo surfaces accurately:

- `src/`: implemented benchmark code;
- `src/frozen_splits/`: frozen split manifests for `dev`, `public_leaderboard`, and `private_leaderboard`;
- `packaging/kaggle/`: Kaggle staging notebook, benchmark card, packaging note, and manifest;
- `reports/`: audit outputs, Gemini readiness evidence, history snapshots, and raw captures.

No active benchmark-facing doc should describe these surfaces as hypothetical, missing, or needing to be created from scratch.

---

## 11. Active Consolidation Sequence

The benchmark is implemented. The active plan is consolidation, evidence progression, and packaging discipline:

1. **Phase 1: documentation and status synchronization**
   - describe the repo as implemented benchmark infrastructure;
   - normalize benchmark-facing terminology;
   - make Gemini-only readiness gating explicit;
   - state that Anthropic and OpenAI exist locally but are outside the current readiness gate.

2. **Phase 2: Gemini evidence progression**
   - keep the current anchor evidence tied to the committed `gemini-2.5-flash` report;
   - keep the current canonical `gemini-2.5-flash-lite` paired run and direct comparison aligned to the same frozen assets and benchmark versions as the `gemini-2.5-flash` anchor;
   - keep Binary primary and Narrative as required same-episode robustness evidence.

3. **Phase 3: packaging and release discipline**
   - keep Kaggle staging aligned to frozen local artifacts;
   - preserve report organization under `reports/<context>/<target>/latest|history|samples`;
   - avoid changing benchmark semantics, scoring logic, or split composition unless explicitly requested.

### Phase 6 closeout: deferred-work boundary

Current v1 readiness remains the implemented Gemini-only gate described above. The following work is explicitly deferred beyond the current v1 readiness decision:

- **Post-v1 empirical expansion**: Anthropic live evidence, OpenAI live evidence, cross-provider comparison, and broader run-store expansion beyond the current provenance contract.
- **Longer-term scientific-validity strengthening**: human pilot, independent rerun, and protocol extensions needed for adaptation-lag or recovery claims.

Anthropic and OpenAI local runners remain available local-only integrations in the repository. They are preserved assets for later empirical work, not mistakes, and they do not block the current v1 readiness decision or the current Kaggle staging path.

---

## 12. Non-goals

Do not add the following to v1 unless they solve a clear validity problem:

- more charges because they appear more sophisticated;
- 3D or temporal motion;
- realistic electrostatic simulation;
- explanation-based main scoring;
- multiple cognitive abilities in one benchmark;
- human-level claims;
- cross-provider readiness claims.

---

## 13. Final Formulation

### Name

**Iron Find Electric**

### Track

**Executive Functions**

### One-sentence description

Implemented benchmark infrastructure for the Iron Find Electric v1 task, measuring cognitive flexibility through hidden rule updating in short two-charge binary episodes.

### Submission-safe description

Iron Find Electric evaluates whether a model can revise an inferred binary interaction rule after contradictory labeled evidence and apply the updated rule to final post-shift probes in a controlled two-charge environment, using Binary as the only leaderboard-primary path and Narrative as the required same-episode robustness companion. The repository already contains the implemented local benchmark, frozen split manifests, Kaggle staging layer, reports tree, and current Gemini evidence.

### Scope statement

This is a targeted benchmark of Executive Functions / cognitive flexibility. It is not a broad AGI claim, a broad physics-competence claim, a human-level claim, or a cross-provider claim.

---

## 14. Current Status Summary

The implemented local benchmark stack is already present in the repository:

1. `rules.py`
   - encodes the two-rule family and invariance behavior.

2. `generator.py`
   - generates deterministic episodes from seeds under the frozen template family.

3. `schema.py`
   - defines the canonical episode format, metadata fields, and versioned payload shape.

4. `render.py`
   - renders Binary and Narrative prompts from the same underlying episodes.

5. `parser.py`
   - parses model outputs deterministically.

6. `metrics.py`
   - computes Post-shift Probe Accuracy and supporting diagnostics.

7. `baselines.py`
   - implements the benchmark baselines, including `last_evidence`.

8. `validate.py` and tests
   - enforce local validity, schema stability, and deterministic replay from frozen fixtures.

9. `splits.py`
   - manages frozen `dev`, `public_leaderboard`, and `private_leaderboard` partitions plus split-overlap checks and audits.

10. `packaging/kaggle/`
   - stages the current benchmark package without redefining local benchmark semantics.

11. `reports/`
   - stores current audit outputs and Gemini readiness evidence.

### Current status

- the benchmark is already implemented;
- the active readiness evidence path is Gemini;
- the current anchor evidence is the committed `gemini-2.5-flash` report;
- the current canonical paired `gemini-2.5-flash-lite` run is present;
- the direct Flash vs Flash-Lite comparison is present;
- Anthropic and OpenAI integrations are present locally but outside the current v1 readiness gate;
- Binary is the only leaderboard-primary path;
- Narrative is the required same-episode robustness companion;
- Post-shift Probe Accuracy is the sole headline metric;
- `hard` remains reserved and not emitted.
