## Design implications from Braem & Egner (2018)

### Benchmark framing
This benchmark should be framed as a **task-switching benchmark** targeting **cognitive flexibility under latent rule change**. The goal is not to measure abstract reasoning in general, but the model's ability to reconfigure its active policy when the governing rule changes without explicit notification.

### Construct definition
The target construct should be defined narrowly as:

**Cognitive flexibility under hidden rule shift**

More specifically, the benchmark should measure whether a model can:

- maintain a currently active rule,
- detect evidence that the rule is no longer valid,
- disengage from the previous rule,
- and adapt to the new rule with minimal delay and perseveration.

This wording is preferable to broader claims such as “general flexibility” or “pure reasoning,” because the literature suggests that flexibility is strongly shaped by contextual learning and control-state adaptation.

### Design requirements
1. **Use controlled latent shifts.** Rule changes should be generated and tracked by the evaluator, but never explicitly announced to the model.
2. **Prevent predictable switch rhythms.** Shift timing should not follow simple periodic or locally exploitable patterns.
3. **Eliminate superficial contextual cues.** Layout, ordering, surface form, token patterns, and presentation details must not reveal when a shift has occurred.
4. **Keep reasoning demands bounded.** The task should require some inference, but the dominant challenge must remain adaptation after a regime change, not solving a deeply complex rule.
5. **Test out-of-context generalization.** Include evaluation subsets where the same hidden-shift logic appears under novel surface conditions.

### Core evaluation metrics
Aggregate accuracy is insufficient. The benchmark should report adaptation-sensitive metrics, including:

- **Pre-shift accuracy**
- **Immediate post-shift drop**
- **Recovery length**: number of examples required to recover stable performance
- **Perseveration rate**: tendency to continue applying the old rule after the shift
- **Adaptation efficiency**: how quickly performance stabilizes under the new rule

These metrics should be reported by phase:

- **Pre-shift**
- **Transition window**
- **Post-shift recovery**

### Validity risks to document
The benchmark specification should explicitly acknowledge the following threats to construct validity:

- exploitation of local shift-frequency statistics,
- reliance on contextual or formatting cues,
- memorization of surface associations rather than policy adaptation,
- strong final accuracy without fast recovery after change,
- benchmark behavior dominated by reasoning complexity instead of switching behavior.

### Recommended interpretation policy
A high score should not be interpreted as evidence of cognitive flexibility unless the model also shows:

- low perseveration,
- a clear but bounded post-shift disruption,
- and rapid recovery after the hidden rule change.

In other words, success should be attributed to **adaptive control reconfiguration**, not merely to overall accuracy.

### Bottom line
The benchmark should be designed and described as a test of **adaptive task-set reconfiguration under latent regime change**, with evaluation centered on **switch cost, recovery, and robustness against contextual shortcuts**.

