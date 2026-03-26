# RuleShift Benchmark v1 Benchmark Card

RuleShift Benchmark v1 is a narrow Executive Functions benchmark for cognitive flexibility. It uses electrostatics only as a controlled substrate for evaluating final post-shift rule application after sparse contradictory evidence in frozen episodes.

A high Binary score is evidence that a model applied the post-shift rule to the final probes in the frozen episodes. It is not evidence of physics skill, broad adaptation ability, broad AGI capability, or general reasoning ability.

The Binary task is the scored evaluation path for the benchmark.

## Scope

- Binary as the only leaderboard-primary path: `ruleshift_benchmark_v1_binary`
- Supplemental task: Narrative audit output and supplementary same-episode robustness evidence on the same frozen episodes and probe targets as Binary
- Post-shift Probe Accuracy as the sole headline metric
- Scored targets: only the final four labels are scored
- Frozen rule family: `R_std`, `R_inv`
- Current emitted difficulty labels: `easy`, `medium`, `hard`
- Diagnostic reporting axes include `template_family`, difficulty, shift position, transition type, and error type
- Invariance reporting is diagnostic-only
- Official Kaggle submission surface: `packaging/kaggle/ruleshift_notebook_task.ipynb`

## Split Contract

- `dev`: local validation only
- `public_leaderboard`: public leaderboard evaluation
- `private_leaderboard`: held-out leaderboard evaluation from an authorized private dataset mount

The public repo and public Kaggle runtime package contain only the public splits. Private split generation, packaging, and isolation checks are defined in `PRIVATE_SPLIT_RUNBOOK.md`.

## Evidence And Limits

- Current benchmark validity is anchored by the committed R13 anti-shortcut gate and R15 re-audit artifacts.
- R13 anti-shortcut validity gate remains the main shortcut-risk check.
- R15 empirical re-audit indicates the recency shortcut was materially reduced.
- Reference baselines: `random`, `never_update`, `last_evidence`, `physics_prior`, `template_position`.
- Current readiness evidence is Gemini-only.
- Narrative is diagnostic-only and never changes the leaderboard score.
- The benchmark does **not** claim physics skill, full executive-function decomposition, switch cost, recovery length, immediate post-shift drop, online detection latency, or broad AGI capability.
