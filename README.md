# RuleShift Benchmark

RuleShift Benchmark is a narrow Executive Functions benchmark for cognitive flexibility. It evaluates whether a model applies the post-shift rule to the final probes after sparse contradictory evidence. Electrostatics is only the controlled substrate. Binary is the only leaderboard-primary path; Narrative is supplemental audit output on the same frozen episodes and never changes the leaderboard score.

This `README.md` is the main development source of truth and repo guide. Benchmark logic lives under `src/`. Kaggle packaging under `packaging/kaggle/` is downstream and must mirror, not redefine, benchmark behavior.

## Current Contract

- Benchmark scope: cognitive flexibility only.
- Leaderboard-primary task: `ruleshift_benchmark_v1_binary`.
- Binary as the only leaderboard-primary path.
- Supplemental task: Narrative audit output and supplementary same-episode robustness evidence over the same episodes and probe targets.
- Headline metric: Post-shift Probe Accuracy.
- Frozen public splits: `dev`, `public_leaderboard`.
- Held-out split: `private_leaderboard`, loaded only from an authorized private dataset mount.
- Current emitted difficulty labels: `easy`, `medium`, `hard`.

Each episode contains 5 labeled items and 4 unlabeled probes. The rule family is:

- `R_std`: same-sign charges repel, opposite-sign charges attract.
- `R_inv`: same-sign charges attract, opposite-sign charges repel.

Only charge sign matters for the correct label.

## Repository Layout

- `src/tasks/ruleshift_benchmark/`: task-specific rules, schema, generation, rendering, and baselines.
- `src/core/`: parsing, metrics, validation, audits, split loading, and runtime plumbing.
- `src/frozen_splits/`: public frozen manifests for `dev` and `public_leaderboard`, stored at `src/frozen_splits/dev.json` and `src/frozen_splits/public_leaderboard.json`.
- `packaging/kaggle/`: Kaggle-facing materials only.
- `scripts/`: operational helpers that validate isolation or build private-only artifacts without redefining benchmark semantics.
- `tests/`: benchmark, packaging, and workflow checks.

## Development

Create a local environment and install the dev requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-dev.txt
```

Run the main validation paths:

```bash
python3 -m pytest
make validity
make reaudit
make integrity
```

## Kaggle And Private Split Workflow

- Benchmark-facing summary: `packaging/kaggle/BENCHMARK_CARD.md`
- Private split handling: `packaging/kaggle/PRIVATE_SPLIT_RUNBOOK.md`
- Official Kaggle submission surface: `packaging/kaggle/ruleshift_notebook_task.ipynb`
- Kaggle runtime-contract manifest: `packaging/kaggle/frozen_artifacts_manifest.json`

The public runtime package includes only the public code and frozen public split manifests. Never place `private_episodes.json`, private seeds, or any repo-local private fallback in public repo paths or public packaging outputs.
