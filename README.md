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
- Kaggle deployment runbook: `packaging/kaggle/DEPLOY_RUNBOOK.md`
- Private split handling: `packaging/kaggle/PRIVATE_SPLIT_RUNBOOK.md`
- Official Kaggle submission surface: `packaging/kaggle/ruleshift_notebook_task.ipynb`
- Canonical Kaggle dataset metadata: `packaging/kaggle/dataset-metadata.json`
- Canonical Kaggle notebook metadata: `packaging/kaggle/kernel-metadata.json`
- Kaggle runtime-contract manifest: `packaging/kaggle/frozen_artifacts_manifest.json`

The public runtime package includes only the public code and frozen public split manifests. Never place `private_episodes.json`, private seeds, or any repo-local private fallback in public repo paths or public packaging outputs.

Kaggle deployment reads the checked-in metadata files directly. `KAGGLE_API_TOKEN` is the only required deployment secret. `KAGGLE_USERNAME`, runtime dataset slug inputs, and other deploy-time metadata overrides are not part of the current workflow.

## Kaggle Deploy Flow

The active workflows are:

- `.github/workflows/deploy-kaggle-dataset.yml`
- `.github/workflows/deploy-kaggle-notebook.yml`
- `.github/workflows/deploy-kaggle.yml`

Deploy sequence:

- Dataset deploy builds the public runtime package with `scripts/cd/build_runtime_dataset_package.py` and publishes the dataset identified by `packaging/kaggle/dataset-metadata.json`.
- Notebook deploy builds the notebook bundle with `scripts/cd/build_kernel_package.py` and pushes the notebook identified by `packaging/kaggle/kernel-metadata.json`.
- Combined deploy runs through `.github/workflows/deploy-kaggle.yml` with `target=all`, which deploys dataset first and notebook second.

Before deploy, run:

```bash
python scripts/check_public_private_isolation.py
python -m pytest tests/test_packaging.py -v
python -m pytest tests/test_cd_build.py -v
python -m pytest tests/test_kbench_notebook.py -v
```

For optional local artifact checks, build the same outputs CI deploys:

```bash
python scripts/cd/build_runtime_dataset_package.py --output-dir /tmp/ruleshift-runtime-package
python scripts/cd/build_kernel_package.py --output-dir /tmp/ruleshift-kernel-bundle
```

## Licensing

This repository uses a split licensing model:

- **Source code** — all Python source files, scripts, CI/CD workflows, tests, and notebook source code in this repository are licensed under the [Apache License 2.0](LICENSE).
- **Dataset and data artifacts** — benchmark datasets and data artifacts published as Kaggle datasets (the `raptorengineer/ruleshift-runtime` dataset and any derived data releases) are dedicated to the public domain under [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/).

The notebook source file (`packaging/kaggle/ruleshift_notebook_task.ipynb`) is source code and remains Apache-2.0 licensed. It is only treated as a data artifact if explicitly published as part of a CC0 data release.
