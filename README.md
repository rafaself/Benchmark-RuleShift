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

## Version Taxonomy

Use the following roles consistently when reading or updating the repo:

| Label/category | Current labels | Represents | Status |
|---|---|---|---|
| Benchmark contract version | `RuleShift Benchmark v1`, `spec_version=v1`, `manifest_version=R14`, `generator_version=R13`, `template_set_version=v2`, `difficulty_version=R13`, `seed_bank_version=R14-dev-4` / `R14-public-4` | The implemented benchmark contract and current frozen public split state under `src/` | Current state |
| Validation / evidence release | `R13` validity gate, `R15` deterministic re-audit | Evidence about the current contract; these labels do not replace the active benchmark contract versions | Evidence |
| Packaging / deployment bundle version | `bundle_version=R16` in `packaging/kaggle/frozen_artifacts_manifest.json` | The Kaggle packaging bundle identity for deployable public artifacts | Packaging |

When a label is ambiguous on its own, prefer the role-explicit phrase: `benchmark contract version`, `validation/evidence release`, or `packaging bundle version`.

## Repository Layout

- `src/tasks/ruleshift_benchmark/`: task-specific runtime rules, schema, generation, rendering, and protocol definitions.
- `src/core/`: Kaggle runtime primitives, split loading, parsing, metrics, and payload plumbing.
- `src/maintainer/`: maintainer-only CLI, audits, contract checks, validation gates, and evidence baselines.
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

Check your environment first:

```bash
make doctor
```

This reports whether the private split is mounted and which commands are available.

### Public-safe baseline path (no private dataset required)

A fresh public clone can run the full test suite and contract audit without any private assets:

```bash
python3 -m pytest
make contract-audit
```

### Private-required commands

The following commands evaluate on the `private_leaderboard` split and require the authorized private dataset artifact (`private_episodes.json`). Run `make doctor` to confirm the private dataset is mounted before using them.

```bash
make validity        # R13 validation/evidence gate (all splits including private)
make reaudit         # R15 validation/evidence re-audit (all splits)
make integrity       # frozen split and artifact integrity (all splits)
make evidence-pass   # composite: test → validity → reaudit → integrity
```

To mount the private split locally:

```bash
export RULESHIFT_PRIVATE_DATASET_ROOT=/path/to/private-dataset
```

See `packaging/kaggle/PRIVATE_SPLIT_RUNBOOK.md` for the artifact generation workflow.

### Command matrix

| Command | Purpose | Requirement | Environment |
|---|---|---|---|
| `make test` | Run the test suite | public-safe | any |
| `make lint` | Ruff lint check | public-safe | any |
| `make type-check` | mypy type check (narrow scope) | public-safe | any |
| `make contract-audit` | P0 public artifact contract audit | public-safe | any |
| `make doctor` | Report environment status | public-safe | any |
| `make compliance-check` | Public/private isolation + notebook | public-safe | any |
| `make notebook-check` | Notebook end-to-end smoke test | public-safe | any |
| `make validity` | R13 anti-shortcut validation/evidence gate | private split | private-enabled |
| `make reaudit` | R15 deterministic validation/evidence re-audit | private split | private-enabled |
| `make integrity` | Frozen split integrity | private split | private-enabled |
| `make evidence-pass` | All checks composite | private split | private-enabled |

## Kaggle And Private Split Workflow

- Benchmark-facing summary: `packaging/kaggle/BENCHMARK_CARD.md`
- Kaggle deployment runbook: `packaging/kaggle/DEPLOY_RUNBOOK.md`
- Private split handling: `packaging/kaggle/PRIVATE_SPLIT_RUNBOOK.md`
- Official Kaggle submission surface: `packaging/kaggle/ruleshift_notebook_task.ipynb`
- Canonical Kaggle dataset metadata: `packaging/kaggle/dataset-metadata.json`
- Canonical Kaggle notebook metadata: `packaging/kaggle/kernel-metadata.json`
- Kaggle runtime-contract manifest: `packaging/kaggle/frozen_artifacts_manifest.json`

The public runtime package includes only the notebook-required source subset:

- `src/core/kaggle/`
- `src/core/parser.py`
- `src/core/metrics.py`
- `src/core/slices.py`
- `src/core/splits.py`
- `src/core/private_split.py`
- `src/core/validate/{episode.py,dataset.py}`
- `src/tasks/ruleshift_benchmark/{generator.py,protocol.py,render.py,rules.py,schema.py}`
- `src/frozen_splits/{dev.json,public_leaderboard.json}`

Maintainer-only audit, CLI, report, validation-gate, contract-audit, and baseline modules now live under `src/maintainer/` and are not staged into the public Kaggle runtime dataset. Never place `private_episodes.json`, private seeds, or any repo-local private fallback in public repo paths or public packaging outputs.

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
