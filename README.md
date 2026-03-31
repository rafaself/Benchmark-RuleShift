# RuleShift Benchmark

RuleShift Benchmark is a narrow Executive Functions benchmark for cognitive flexibility. The public repository is trimmed to the maintained Kaggle release path only: the official notebook, the official binary task, the official payload, frozen split loading, packaging, and deploy workflows.

## Runtime Surface

- `src/tasks/ruleshift_benchmark/`: benchmark rules, schema, generation, rendering, and protocol definitions.
- `src/core/`: Kaggle runtime helpers plus public/private frozen split loading.
- `src/frozen_splits/`: public frozen manifests for `dev` and `public_leaderboard`.
- `packaging/kaggle/`: official notebook, runtime metadata, and the public packaging manifest.
- `.github/workflows/`: the two official Kaggle deploy workflows.
- `scripts/`: the public build scripts and shared packaging helpers for the runtime dataset and notebook bundle.

## Official Kaggle Contract

The Kaggle release path has one official benchmark contract, emitted by the notebook for the frozen leaderboard evaluation path. The official payload contains exactly these fields:

- `score`
- `numerator`
- `denominator`
- `total_episodes`
- `benchmark_version`
- `split`
- `manifest_version`

The official contract does not include narrative result requirements, comparison fields, diagnostics summary fields, slice fields, or extra release-only metadata.

The public runtime dataset package includes only:

- `src/core/kaggle/`
- `src/core/splits.py`
- `src/core/private_split.py`
- `src/tasks/ruleshift_benchmark/{generator.py,protocol.py,render.py,rules.py,schema.py}`
- `src/frozen_splits/{dev.json,public_leaderboard.json}`

Within `src/core/kaggle/`, the official release path is limited to:

- `runner.py`
- `payload.py`
- `manifest.py`

The repo keeps exactly two deploy workflows:

- `.github/workflows/deploy-kaggle-dataset.yml`
- `.github/workflows/deploy-kaggle-notebook.yml`

The checked-in public split manifests are:

- `src/frozen_splits/dev.json`
- `src/frozen_splits/public_leaderboard.json`

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-dev.txt
python3 -m pip install -e .
```

## Local Validation

Run the release-path validation tests:

```bash
python3 -m pytest tests/test_packaging.py -v
python3 -m pytest tests/test_cd_build.py -v
python3 -m pytest tests/test_kbench_notebook.py -v
python3 -m pytest tests/test_kaggle_execution.py -v
python3 -m pytest tests/test_kaggle_payload.py -v
python3 -m pytest tests/test_run_manifest.py -v
python3 -m pytest tests/test_private_split.py -v
```

## Pre-publish Checks

Run the pre-deploy gate before any Kaggle packaging or publish action:

```bash
./scripts/pre_deploy_check.sh
```

The gate checks the local environment, runs the preflight path and targeted schema/runtime regression tests, and rebuilds the public Kaggle artifacts to catch manifest/metadata drift before release.

## Build Outputs

Build the public Kaggle runtime dataset:

```bash
python3 scripts/build_runtime_dataset_package.py --output-dir /tmp/ruleshift-runtime-package
```

Build the Kaggle notebook bundle:

```bash
python3 scripts/build_kernel_package.py --output-dir /tmp/ruleshift-kernel-bundle
```

## Kaggle Publish Flow

For the hosted Kaggle full-run checklist and post-run evidence capture, see [docs/kaggle-full-run-checklist.md](docs/kaggle-full-run-checklist.md).

1. Run all local validation tests.
2. Run `./scripts/pre_deploy_check.sh`.
3. Build runtime dataset and notebook bundle.
4. Publish the runtime dataset to Kaggle.
5. Publish or update the notebook bundle.
6. Run the notebook on Kaggle and capture evidence per the checklist.

## Private Evaluation Mount

The notebook can optionally load a mounted private dataset for `private_leaderboard`. The public repository does not generate or package that artifact; it only discovers and reads an attached `private_episodes.json`.

To attach a local private dataset mount:

```bash
export RULESHIFT_PRIVATE_DATASET_ROOT=/path/to/private-dataset
```

## Packaging Files

- `packaging/kaggle/ruleshift_notebook_task.ipynb`
- `packaging/kaggle/kernel-metadata.json`
- `packaging/kaggle/dataset-metadata.json`
- `packaging/kaggle/frozen_artifacts_manifest.json`
