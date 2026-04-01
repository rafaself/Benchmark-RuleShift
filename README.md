# RuleShift Benchmark

RuleShift Benchmark is a narrow Executive Functions benchmark for cognitive flexibility. The public repository is trimmed to the maintained Kaggle release path only: the official notebook, the official binary task, the official payload, frozen split loading, packaging, and local deploy entrypoint.

## Runtime Surface

- `src/tasks/ruleshift_benchmark/`: benchmark rules, schema, generation, rendering, and protocol definitions.
- `src/core/`: Kaggle runtime helpers plus public/private frozen split loading.
- `src/frozen_splits/`: public frozen manifest for `public_leaderboard`.
- `packaging/kaggle/`: official notebook, runtime metadata, and the public packaging manifest.
- `scripts/`: the public build scripts, local deploy entrypoint, and shared packaging helpers for the runtime dataset and notebook bundle.

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
- `src/frozen_splits/public_leaderboard.json`

Within `src/core/kaggle/`, the official release path is limited to:

- `runner.py`
- `payload.py`
- `manifest.py`

The repo keeps exactly one official local deploy entrypoint:

- `python -m scripts.deploy`

The checked-in public split manifests are:

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
python3 -m pytest tests/test_kbench_notebook.py -v
python3 -m pytest tests/test_kaggle_execution.py -v
python3 -m pytest tests/test_kaggle_payload.py -v
python3 -m pytest tests/test_kaggle_audit.py -v
python3 -m pytest tests/test_public_split.py -v
python3 -m pytest tests/test_private_split.py -v
```

## Pre-publish Checks

Run the full local deploy flow without publishing:

```bash
python -m scripts.deploy --skip-publish
```

This rebuilds the public Kaggle artifacts locally before publish.

For the full local release checklist, datasource assumptions, and version-alignment rules, see [docs/kaggle-release-preflight.md](docs/kaggle-release-preflight.md).

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

Deploy from the local machine with:

```bash
python -m scripts.deploy --release-message "your Kaggle dataset version note"
```

The deploy entrypoint rebuilds the runtime dataset and notebook bundle from the same repo state, versions or creates the runtime dataset on Kaggle, and then pushes the notebook bundle.

## Private Evaluation Mount

The notebook can optionally load a mounted private dataset for `private_leaderboard`. The public runtime package never includes that artifact; it only discovers and reads an attached `private_episodes.json`.

The current operational private freeze is `270` episodes, keeping the intended `5:1` ratio relative to `public_leaderboard = 54`.

Build a local private attachment from the ignored private manifest:

```bash
python3 scripts/build_private_dataset_artifact.py --output-dir /tmp/ruleshift-private-dataset
```

To attach a local private dataset mount:

```bash
export RULESHIFT_PRIVATE_DATASET_ROOT=/tmp/ruleshift-private-dataset
```

For the operational attachment note, see [docs/private-dataset-attachment.md](docs/private-dataset-attachment.md).

## Packaging Files

- `packaging/kaggle/ruleshift_notebook_task.ipynb`
- `packaging/kaggle/kernel-metadata.json`
- `packaging/kaggle/dataset-metadata.json`
- `packaging/kaggle/frozen_artifacts_manifest.json`
