# Kaggle Deployment Runbook

## Canonical Metadata

The checked-in metadata files are the only source of truth for Kaggle deploys:

- `packaging/kaggle/dataset-metadata.json`
- `packaging/kaggle/kernel-metadata.json`

Current canonical ids:

- Runtime dataset: `raptorengineer/ruleshift-runtime`
- Notebook: `raptorengineer/ruleshift-notebook-task`

The deploy workflows read these files directly. They do not accept runtime inputs for ids, titles, dataset sources, or usernames.

## Required Secret

`KAGGLE_API_TOKEN` is the only required deployment secret.

Set it in GitHub Actions secrets. The workflows authenticate the Kaggle CLI with `KAGGLE_API_TOKEN` directly; they do not use `KAGGLE_USERNAME` or `kaggle.json`.

## Local Validation Before Deploy

Run the same checks the deploy workflows rely on:

```bash
python scripts/check_public_private_isolation.py
python -m pytest tests/test_packaging.py -v
python -m pytest tests/test_cd_build.py -v
python -m pytest tests/test_kbench_notebook.py -v
```

If you want a broader local pass first, also run:

```bash
make validity
make reaudit
make integrity
```

Optional local build checks:

```bash
python scripts/cd/build_runtime_dataset_package.py --output-dir /tmp/ruleshift-runtime-package
python scripts/cd/build_kernel_package.py --output-dir /tmp/ruleshift-kernel-bundle
```

## Dataset Deploy

Workflow: `.github/workflows/deploy-kaggle-dataset.yml`

Manual input:

- `release_message`: Kaggle version note for a new dataset upload

Behavior:

1. Runs `scripts/check_public_private_isolation.py`.
2. Runs `tests/test_packaging.py` and `tests/test_cd_build.py`.
3. Builds the dataset package with `scripts/cd/build_runtime_dataset_package.py`.
4. Reads `packaging/kaggle/dataset-metadata.json` for the canonical dataset id.
5. Publishes with:
   - `kaggle datasets create` if the dataset does not exist yet
   - `kaggle datasets version -m <release_message>` otherwise

## Notebook Deploy

Prerequisite: the runtime dataset referenced by `packaging/kaggle/kernel-metadata.json` must already exist on Kaggle.

Workflow: `.github/workflows/deploy-kaggle-notebook.yml`

Manual inputs: none

Behavior:

1. Runs `scripts/check_public_private_isolation.py`.
2. Runs `tests/test_packaging.py`, `tests/test_cd_build.py`, and `tests/test_kbench_notebook.py`.
3. Builds the notebook bundle with `scripts/cd/build_kernel_package.py`.
4. Copies `packaging/kaggle/ruleshift_notebook_task.ipynb` and `packaging/kaggle/kernel-metadata.json` verbatim into the bundle.
5. Publishes with `kaggle kernels push`.

## Combined Deploy

Workflow: `.github/workflows/deploy-kaggle.yml`

Manual inputs:

- `target`: `dataset`, `notebook`, or `all`
- `release_message`: required by the workflow; used by dataset deploys

Behavior:

- `target=dataset`: runs only the dataset workflow.
- `target=notebook`: runs only the notebook workflow.
- `target=all`: runs dataset first, then notebook only if dataset succeeds.

This workflow is the single manual entrypoint for a full Kaggle release sequence.
