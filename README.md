# RuleShift Benchmark

Minimal Kaggle benchmark project for the **RuleShift** task.

This repository contains only the assets required to publish and run the benchmark on Kaggle:

* the packaged dataset
* the benchmark notebook
* small deploy scripts

## Repository Layout

```text
kaggle/
  dataset/
    dataset-metadata.json
    public_leaderboard_rows.json
  notebook/
    kernel-metadata.json
    ruleshift_notebook_task.ipynb
scripts/
  deploy_dataset.sh
  deploy_notebook.sh
Makefile
```

## What the Notebook Does

The notebook implements the full benchmark flow in a clear, sequential format:

1. locate the packaged dataset
2. define the benchmark types and constants
3. normalize model responses
4. score each benchmark episode
5. load the frozen benchmark rows
6. register the official Kaggle task
7. run a smoke check inside the notebook
8. mark the official entry point with `%choose`

Official task name:

```text
ruleshift_benchmark_v1_binary
```

## Requirements

* Python environment with the Kaggle CLI installed
* Kaggle API token available through `.env`
* access to the Kaggle account that owns these assets

## Local Usage

Open the notebook locally:

```bash
make notelab
```

This launches Jupyter Lab with:

```text
kaggle/notebook/ruleshift_notebook_task.ipynb
```

## Deployment

### 1. Publish the dataset

```bash
make deploy-dataset
```

Or with a custom version message:

```bash
./scripts/deploy_dataset.sh "Update RuleShift dataset"
```

### 2. Publish the notebook

```bash
make deploy-notebook
```

## Environment

The deploy scripts expect:

* a `.env` file at the repository root
* `KAGGLE_API_TOKEN` defined in that file
* the Kaggle CLI available on `PATH` (i.e. `kaggle` resolves)

Override the CLI path with `KAGGLE_BIN` if needed:

```bash
KAGGLE_API_TOKEN=your_token_here
KAGGLE_BIN=/path/to/kaggle   # optional; defaults to kaggle in PATH
```

## Kaggle Asset IDs

Dataset:

```text
raptorengineer/ruleshift-runtime
```

Notebook:

```text
raptorengineer/ruleshift-notebook
```

## Notes

* The notebook is the source of truth for the benchmark runtime logic.
* The dataset contains the frozen public benchmark rows used during evaluation.
* The repository is intentionally kept small and Kaggle-oriented, with minimal abstraction and minimal supporting files.

## References

- [Kaggle Competition — Measuring Progress Toward AGI: Cognitive Abilities](https://www.kaggle.com/competitions/kaggle-measuring-agi)
- [Competition Rules](https://www.kaggle.com/competitions/kaggle-measuring-agi/rules)
- [Kaggle Benchmarks Repository](https://github.com/Kaggle/kaggle-benchmarks)
- [Kaggle Benchmarks Cookbook](https://github.com/Kaggle/kaggle-benchmarks/blob/ci/cookbook.md)
- [Kaggle Notebooks Documentation](https://www.kaggle.com/docs/notebooks)
- [Kaggle Public API / CLI Documentation](https://www.kaggle.com/docs/api)
- [Kaggle CLI Repository](https://github.com/Kaggle/kaggle-cli)
- [Kaggle CLI — General Docs](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md)
- [Kaggle CLI — Kernels / Notebooks Commands](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md)
- [DeepMind Paper PDF — Measuring Progress Toward AGI: A Cognitive Framework](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/measuring-progress-toward-agi/measuring-progress-toward-agi-a-cognitive-framework.pdf)
- [DeepMind Blog Post — Measuring progress toward AGI: A cognitive framework](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/measuring-agi-cognitive-framework/)