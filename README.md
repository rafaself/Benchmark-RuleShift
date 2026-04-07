# RuleShift Benchmark

Minimal Kaggle benchmark project for the **RuleShift** task.

This repository contains the public benchmark assets required to publish and run the benchmark on Kaggle, plus the scripts used to generate the private artifacts locally:

* the packaged public dataset
* the private dataset generator and deploy path
* the benchmark notebook
* small deploy scripts

## Repository Layout

```text
kaggle/
  dataset/
    public/
      dataset-metadata.json
      public_leaderboard_rows.json
    private/
      dataset-metadata.json
      private_answer_key.json
      private_leaderboard_rows.json
      private_split_manifest.json
  notebook/
    kernel-metadata.json
    ruleshift_notebook_task.ipynb
scripts/
  build_ruleshift_dataset.py
  deploy_dataset.sh
  deploy_private_dataset.sh
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
7. define optional diagnostics helpers outside the official score
8. mark the official entry point with `%choose`

Official task name:

```text
ruleshift_benchmark_v1_binary
```

## Release 1 Scope

Release 1 is a single-turn cognitive-flexibility benchmark. Each scored episode contains exactly 5 labeled examples, 4 probes, one response, and probe-wise scoring over `type_a` / `type_b`.

The scoped Release 1 benchmark evaluates only the `simple`, `exception`, `distractor`, and `hard` groups.

The cleaned prompt contract for the scoped benchmark is fixed: `RuleShift classification task. Episode XXXX.`, then `Examples:`, then `Probes:`, then one shared output instruction. Group differences now come from the row content rather than coaching in the prompt wrapper.

The public dataset is generated deterministically from a formal rule catalog in `scripts/build_ruleshift_dataset.py`. The private split now depends on a maintainer-only manifest at `kaggle/dataset/private/private_split_manifest.json`, and the private answer key is generated locally as `kaggle/dataset/private/private_answer_key.json`.

Dataset rows are split into explicit layers:

* `inference.prompt` is the only task content sent to the model.
* `scoring.probe_targets` is shipped only in the public sample split.
* The private split is published as inference-only rows; private scoring comes from a separate maintainer-only answer key.
* `analysis.group_id` is retained for balance checks and debugging, but is not passed to the model.

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

Select the evaluation split by setting `RULESHIFT_EVAL_SPLIT` to `public` or `private` before running the notebook.

Private scoring also requires `RULESHIFT_PRIVATE_ANSWER_KEY_PATH` to point at a local maintainer-only `private_answer_key.json`.

## Verification

Run the deterministic local verification path:

```bash
make verify-public
make verify-private
```

`make verify-public` works from a clean clone.

`make verify-private` requires the local private dataset artifacts under `kaggle/dataset/private/`, including `private_leaderboard_rows.json`, `private_answer_key.json`, `private_split_manifest.json`, and `dataset-metadata.json`. These files are intentionally gitignored and not committed to the public repository.

When both split files are present locally, verification also asserts that the private split is semantically disjoint from the public split.

## Diagnostics

The notebook keeps one simple official score and a separate optional diagnostics helper. After a run, `build_ruleshift_diagnostics(runs, leaderboard_rows)` can summarize accuracy by `group_id`, `rule_id`, `shortcut_type`, and episode miss-count pattern without affecting the benchmark score.

## Deployment

### 1. Publish the public dataset

```bash
make deploy-dataset
```

Or with a custom version message:

```bash
./scripts/deploy_dataset.sh "Update RuleShift public dataset"
```

### 2. Publish the private dataset

```bash
make deploy-private-dataset
```

Or with a custom version message:

```bash
./scripts/deploy_private_dataset.sh "Update RuleShift private dataset"
```

### 3. Publish the notebook

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
RULESHIFT_EVAL_SPLIT=public  # optional for local notebook runs
RULESHIFT_PRIVATE_ANSWER_KEY_PATH=/abs/path/to/private_answer_key.json  # required for private scoring
```

## Kaggle Asset IDs

Dataset:

```text
raptorengineer/ruleshift-runtime
```

Private dataset:

```text
raptorengineer/ruleshift-runtime-private
```

Notebook:

```text
raptorengineer/ruleshift-notebook
```

## Notes

* The notebook is the source of truth for the benchmark runtime logic and the baseline artifact.
* The public dataset contains 80 audited benchmark rows, with 20 rows per scoped group.
* The private dataset contains 400 audited benchmark rows, with 100 rows per scoped group.
* The notebook uses `EVAL_SPLIT = "public"` by default and supports `EVAL_SPLIT = "private"` when the private dataset is available.
* Benchmark invariants are enforced while rows are loaded.
* Private dataset artifacts remain local-only in this repository; `kaggle/dataset/private/` is gitignored.
* The private dataset published to Kaggle is inference-only; `private_answer_key.json` and `private_split_manifest.json` stay maintainer-only and are excluded from deploy.
* Running `scripts/build_ruleshift_dataset.py` regenerates the public payload, the inference-only private payload, the local private dataset metadata, and the local private answer key before deployment.
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
