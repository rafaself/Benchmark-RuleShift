# RuleShift Benchmark

A focused LLM benchmark targeting a single sub-skill of cognitive flexibility: rule updating under conflicting evidence. This repository contains only the maintained Kaggle release path — the official notebook, the official binary task, the official payload contract, frozen split loading, packaging, and the local deploy entrypoint.

## What This Benchmark Measures

RuleShift measures one narrow operation within cognitive flexibility: whether an LLM can detect that a previously inferred rule has been replaced by a new one, based on conflicting labeled evidence, and predict correctly under the updated rule.

Each episode is a sequence of 9 items over two markers (`r1`, `r2`). Items 1-5 are labeled with the observed state (`zark` or `blim`). Items 6-9 are unlabeled probes the model must answer with `type_a` or `type_b`, where `zark = type_a` and `blim = type_b`. Two latent rules govern the states, and partway through the labeled items the active rule silently shifts to its inverse, so some post-shift labels contradict the pattern established by the pre-shift items.

Public contract lock: `markers-v1`.
This lock fixes the public surface to `markers`, `r1`/`r2`, `zark`/`blim`, and `type_a`/`type_b`.
The canonical benchmark version remains `R14` because the latent task, scoring, and seed schedule are unchanged.

The model must:

1. Infer the initial rule from the first labeled items.
2. Recognize that later labels conflict with that rule.
3. Update its inference to the new rule.
4. Predict the 4 probe labels under the updated rule.

Scoring is `num_correct / total_probes` across all episodes. Each episode contributes a `(num_correct, 4)` tuple.

This benchmark does not assess executive functions broadly. It tests one specific sub-skill: detecting rule-conflicting evidence in a short labeled sequence and updating predictions accordingly.

Migration note:
- Older public wording or labels are superseded. The locked public contract is `markers-v1`; only `type_a` and `type_b` are valid task outputs.

## Canonical Invariants

Before any future reframing work, these benchmark invariants are frozen and must remain stable unless the benchmark version is intentionally changed:

- Episode structure: every episode has exactly 9 ordered items, with 5 labeled examples followed by 4 unlabeled probes; template counts stay fixed as `T1 = 2/3`, `T2 = 3/2`, `T3 = 1/4` for pre-shift/post-shift labeled items.
- Rule logic: `R_std` and `R_inv` remain the same latent inverse pair used by the benchmark generator and scorer.
- Shift behavior: the governing rule changes exactly once, at `shift_after_position = pre_count`, and all later labeled items and probes are interpreted against that shift structure.
- Probe target generation: probe labels are derived deterministically from the updated post-shift sign-pattern coverage, not from prompt phrasing or free-form metadata.
- Scoring: each episode is scored only by exact probe-label matches, producing `(num_correct, 4)` with no partial-credit weighting beyond the count of correct probes.
- Seed and determinism behavior: seed controls the canonical difficulty/template/template-family/transition schedule and deterministic episode sampling; for a fixed seed, generation must stay byte-stable except for the stored split label.

## Canonical Environment

The canonical evaluation environment is Kaggle Notebooks. The runtime package is published as a Kaggle dataset (`raptorengineer/ruleshift-runtime`) and consumed by the official notebook, which produces the contract payload. Local execution is supported for development and validation but is not the canonical evaluation path.

## Architecture

### Conceptual Layers

- **`tasks/`** — the benchmark problem definition. Contains the two latent rules (`R_std`, `R_inv`), the episode schema, the deterministic episode generator, the prompt renderer, and protocol enums. This layer defines *what the benchmark is*.
- **`core/`** — the runtime infrastructure. Contains frozen split loading, the Kaggle notebook runner, the official payload builder, the manifest validator, and audit views. This layer defines *how the benchmark runs on Kaggle*.
- **Official contract** — the 7-field payload emitted by `build_kaggle_payload()` in `src/core/kaggle/payload.py`. This is the only structured output that matters for leaderboard evaluation.

### Runtime Package Contents

Files shipped in the public Kaggle dataset (`raptorengineer/ruleshift-runtime`):

**Task definition (`src/tasks/ruleshift_benchmark/`):**

- `protocol.py` — enums, constants, and public vocabulary mappings for markers, states, and outputs.
- `rules.py` — the two latent rules (`R_std`, `R_inv`).
- `schema.py` — episode data model, validation, and difficulty derivation.
- `generator.py` — deterministic episode generation from seeds.
- `render.py` — prompt rendering for the binary task format.

**Runtime infrastructure (`src/core/`):**

- `splits.py` — frozen split manifest loading and episode regeneration.
- `private_split.py` — private dataset discovery and loading at runtime (no private data is shipped).
- `kaggle/runner.py` — binary task execution and scoring.
- `kaggle/payload.py` — official 7-field payload construction and validation.
- `kaggle/manifest.py` — staging manifest validation.
- `kaggle/audit.py` — notebook-side episode inspection views (not part of the official contract).

**Frozen split:**

- `src/frozen_splits/public_leaderboard.json` — frozen public manifest (54 episodes).

### Official Contract Fields

The official payload contains exactly these fields:

- `score`
- `numerator`
- `denominator`
- `total_episodes`
- `benchmark_version`
- `split`
- `manifest_version`

The official contract does not include narrative result requirements, comparison fields, diagnostics summary fields, slice fields, or extra release-only metadata.

Within `src/core/kaggle/`, the contract path is limited to `runner.py`, `payload.py`, and `manifest.py`. `audit.py` is shipped for notebook-side episode inspection but does not contribute to the official contract payload.

### Development Tooling

Not shipped — used locally for build, test, and deploy:

- `scripts/`: build scripts, local deploy entrypoint, and shared packaging helpers.
- `tests/`: release-path validation test suite.
- `docs/`: operational checklists and design notes.
- `packaging/kaggle/`: notebook source, kernel/dataset metadata, and the frozen artifacts manifest.
- `scripts/_private_builder.py`: private episode generation from the ignored seed manifest (not included in the public package).

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
