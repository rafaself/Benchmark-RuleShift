# CogFlex Suite Benchmark

CogFlex is a Kaggle benchmark for rule-switching within `executive_functions/cognitive_flexibility`.
The published benchmark task name is `cogflex_suite_flexible`.

Scope note: this repository implements a held-out benchmark task for model evaluation.
It does not include human baseline collection or human-relative score mapping.

This repository contains:

- the deterministic public dataset generator
- the Kaggle notebook runtime
- public and private verification scripts
- deployment helpers for datasets and notebook publishing
- local tooling for a synthetic private bundle

## Repository

```text
kaggle/
  dataset/
    public/
    public-test/
  notebook/
scripts/
tests/
Makefile
```

## Episode Contract

Each row contains:

- `episode_id`
- `analysis`: `faculty_id`, `suite_task_id`, `shift_mode`, `difficulty_bin`, `structure_family_id`
- `inference.turns`: ordered rendered turns
- `inference.turn_specs`: one `{kind, item_count}` entry per turn
- `inference.response_spec`

Public rows also include `scoring` with:

- `final_probe_targets`
- `probe_annotations`
- required `probe_metadata`

`probe_metadata` carries these benchmark-facing fields:

- `target_label`
- `obsolete_rule_label`
- `congruency`
- `requires_switch`
- `diagnostic_role`
- `shift_window_rank`

The decision turn still has one final scored response. Within that response, the first `1` or `2` probes are reserved as an early shift-diagnostic window:

- `2` probes when `probe_count >= 4`
- otherwise `1` probe

Those leading diagnostic probes are generated to require a rule switch and to disagree with the obsolete or default rule.

Private leaderboard rows are inference-only and must be paired with a separate answer key.
That answer key is also expected to include `probe_metadata` for every episode.

`response_spec` uses this shape:

```json
{
  "schema_version": "cogflex.v2",
  "format": "ordered_labels",
  "probe_count": 5,
  "label_vocab": ["accept", "reject"],
  "suite_task_id": "explicit_rule_update",
  "output_schema": { "...": "..." }
}
```

Runtime note: the notebook scoring path only uses `format`, `probe_count`, `label_vocab`, and `suite_task_id`.
`schema_version` and `output_schema` remain part of the dataset contract, but the notebook ignores them during response normalization and scoring.

`analysis.difficulty_bin` is an empirical model-panel calibration bin derived from the tracked difficulty calibration files. It is not a human difficulty claim or a human-relative percentile.

## Shipped Artifacts

- `kaggle/dataset/public`: 120 public rows, 30 per suite task
- `kaggle/dataset/public-test`: deterministic 10-row public subset
- `kaggle/dataset/public/public_difficulty_calibration.json`: tracked public difficulty calibration snapshot
- `kaggle/notebook/cogflex_notebook_task.ipynb`: Kaggle runtime notebook

This repository is public. The private benchmark rows and private scoring artifacts must not be committed here.
During development, the generator writes local private release surfaces into gitignored directories:

- `kaggle/dataset/private`: local private leaderboard rows working surface
- `kaggle/dataset/private-scoring`: local private scoring working surface

Those local surfaces can be deployed to private Kaggle datasets, but they should be versioned in a separate private repository rather than this public one.

Suite tasks:

- `explicit_rule_update`: a later evidence turn explicitly announces the replacement rule
- `latent_rule_update`: the evidence changes without explicit switch language
- `context_binding`: labels depend on a context token attached to each item
- `trial_cued_switch`: labels depend on a cue that selects between competing rules

## Notebook Runtime

The notebook at `kaggle/notebook/cogflex_notebook_task.ipynb` currently:

- hard-codes `EVAL_SPLIT = "private"`
- uses `/kaggle/input/datasets/raptorengineer/cogflex-suite-runtime` as the public dataset root
- uses `/kaggle/input/datasets/raptorengineer/cogflex-suite-runtime-private` as the private rows root
- uses `/kaggle/input/datasets/raptorengineer/cogflex-suite-runtime-private-scoring` as the private scoring root
- loads `private_leaderboard_rows.json` from the private rows root and `private_answer_key.json` from the private scoring root when the split is `private`
- appends a JSON-only ordered-label instruction to the final decision turn
- keeps the single final decision turn contract while reserving an early shift-diagnostic probe window inside that probe set
- accepts ordered-label responses from JSON strings, lists or tuples, dicts containing `ordered_labels`, dataclasses exposing `ordered_labels`, or objects exposing `ordered_labels`
- treats malformed responses, wrong label counts, and out-of-vocabulary labels as protocol failures
- rejects public rows or private answer-key episodes that omit `probe_metadata` or violate the shift-diagnostic metadata contract

The registered notebook task is a held-out model evaluation of rule switching / rule induction within cognitive flexibility. Its summary is model-relative and dataset-relative; no human baselines or human-relative normalization are included.

The leaderboard-facing score is:

```text
score = average(
  macro_accuracy,
  incongruent_accuracy,
  first_probe_accuracy,
  protocol_valid_rate * (1 - obsolete_rule_error_rate)
)
```

`summarize_suite_benchmark(..., include_debug=True)` also emits additional diagnostics such as `micro_accuracy`, `congruent_accuracy`, `switch_cost`, `per_task_metrics`, `structure_family_accuracy`, and raw numerator and denominator counts.
The debug summary also includes `shift_window_accuracy`, `shift_window_numerator`, and `shift_window_denominator` so early post-shift recovery can be read separately from the stricter `first_probe_accuracy` signal.
The debug summary also marks the task scope explicitly as held-out-only and declares that human baseline and human-relative mapping are absent.

## Verification

Public verification checks:

- row schema and public surface constraints
- exact reproducibility against `build_public_artifacts()`
- difficulty calibration consistency
- quality report consistency
- identifiability

Run it with:

```bash
make verify-public
```

or:

```bash
python3 -m scripts.verify_cogflex --split public --emit-audit-report /tmp/cogflex-public-audit.json
```

Private release verification checks:

- required files are present across the split local private release surfaces:
  `kaggle/dataset/private/private_leaderboard_rows.json`
  `kaggle/dataset/private-scoring/private_answer_key.json`
  `kaggle/dataset/private-scoring/private_calibration_predictions.json`
  `kaggle/dataset/private-scoring/private_release_manifest.json`
  `kaggle/dataset/private-scoring/private_quality_report.json`
- answer-key consistency
- manifest digests
- calibration prediction consistency and empirical difficulty bins
- quality report schema and reproducibility
- quality report `generator_isolation_summary` consistency
- generator metadata `operator_class` coverage and non-overlap checks
- isolation from the public split and public generator reference
- required private structure family coverage
- identifiability

Run it with:

```bash
make verify-private
```

or:

```bash
python3 -m scripts.verify_cogflex --split private \
  --private-rows-dir /abs/path/to/private \
  --private-scoring-dir /abs/path/to/private-scoring \
  --emit-audit-report /tmp/cogflex-private-audit.json
```

## Local Usage

Run tests:

```bash
make test
```

Rebuild tracked public artifacts:

```bash
python3 -m scripts.build_cogflex_dataset
```

Rebuild the local gitignored split private release surfaces:

```bash
python3 -m scripts.build_private_cogflex_dataset
```

Deploy artifacts:

```bash
make deploy-dataset
make deploy-test-dataset
make deploy-private-dataset
make deploy-notebook
```

`make deploy-private-dataset` publishes both private Kaggle datasets from the local split private release surfaces and refuses to run unless `./scripts/release_check.sh` has already passed for the current repository state.
