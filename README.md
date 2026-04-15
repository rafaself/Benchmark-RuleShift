# CogFlex Suite Benchmark

Kaggle-oriented benchmark for rule-switching within cognitive flexibility within executive functions:

- faculty: `executive_functions/cognitive_flexibility`
- benchmark form: multi-turn suite evaluation
- official task name: `cogflex_suite_flexible`

This benchmark targets rule-switching specifically. It does not claim broad coverage of all cognitive flexibility constructs.

This repository publishes the public CogFlex contract, the deterministic public split generator, the Kaggle notebook runtime, validators for externally managed private bundles, and wrapper entrypoints for local private-bundle workflows that live under ignored `scripts/private_local/`.

## Repository Layout

```text
kaggle/
  dataset/
    public/
      dataset-metadata.json
      public_difficulty_calibration.json
      public_leaderboard_rows.json
      public_quality_report.json
    public-test/
      dataset-metadata.json
      public_difficulty_calibration.json
      public_leaderboard_rows.json
      public_quality_report.json
  notebook/
    cogflex_notebook_task.ipynb
    kernel-metadata.json
scripts/
  build_cogflex_dataset.py
  deploy_dataset.sh
  deploy_test_dataset.sh
  deploy_private_dataset.sh
  deploy_notebook.sh
  verify_cogflex.py
tests/
  cogflex_fixtures.py
  test_cogflex_dataset_generation.py
  test_cogflex_notebook_runtime.py
  test_cogflex_verification.py
Makefile
```

## Flexible Episode Contract

Each scored row exposes:

- `inference.turns`: ordered textual turns with variable length
- `inference.turn_specs`: one entry per turn with `{kind, item_count}`
- `inference.response_spec`: see below
- `analysis.faculty_id`
- `analysis.suite_task_id`
- `analysis.shift_mode`
- `analysis.difficulty_bin`
- `analysis.structure_family_id`

Public rows also include a `scoring` block with `final_probe_targets`, `probe_annotations`, and optionally `probe_metadata`. Private rows are inference-only and must be paired with an external answer key.

### `response_spec` shape

`inference.response_spec` carries the full contract for the final model response:

```json
{
  "schema_version": "cogflex.v2",
  "format": "ordered_labels",
  "probe_count": <int>,
  "label_vocab": ["<label>", ...],
  "suite_task_id": "<task>",
  "output_schema": { ... }
}
```

- `schema_version`: always `"cogflex.v2"` for the current generator
- `format`: always `"ordered_labels"`
- `probe_count`: number of probes in the decision turn
- `label_vocab`: allowed output labels for this episode
- `suite_task_id`: the suite task identifier for the episode
- `output_schema`: a strict JSON Schema object retained in the dataset contract for validation and documentation. The notebook runtime rebuilds it locally before normalizing final responses for scoring.

### `scoring` block (public split only)

- `final_probe_targets`: gold label sequence aligned to the probe order
- `probe_annotations`: per-probe congruency annotation (`"congruent"` or `"incongruent"`)
- `probe_metadata` (optional): rich per-probe dict with `target_label`, `obsolete_rule_label`, `congruency`, `requires_switch`, and optional `route_metadata`

`analysis.difficulty_bin` is an empirical model-calibration bin, not a claim about human difficulty and not a shortcut-resistance label. The public split consumes the tracked `public_difficulty_calibration.json` snapshot, and the private verifier recomputes the same bin from the fixed panel predictions shipped in the private bundle.

## Public Split Structure

The current public split (120 rows, 30 per suite task) exercises eight structural families:

| Family | Turn layout | Probes |
|---|---|---|
| `two_step_focus` | 2 evidence + decision | 5 |
| `three_step_bridge` | 3 evidence + decision | 6 |
| `wide_then_narrow` | 2 evidence + decision | 4 |
| `staggered_refresh` | 3 evidence + decision | 5 |
| `four_step_ladder` | 4 evidence + decision | 7 |
| `cue_dense_compact` | 2 evidence + decision | 5 |
| `cue_dense_balanced` | 2 evidence + decision | 6 |
| `cue_dense_wide` | 2 evidence + decision | 4 |

The first five families appear across `explicit_rule_update`, `latent_rule_update`, and `context_binding`. The cue-dense families plus `two_step_focus` are used exclusively for `trial_cued_switch`.

Public rows vary across:

- total turn count: `3`, `4`, or `5`
- decision probes: `4`, `5`, `6`, or `7`
- label vocabulary size: `2`, `3`, or `4`
- routing metadata: `context_binding` episodes attach a `context` field; `trial_cued_switch` episodes attach a `cue` field

### Public Difficulty Calibration

`kaggle/dataset/public/public_difficulty_calibration.json` records the public difficulty snapshot with:

- `version`
- `policy`: `median_split`
- `score_kind`: `mean_panel_episode_accuracy`
- `episodes`: per-episode `panel_mean_accuracy`, `difficulty_bin`, and rank

Episodes are ordered by `(panel_mean_accuracy asc, episode_id asc)`. The lower half is assigned `hard`; the upper half is assigned `medium`.

## Suite Tasks

- `explicit_rule_update`: a later evidence turn explicitly announces the replacement rule
- `latent_rule_update`: the sequence changes behavior without explicit switch language
- `context_binding`: labels depend on the context token attached to an item
- `trial_cued_switch`: labels depend on a cue that selects between competing rules

Each public suite task appears in at least two structural formats so the runtime and verifier validate the flexible contract end to end.

## Notebook Runtime

The notebook (`kaggle/notebook/cogflex_notebook_task.ipynb`) drives the benchmark evaluation. Key behaviors:

- **Default dataset**: the notebook defaults to the full 120-episode public runtime dataset (`raptorengineer/cogflex-suite-runtime`). For faster smoke tests, override it with `COGFLEX_DATASET_ROOT=/kaggle/input/datasets/raptorengineer/cogflex-suite-runtime-test` and `COGFLEX_EXPECTED_PUBLIC_EPISODE_COUNT=10`.
- **Private split wiring**: when `COGFLEX_EVAL_SPLIT=private`, the notebook reads `private_leaderboard_rows.json` and automatically resolves `private_answer_key.json` from `COGFLEX_PRIVATE_DATASET_ROOT`. Set `COGFLEX_PRIVATE_ANSWER_KEY_PATH` only if you need to override that bundled answer-key location.
- **Schema-based final prompt**: the final decision turn is constructed from `response_spec`. The notebook rebuilds the stored `output_schema` via `build_strict_output_schema` and uses the same contract during response normalization and scoring.
- **Response normalization**: the benchmark expects `ordered_labels` format. The `_normalize_response_spec` function re-derives `output_schema` from `response_spec` fields, and the runtime accepts only structured ordered-label responses.
- **Episode scoring**: `score_episode(targets, predictions, probe_metadata)` returns per-episode numerator/denominator counts and diagnostic breakdowns: `incongruent_numerator/denominator`, `congruent_numerator/denominator`, `first_probe_numerator/denominator`, `obsolete_rule_error_numerator/denominator`.
- **Suite summary**: `summarize_suite_benchmark(runs, rows, *, include_debug=False)` returns:
  - `score`: `(macro_accuracy + incongruent_accuracy + first_probe_accuracy + protocol_valid_rate * (1 - obsolete_rule_error_rate)) / 4`
  - `protocol_valid_rate`, `scorable_episodes`, `episodes`
  - `macro_accuracy`: aggregate label accuracy across all probes
  - `incongruent_accuracy`: accuracy on switch-required probes
  - `first_probe_accuracy`: accuracy on the first probe after each rule shift
  - `obsolete_rule_error_rate`: rate of predictions matching the obsolete rule instead of the active one, computed only over protocol-valid (`scorable == True`) episodes
  - `include_debug=True` adds diagnostic fields including `switch_cost`, `congruent_accuracy`, `micro_accuracy`, `structure_family_accuracy`, `per_task_metrics`, per-slice breakdowns, and raw numerator/denominator counts.

`score` is the single leaderboard-facing metric.

The full public runtime remains the source of truth for public verification and release checks. The test runtime is a deterministic 10-episode subset intended only for faster notebook iteration.

## Public Split Verification

`verify_public_split(...)` in `scripts/verify_cogflex.py` enforces:

- full schema validation against the public episode contract
- surface constraint enforcement (structural families, routing metadata)
- reproducibility: tracked rows must exactly match what the deterministic generator produces from `build_public_artifacts()`
- public difficulty calibration integrity
- identifiability: each episode must be resolvable to a unique rule by a canonical rule catalogue

Run it with:

```bash
make verify-public
```

or with an optional audit artifact:

```bash
python3 -m scripts.verify_cogflex --split public --emit-audit-report /tmp/cogflex-public-audit.json
```

## Private Bundle Contract

`scripts/verify_cogflex.py --split private` validates an external private bundle directory exposed through `--private-bundle-dir` or `COGFLEX_PRIVATE_BUNDLE_DIR`.
For local workflows, `python -m scripts.build_private_cogflex_dataset` materializes the synthetic private bundle into ignored `kaggle/dataset/private_local`.

Required files inside that directory:

- `private_leaderboard_rows.json`
- `private_answer_key.json`
- `private_calibration_predictions.json`
- `private_release_manifest.json`
- `private_quality_report.json`

Validation covers:

- inference row schema with `turn_specs` and `response_spec`
- answer-key joins by `episode_id`
- file SHA256 digests declared in the manifest
- exact, structural, and near-duplicate isolation from the public split
- generator-level isolation from the public generator reference:
  - zero overlap in hidden `generator.family_id`
  - zero overlap in hidden `generator.template_id`
  - zero overlap in hidden `generator.operator_class`
- reported calibration metrics recomputed from per-episode panel predictions
- empirical `difficulty_bin` recomputed from the fixed panel predictions
- canonical attack slices recomputed from public metadata dimensions only
- required private structure families:
  - `delayed_reversal`
  - `irrelevant_feature_interference`
  - `competitive_rule_switch`
  - `latent_rebinding`
  - `variable_evidence_budget`
  - `interleaved_context_rebinding`
- private quality report coverage:
  - `structure_family_counts`
  - `turn_count_distribution`
  - `probe_count_distribution`
  - `label_vocab_size_distribution`
  - `stimulus_space_summary`
  - `calibration_summary`
  - `attack_suite`
  - `semantic_isolation_summary`
  - `generator_isolation_summary`

The public repo does not ship private formulas or a private production generator.

### `private_answer_key.json`

Each private answer-key episode must expose:

- `episode_id`
- `faculty_id`
- `suite_task_id`
- `shift_mode`
- `difficulty_bin`
- `structure_family_id`
- `generator`: hidden generator metadata with:
  - `family_id`
  - `template_id`
  - `operator_class`
- `inference`
- `final_probe_targets`

`private_leaderboard_rows.json` remains inference-only. The hidden `generator` block lives only in the answer key and is used to enforce generator-level isolation.

### `private_calibration_predictions.json`

The private bundle must expose a panel-prediction file with this schema:

- top-level object with `version`, `split`, and `models`
- `models` must contain exactly 3 model objects
- each model object must expose:
  - `name`
  - `episodes`
- each episode entry must expose:
  - `episode_id`
  - `predicted_labels`

The verifier checks that every model covers every private episode exactly once, that each prediction list matches the episode `probe_count`, and that every predicted label belongs to that episode's `label_vocab`.

The verifier then recomputes these claims from the private bundle itself and rejects mismatches:

- `calibration_summary`
- `attack_suite`
- `semantic_isolation_summary`
- `generator_isolation_summary`

### `generator_isolation_summary`

The private quality report must expose:

- `family_ids`
- `template_ids`
- `operator_class_counts`
- `operator_diversity.distinct_operator_class_count`
- `public_non_overlap_assertion`

The verifier recomputes this summary from the answer key's hidden generator metadata and a canonical public generator reference derived from the tracked public generator. This check complements episode-level isolation instead of replacing it.

## Local Usage

Run the repo test suite:

```bash
make test
```

Rebuild the tracked public assets:

```bash
python3 -m scripts.build_cogflex_dataset
```

Build the local synthetic private bundle:

```bash
python3 -m scripts.build_private_cogflex_dataset
```

Verify the tracked public split:

```bash
make verify-public
```

Verify an external private bundle:

```bash
COGFLEX_PRIVATE_BUNDLE_DIR=/abs/path/to/private-bundle make verify-private
```

Emit a compact audit report for a clean verification run:

Use a clean git worktree if you want the recorded `git_commit` to match the verified tree state exactly. The audit artifact is success-only and intentionally excludes hidden private benchmark contents.

```bash
python3 -m scripts.verify_cogflex --split public --emit-audit-report /tmp/cogflex-public-audit.json
```

```bash
COGFLEX_PRIVATE_BUNDLE_DIR=/abs/path/to/private-bundle python3 -m scripts.verify_cogflex --split private --emit-audit-report /tmp/cogflex-private-audit.json
```

Deploy the local private bundle from `kaggle/dataset/private_local`:

```bash
make deploy-private-dataset
```
