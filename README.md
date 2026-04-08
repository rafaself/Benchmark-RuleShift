# CogFlex Suite Benchmark

Kaggle-oriented benchmark project for a targeted executive-functions suite:

- faculty: `executive_functions/cognitive_flexibility`
- benchmark form: multi-turn suite evaluation
- official task name: `cogflex_suite_binary`

This repository replaces the old single RuleShift microbenchmark with a four-task cognitive-flexibility suite designed to resist simple symbolic shortcuts, increase stimulus diversity, and preserve the same Kaggle dataset and notebook publishing workflow.

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
  verify_ruleshift.py
tests/
  test_ruleshift_dataset_generation.py
  test_ruleshift_notebook_prompt_validation.py
  test_ruleshift_verification.py
Makefile
```

## Suite Shape

Each scored episode contains:

1. `learn_turn`: 4 labeled examples for the current rule
2. `shift_turn`: 4 labeled examples after a task-specific shift
3. `decision_turn`: 4 probes scored only on the final turn

Each item now mixes numeric and symbolic stimulus attributes:

- `r1`, `r2`
- `shape`
- `tone`

Published public and private rows share the same analysis schema:

- `analysis.faculty_id`
- `analysis.suite_task_id`
- `analysis.shift_mode`
- `analysis.difficulty_bin`

Public rows include `scoring.final_probe_targets`.

Private rows are inference-only. Private scoring is attached locally from `kaggle/dataset/private/private_answer_key.json`.

## Suite Tasks

- `explicit_rule_update`: turn 2 explicitly states that the rule changed
- `latent_rule_update`: turn 2 changes the rule without explicit switch language and exposes only partial conflicting evidence
- `context_binding`: turn 1 teaches `context=alpha`, turn 2 teaches `context=beta`, turn 3 mixes contexts
- `trial_cued_switch`: turn 2 introduces a cue legend that selects either the original rule or an alternate rule

The generator enforces stronger acceptance checks than the prior benchmark:

- `explicit_rule_update` and `latent_rule_update`: previous-rule accuracy must stay at or below `1/4`
- `context_binding` and `trial_cued_switch`: one-rule or cue-agnostic accuracy must stay at or below `2/4`
- `explicit_rule_update`: turn-2 evidence must include at least 3 rule-conflicting examples
- `latent_rule_update`: turn-2 evidence must include exactly 2 rule-conflicting examples
- every episode must keep the symbolic version-space majority baseline at or below `3/4`
- every split must keep symbolic-majority micro accuracy at or below `0.58`
- every suite task must keep symbolic-majority accuracy at or below `0.60`
- every split must keep the blended adversarial baseline at or below `0.65` micro and per task

## Split Design

- Public split: 80 rows, 20 per suite task
- Private split: 400 rows, 100 per suite task
- Public rows are generated only from held-in rule families: axis-threshold, sign, parity, shape-identity, and tone-state predicates
- Private rows are generated only from held-out rule families: linear, relational, absolute-value, cross-feature binding, and gated predicates
- Public/private validation checks semantic disjointness, rule-template isolation, cue-template isolation, and context-template isolation

The public split is tracked in the repository. The private split remains local-only and is expected under `kaggle/dataset/private/`.

## Local Usage

Run the repo test suite:

```bash
make test
```

Verify the tracked public split:

```bash
make verify-public
```

Verify the local private split:

```bash
make verify-private
```

`make verify-private` requires:

- `kaggle/dataset/private/private_leaderboard_rows.json`
- `kaggle/dataset/private/private_answer_key.json`
- `kaggle/dataset/private/private_split_manifest.json`
- `kaggle/dataset/private/dataset-metadata.json`

Private scoring in the notebook also requires:

```bash
RULESHIFT_PRIVATE_ANSWER_KEY_PATH=/abs/path/to/private_answer_key.json
```

## Regeneration

`scripts/build_ruleshift_dataset.py` regenerates:

- `kaggle/dataset/public/public_leaderboard_rows.json`
- `kaggle/dataset/public/dataset-metadata.json`
- `kaggle/dataset/private/private_leaderboard_rows.json`
- `kaggle/dataset/private/dataset-metadata.json`
- `kaggle/dataset/private/private_answer_key.json`

The script requires a local private manifest because private artifacts are deterministic from the maintainer seed.

## Deployment

Publish the public dataset:

```bash
make deploy-dataset
```

Publish the private dataset:

```bash
make deploy-private-dataset
```

Publish the notebook:

```bash
make deploy-notebook
```

## Kaggle Asset IDs

Public dataset:

```text
raptorengineer/cogflex-suite-runtime
```

Private dataset:

```text
raptorengineer/cogflex-suite-runtime-private
```

Notebook:

```text
raptorengineer/cogflex-suite-notebook
```

## Notes

- The notebook is the source of truth for the Kaggle runtime contract.
- The verifier checks schema, reproducibility, split isolation, label balance, difficulty bins, prompt-compatible rows, and stronger symbolic plus adversarial baselines.
- Public/private published rows intentionally omit answer-relevant latent rule identifiers.
- Difficulty bins are derived from a blended adversarial proxy baseline instead of a single symbolic solver.
- The notebook includes a final `%choose` cell so the main benchmark task is publishable through Kaggle Benchmarks.

## References

- [Kaggle Competition — Measuring Progress Toward AGI: Cognitive Abilities](https://www.kaggle.com/competitions/kaggle-measuring-agi)
- [Competition Rules](https://www.kaggle.com/competitions/kaggle-measuring-agi/rules)
- [Kaggle Benchmarks Repository](https://github.com/Kaggle/kaggle-benchmarks)
- [Kaggle Benchmarks Cookbook](https://github.com/Kaggle/kaggle-benchmarks/blob/ci/cookbook.md)
- [DeepMind Paper PDF — Measuring Progress Toward AGI: A Cognitive Framework](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/measuring-progress-toward-agi/measuring-progress-toward-agi-a-cognitive-framework.pdf)
