# RuleShift CogFlex v2 Benchmark

Minimal Kaggle benchmark project for a targeted executive-functions task:

- faculty: `executive_functions/cognitive_flexibility`
- benchmark form: multi-turn dataset evaluation
- official task name: `ruleshift_cogflex_v2_binary`

This repository keeps the original Kaggle-oriented structure, but replaces the old single-turn RuleShift task with a three-turn cognitive-flexibility benchmark focused on switching between active classification rules.

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

## Benchmark Shape

Each scored episode contains:

1. `learn_turn`: 4 labeled examples for the initial rule
2. `shift_turn`: 4 labeled examples for the shifted rule regime
3. `decision_turn`: 4 probes scored only on the final turn

Public and private rows share the same schema:

- `inference.turns`: exactly 3 user turns
- `analysis.faculty_id`: always `executive_functions/cognitive_flexibility`
- `analysis.group_id`: one of `explicit_switch`, `reversal`, `latent_switch`, `context_switch`
- `analysis.transition_family_id`
- `analysis.initial_rule_id`
- `analysis.shift_rule_id`
- `analysis.shift_mode`

Public rows include `scoring.final_probe_targets`.

Private rows are inference-only. Private scoring is attached locally from `kaggle/dataset/private/private_answer_key.json`.

## Groups

- `explicit_switch`: turn 2 explicitly says the active rule changed
- `reversal`: the condition stays fixed but the label mapping flips
- `latent_switch`: turn 2 changes the rule without explicit switch language
- `context_switch`: turns 1 and 2 teach context-bound rules; turn 3 mixes contexts

The generator enforces adversarial probe constraints:

- `explicit_switch`, `reversal`, `latent_switch`: following the previous rule scores at most `1/4`
- `context_switch`: using one rule across all probes scores at most `2/4`

## Split Design

- Public split: 80 rows, 20 per group
- Private split: 400 rows, 100 per group
- Public/private splits use disjoint `transition_family_id` values
- Private rows are regenerated from a maintainer-only manifest seed

The public split is tracked in the repository. The private split remains local-only and is expected under `kaggle/dataset/private/`.

## Local Usage

Open the notebook locally:

```bash
make notelab
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
raptorengineer/ruleshift-cogflex-runtime-v2
```

Private dataset:

```text
raptorengineer/ruleshift-cogflex-runtime-private-v2
```

Notebook:

```text
raptorengineer/ruleshift-cogflex-notebook-v2
```

## Notes

- The notebook is the source of truth for the Kaggle runtime contract.
- The verifier checks row counts, turn counts, final probe counts, family disjointness, semantic split isolation, and baseline behavior.
- Local verification reports deterministic non-LLM baselines: oracle, invalid response, previous-rule heuristic, majority-label heuristic, and context-agnostic heuristic.
- Human-baseline collection is intentionally out of scope for this repository revision.

## References

- [Kaggle Competition — Measuring Progress Toward AGI: Cognitive Abilities](https://www.kaggle.com/competitions/kaggle-measuring-agi)
- [Competition Rules](https://www.kaggle.com/competitions/kaggle-measuring-agi/rules)
- [Kaggle Benchmarks Repository](https://github.com/Kaggle/kaggle-benchmarks)
- [Kaggle Benchmarks Cookbook](https://github.com/Kaggle/kaggle-benchmarks/blob/ci/cookbook.md)
- [DeepMind Paper PDF — Measuring Progress Toward AGI: A Cognitive Framework](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/measuring-progress-toward-agi/measuring-progress-toward-agi-a-cognitive-framework.pdf)
