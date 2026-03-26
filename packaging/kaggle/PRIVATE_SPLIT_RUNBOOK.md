# Private Split Runbook

This runbook covers the only approved flow for private split generation, packaging, and public isolation.

## 1. Generate The Private Artifact Offline

Run the generator outside the public repository tree:

```bash
python scripts/generate_private_split_artifact.py \
  --benchmark-version R14 \
  --seeds-file /path/to/private_seeds.json \
  --output /external/path/private_episodes.json
```

- Keep the seed file outside the public repo.
- Keep the output outside the public repo.
- Do not change `--benchmark-version` without an explicit benchmark version bump.

## 2. Package The Private Dataset

```bash
python scripts/cd/build_private_dataset_package.py \
  --artifact /external/path/private_episodes.json \
  --output-dir /external/path/private_dataset_package \
  --dataset-id <authorized-kaggle-dataset-id>
```

- The package must contain only `private_episodes.json` and `dataset-metadata.json`.
- Upload that package as the authorized private Kaggle dataset.
- Never commit the artifact or package directory to the public repo.

## 3. Verify Public Isolation

Before any public deploy or submission:

```bash
python scripts/check_public_private_isolation.py
python scripts/build_deploy.py
python scripts/check_public_private_isolation.py
```

The public repo and `deploy/kaggle-runtime/` must not contain `private_episodes.json`, `src/frozen_splits/private_leaderboard.json`, or any repo-local private fallback.

## 4. Submission Checklist

- `python scripts/check_public_private_isolation.py` passes
- `make integrity` passes
- `make test` passes
- The private dataset is attached separately on Kaggle
- The official notebook evaluates only `public_leaderboard` and `private_leaderboard`
- The final notebook task selection remains `%choose ruleshift_benchmark_v1_binary`
