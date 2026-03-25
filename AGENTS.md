# AGENTS.md

## Scope
Repo-wide defaults only. Child `AGENTS.md` files may add local deltas for their own subtrees.

## Repo anchors
- This repository is benchmark infrastructure for RuleShift Benchmark v1.
- Keep changes aligned with the implemented local benchmark.
- Local code in `src/` and frozen assets in `src/frozen_splits/` are the source of truth.
- Kaggle under `packaging/kaggle/` is packaging, not the benchmark source of truth.

## Guardrails
- Prefer narrow, targeted fixes.
- Preserve determinism, replayability, and frozen split compatibility.
- Do not change benchmark semantics, validity thresholds, split composition, or task rules unless explicitly requested.
- Update tests when behavior changes.
- Prefer TDD when logic changes or bug fixes justify it.
- Private split evaluation order: iterate exclusively on `dev` and `public_leaderboard`; run private evaluation only after code, prompt, and parameters are frozen. Never route private-only assets (`private_leaderboard.json`, `private_episodes.json`) into `deploy/kaggle-runtime/` or `packaging/kaggle/frozen_artifacts_manifest.json`.

## Verification
- Start with the smallest relevant check first.
- `make test`
- `make validity`
- `make reaudit`
- `make integrity`
- `make evidence-pass`

## Pointers
- `README.md`
- `src/README.md`