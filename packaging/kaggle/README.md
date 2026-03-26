# Kaggle Packaging Governance

> **Status: AUTHORITATIVE KAGGLE RUNBOOK**
> This is the single authoritative operational path description for Kaggle packaging, staging, and submission in this repository.
> For benchmark description and current evidence summary, use [`BENCHMARK_CARD.md`](./BENCHMARK_CARD.md).

This directory packages the implemented benchmark for Kaggle. It does not create the benchmark from scratch and does not redefine benchmark semantics locally.

Top-level `packaging/kaggle/` is the active operational surface only. Staging-only and archive-only files live under `staging/` and `archive/`.

## Minimum Kaggle Runtime Package

The official Kaggle notebook runs with this minimum packaged subset:

- `packaging/kaggle/ruleshift_notebook_task.ipynb`
- `packaging/kaggle/kernel-metadata.json`
- `packaging/kaggle/frozen_artifacts_manifest.json` as the Kaggle runtime-contract manifest
- `src/` runtime modules, including `src/kaggle.py` and the imported `src/core/` and `src/tasks/ruleshift_benchmark/` modules
- `src/frozen_splits/dev.json`
- `src/frozen_splits/public_leaderboard.json`

Private evaluation data is loaded from a separate private-only dataset (`private_episodes.json`) and is not part of the public runtime package.
Private evaluation data must come from an authorized private dataset mount. The public repo and public runtime package do not provide a repo-local fallback.
Generate that private artifact offline before publication with `scripts/generate_private_split_artifact.py`, then attach it as the authorized private dataset mount.

The active runtime contract does not require `BENCHMARK_CARD.md`, this runbook, staging notebooks, archive files, `reports/`, or `tests/fixtures/`.

## Official Kaggle Entry Point

- Official leaderboard notebook: `ruleshift_notebook_task.ipynb`
- Official Kaggle submission path: `packaging/kaggle/kernel-metadata.json` with `"code_file": "ruleshift_notebook_task.ipynb"`

No other notebook or local runtime path is an official Kaggle leaderboard submission surface.

## Official Packaged Evidence Anchor

- Official packaged readiness anchor: `reports/m1_binary_vs_narrative_robustness_report.md`
- Source report behind that anchor: the committed paired Gemini report preserving the requested model label `gemini-2.5-flash` at `reports/live/gemini-first-panel/binary-vs-narrative/history/report__20260323_120000.md`
- Supporting comparison-only material: `reports/live/gemini-first-panel/binary-vs-narrative/latest/report.md` and `reports/live/gemini-first-panel/comparison/latest/report.md`

The supporting comparison material is preserved for history and inspection. It is not a second active readiness anchor and it is not part of the Kaggle submission path.

## Layout

Top-level active operational surface:

- `ruleshift_notebook_task.ipynb`: runtime artifact; official leaderboard notebook
- `kernel-metadata.json`: metadata; official Kaggle submission manifest pointing to the leaderboard notebook
- `frozen_artifacts_manifest.json`: metadata; Kaggle runtime-contract manifest for the official notebook and frozen split inputs
- `README.md`: active operational doc; Kaggle packaging governance and flow
- `BENCHMARK_CARD.md`: active operational doc; benchmark description and current evidence summary

Staging-only:

- `staging/ruleshift_benchmark_v1_kaggle_staging.ipynb`: optional package-validation and dry-run notebook

Archive/obsolete:

- `archive/PACKAGING_NOTE.md`: archived release note for prior packaging changes

Non-Kaggle execution surfaces:

- local panel runners under `src/core/*_panel.py` and repo CLI entry points: local-only runtime tools, not Kaggle submission paths

## Intended Kaggle Flow

1. Upload the minimum runtime package needed by the official notebook: `src/`, `src/frozen_splits/`, and `packaging/kaggle/`.
2. Submit `ruleshift_notebook_task.ipynb` via `kernel-metadata.json`.
3. Attach the authorized private evaluation dataset mount that provides `private_episodes.json` before running the official leaderboard notebook.
4. Optionally use `staging/ruleshift_benchmark_v1_kaggle_staging.ipynb` before submission to validate the frozen artifact manifest, inspect packaged resources, and run a dry run over the packaged frozen episodes.
5. Keep Binary as the only leaderboard-primary path and treat Narrative as the required same-episode robustness companion on the same episode order and probe targets.
6. Confirm that parsing, scoring, and report rendering complete end to end, with Post-shift Probe Accuracy as the headline metric.

## Public And Private Packaging Boundary

- Public deploy build: `scripts/build_deploy.py` builds only the public Kaggle notebook artifact and the public runtime dataset containing `dev.json` and `public_leaderboard.json`.
- Private artifact generation: `scripts/generate_private_split_artifact.py` generates `private_episodes.json` offline from the fixed private seed list.
- Private dataset packaging: `scripts/cd/build_private_dataset_package.py` packages only `private_episodes.json` plus private dataset metadata, outside the public repo tree.
- The public publish flow must never package, version, or upload `private_episodes.json`.

## Reproducibility Notes

- Resource paths are explicit and relative to the repo root.
- The notebook relies on the local `src/` modules and the frozen JSON artifacts already present in the repository.
- The held-out private split is not present in the public repo; it is resolved only from the authorized private dataset mount at runtime.
- The runtime-contract manifest records integrity hashes for the official notebook and frozen split manifests.
- The runtime implementation under `src/` and the frozen manifests under `src/frozen_splits/` are the source of truth for executable benchmark behavior.
- The official Kaggle submission surface is only `ruleshift_notebook_task.ipynb` through `kernel-metadata.json`.
- Kaggle staging is a clean replay layer over those frozen artifacts and local evidence; it is not an independent benchmark definition.
- Current live readiness evidence remains Gemini-only and is preserved under `reports/`. Kaggle staging does not rerun or reinterpret that live evidence.

## Environment Assumptions

- The notebook only requires Python and the files bundled in this repository.
- No production dependency installation is needed for the staging notebook itself, and Kaggle staging stays independent of optional local-only provider SDKs.
- The staging notebook dry run validates packaged assets, parsing, scoring, and reporting without live external inference.
- Anthropic and OpenAI integrations exist locally in the repo, but they are outside the current v1 readiness gate and outside the Kaggle staging path.
