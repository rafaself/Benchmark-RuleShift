# Kaggle Packaging Governance

> **Status: AUTHORITATIVE KAGGLE RUNBOOK**
> This is the single authoritative operational path description for Kaggle packaging, staging, and submission in this repository.
> It does not redefine the benchmark contract; contract questions defer to [`../../KAGGLE_BENCHMARK_CONTRACT.md`](../../KAGGLE_BENCHMARK_CONTRACT.md).

This directory packages the implemented benchmark for Kaggle. It does not create the benchmark from scratch and does not redefine benchmark semantics locally.

## Official Kaggle Entry Point

- Official leaderboard notebook: `iron_find_electric_v1_kbench.ipynb`
- Official Kaggle submission path: `packaging/kaggle/kernel-metadata.json` with `"code_file": "iron_find_electric_v1_kbench.ipynb"`

No other notebook or local runtime path is an official Kaggle leaderboard submission surface.

## Official Packaged Evidence Anchor

- Official packaged readiness anchor: `reports/m1_binary_vs_narrative_robustness_report.md`
- Source report behind that anchor: the committed paired Gemini report preserving the requested model label `gemini-2.5-flash` at `reports/live/gemini-first-panel/binary-vs-narrative/history/report__20260323_120000.md`
- Supporting comparison-only material: `reports/live/gemini-first-panel/binary-vs-narrative/latest/report.md` and `reports/live/gemini-first-panel/comparison/latest/report.md`

The supporting comparison material is preserved for history and inspection. It is not a second active readiness anchor and it is not part of the Kaggle submission path.

## Included Artifacts

- `iron_find_electric_v1_kbench.ipynb`: official leaderboard notebook
- `kernel-metadata.json`: official Kaggle submission manifest pointing to the leaderboard notebook
- `iron_find_electric_v1_kaggle_staging.ipynb`: staging-only notebook for package validation and dry runs
- `BENCHMARK_CARD.md`: benchmark description and current evidence summary
- `PACKAGING_NOTE.md`: short release note for this staging bundle
- `frozen_artifacts_manifest.json`: explicit frozen paths, versions, and integrity hashes
- `KAGGLE_BENCHMARK_CONTRACT.md`: obsolete archive copy retained for Phase 2 history only

## Path Status

- `iron_find_electric_v1_kbench.ipynb`: official leaderboard notebook
- `kernel-metadata.json`: official Kaggle submission path
- `iron_find_electric_v1_kaggle_staging.ipynb`: staging-only; validates packaged artifacts and dry-run behavior
- `KAGGLE_BENCHMARK_CONTRACT.md`: archive-only; obsolete Phase 2 contract copy
- local panel runners under `src/core/*_panel.py` and repo CLI entry points: local-only runtime tools, not Kaggle submission paths

## Intended Kaggle Flow

1. Upload the repository contents needed by the official notebook, keeping `src/`, `tests/fixtures/`, `reports/`, and `packaging/kaggle/` together.
2. Submit `iron_find_electric_v1_kbench.ipynb` via `kernel-metadata.json`.
3. Optionally use `iron_find_electric_v1_kaggle_staging.ipynb` before submission to validate the frozen artifact manifest, inspect packaged resources, and run a dry run over the packaged frozen episodes.
4. Keep Binary as the only leaderboard-primary path and treat Narrative as the required same-episode robustness companion on the same episode order and probe targets.
5. Confirm that parsing, scoring, and report rendering complete end to end, with Post-shift Probe Accuracy as the headline metric.

## Reproducibility Notes

- Resource paths are explicit and relative to the repo root.
- The notebook relies on the local `src/` modules and the frozen JSON artifacts already present in the repository.
- The manifest records integrity hashes for the notebook, docs, frozen split manifests, and bundled evidence reports.
- The benchmark contract at the repository root is the source of truth for benchmark definition.
- The runtime implementation under `src/` and the frozen manifests under `src/frozen_splits/` are the source of truth for executable benchmark behavior.
- The official Kaggle submission surface is only `iron_find_electric_v1_kbench.ipynb` through `kernel-metadata.json`.
- Kaggle staging is a clean replay layer over those frozen artifacts and local evidence; it is not an independent benchmark definition.
- Current live readiness evidence remains Gemini-only and is preserved under `reports/`. Kaggle staging does not rerun or reinterpret that live evidence.

## Environment Assumptions

- The notebook only requires Python and the files bundled in this repository.
- No production dependency installation is needed for the staging notebook itself, and Kaggle staging stays independent of optional local-only provider SDKs.
- The staging notebook dry run validates packaged assets, parsing, scoring, and reporting without live external inference.
- Anthropic and OpenAI integrations exist locally in the repo, but they are outside the current v1 readiness gate and outside the Kaggle staging path.
