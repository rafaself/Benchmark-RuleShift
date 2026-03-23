# Kaggle Staging Usage

This directory packages the repaired benchmark for Kaggle staging without redefining the benchmark locally.

## Included Artifacts

- `iron_find_electric_v1_kaggle_staging.ipynb`: staging notebook entry point
- `BENCHMARK_CARD.md`: benchmark description and current evidence summary
- `PACKAGING_NOTE.md`: short release note for this staging bundle
- `frozen_artifacts_manifest.json`: explicit frozen paths, versions, and integrity hashes

## Intended Kaggle Flow

1. Upload the repository contents needed by the notebook, keeping `src/`, `tests/fixtures/`, and `packaging/kaggle/` together.
2. Open `iron_find_electric_v1_kaggle_staging.ipynb`.
3. Run the notebook cells that validate the frozen artifact manifest, inspect the packaged benchmark resources, and load the frozen split manifests.
4. Run the notebook staging dry run over the packaged frozen episodes in both Binary and Narrative modes.
5. Keep Binary as the sole leaderboard-primary path and treat Narrative as the required robustness companion on the same episode order and probe targets.
6. Confirm that parsing, scoring, and report rendering complete end to end, with Binary-only Post-shift Probe Accuracy as the headline metric.

## Reproducibility Notes

- Resource paths are explicit and relative to the repo root.
- The notebook relies on the local `src/` modules and the frozen JSON artifacts already produced before staging.
- The manifest records integrity hashes for the notebook, docs, frozen split manifests, and bundled evidence reports.
- The local validation and audit outputs remain the source of truth; Kaggle staging is a clean replay layer over those artifacts.

## Environment Assumptions

- The notebook only requires Python and the files bundled in this repository.
- No production dependency installation is needed for the staging notebook itself, and Kaggle staging must stay independent of optional local-only provider SDKs.
- The staging notebook dry run validates packaged assets, parsing, scoring, and reporting without live external inference. M1 live Gemini panel evidence exists separately in the reports tree and is not part of the staging dry-run pathway.
