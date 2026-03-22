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
3. Run the notebook cells that validate the frozen artifact manifest, inspect the benchmark card, and load the frozen split manifests.
4. Use Binary as the leaderboard-primary path and treat Narrative as the robustness companion.

## Reproducibility Notes

- Resource paths are explicit and relative to the repo root.
- The notebook relies on the local `src/` modules and the frozen JSON artifacts already produced before staging.
- The manifest records integrity hashes for the notebook, docs, frozen split manifests, and bundled evidence reports.
- The local validation and audit outputs remain the source of truth; Kaggle staging is a clean replay layer over those artifacts.

## Environment Assumptions

- The notebook only requires Python and the files bundled in this repository.
- No production dependency installation is needed for the staging notebook itself.
- The repository does not bundle provider-specific model runs. Kaggle staging here validates assets and benchmark metadata; it does not supply external model inference.
