# R16 Packaging Note

- Added Kaggle staging artifacts: `iron_find_electric_v1_kaggle_staging.ipynb`, `BENCHMARK_CARD.md`, `README.md`, and `frozen_artifacts_manifest.json`.
- Reproducibility is preserved by freezing explicit paths to the `R14` split manifests, carrying forward the local `R13` validity report and `R15` re-audit report, and verifying file integrity with SHA-256 hashes.
- The package currently claims a reproducible Iron Find Electric v1 Binary benchmark with Narrative as a robustness companion, scored by Post-shift Probe Accuracy over the frozen repaired implementation.
- The package explicitly does not claim full executive-function decomposition, online detection latency, switch cost measurement, or emitted `hard` slices.
- Remaining limitations before final submission are unchanged: the R13 anti-shortcut gate still fails on private-leaderboard subset separation, no real-model runs are bundled in-repo, and `hard` remains reserved rather than emitted.
