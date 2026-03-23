# R16 Packaging Note

- Added Kaggle staging artifacts: `iron_find_electric_v1_kaggle_staging.ipynb`, `BENCHMARK_CARD.md`, `README.md`, and `frozen_artifacts_manifest.json`.
- Reproducibility is preserved by freezing explicit paths to the `R14` split manifests, carrying forward the local `R13` validity report and `R15` re-audit report, and verifying file integrity with SHA-256 hashes.
- The package currently claims a reproducible Iron Find Electric v1 Binary benchmark with Narrative as required non-leaderboard robustness evidence over the same frozen episodes and probe targets.
- The sole headline metric is Binary-only Post-shift Probe Accuracy; Narrative does not change the headline score, and only the final four labels are scored.
- The package explicitly does not claim physics skill, broad executive-function coverage, broad AGI capability, switch cost measurement, recovery length, immediate post-shift drop, online change-detection latency, or emitted `hard` slices.
- M1 live Gemini panel evidence now exists in the reports tree (Binary accuracy = 0.781250, Narrative accuracy = 0.458333, delta = 0.322917). The Kaggle staging package itself validates packaged assets and does not perform live inference.
- `hard` remains reserved and is not emitted.
