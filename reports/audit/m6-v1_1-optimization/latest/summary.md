# M6 v1.1 Optimization Summary

M6 is a non-breaking v1.1 cleanup pass over the current RuleShift Benchmark surface.

Completed scope:

- resynced the committed M1 Gemini paired Binary/Narrative evidence to the current report schema without running a new live provider job
- preserved the original legacy requested model label `gemini-2.5-flash` and added an explicit provenance note instead of rewriting the historical evidence claim
- kept Post-shift Probe Accuracy as the sole headline metric
- kept Binary as the only leaderboard-primary path and Narrative as the required same-episode robustness companion
- added diagnostic-only disagreement, slice, and execution-provenance sections to the committed live report surface
- split canonical live artifacts from raw provider captures by keeping raw response text under `samples/` instead of the canonical `latest/artifact.json`
- added regression coverage for live-artifact schema, report alias synchronization, and stale status text
- updated top-level repo, status, and Kaggle-facing docs to match the implemented repo state and Gemini-only readiness path

Out of scope for M6:

- no frozen split manifests changed
- no new live provider runs were performed
- no benchmark claim was broadened
- no new headline metrics were introduced
