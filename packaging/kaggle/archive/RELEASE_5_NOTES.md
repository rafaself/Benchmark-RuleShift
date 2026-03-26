# Release 5 Notes: Final Compliance Validation

## What Was Validated

Five compliance requirements were made explicitly checkable and repeatable:

1. **No private artifact in public repo** — `check_public_private_isolation.py` already covered this; no change needed.
2. **No private artifact in public package** — same script covers the deploy runtime tree; no change needed.
3. **Private split loaded through authorized flow** — static check added to `check_public_private_isolation.py`: fails if `resolve_private_dataset_root` is absent from the official notebook source.
4. **Leaderboard evaluation excludes dev** — static check added: fails if `_LEADERBOARD_PARTITIONS = ("public_leaderboard", "private_leaderboard")` is absent from the notebook; `TestNotebookEndToEnd` verifies dynamically that `leaderboard_df` contains no dev rows.
5. **Single main task in final cell** — static check added: fails if the last code cell does not contain exactly `%choose ruleshift_benchmark_v1_binary`.

## What Was Added or Changed

- `scripts/check_public_private_isolation.py`: added `_collect_notebook_compliance_errors()` with the three new static checks (requirements 3–5); `main()` includes them and updates the output label.
- `Makefile`: added `compliance-check` target (`check_public_private_isolation.py` + `pytest tests/test_kbench_notebook.py::TestNotebookEndToEnd`).
- `tests/test_packaging.py`: updated `test_kaggle_directory_layout_separates_active_staging_and_archive_files` to include `PRIVATE_SPLIT_RUNBOOK.md` (added in Release 4).
- `packaging/kaggle/PRIVATE_SPLIT_RUNBOOK.md`: added "Submission-Readiness Checklist" section referencing `make compliance-check` and the five requirements.

## What Was Not Changed

Benchmark logic, scoring, notebook semantics, artifact format, frozen splits, and all evidence artifacts are unchanged.
