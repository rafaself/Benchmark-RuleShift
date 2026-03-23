# V1 Readiness Checklist

Date assessed: 2026-03-23

This gate judges v1 readiness against the implemented repository surface only. It does not depend on speculative future infrastructure, and it does not require Anthropic or OpenAI live evidence.

## Active Gate Scope

- Active live-evidence path: Gemini only
- Canonical Gemini anchor run: `reports/live/gemini-first-panel/binary-vs-narrative/history/report__20260323_120000.md`
- Canonical Gemini Flash-Lite run: `reports/live/gemini-first-panel/binary-vs-narrative/latest/report.md`
- Canonical Flash vs Flash-Lite comparison: `reports/live/gemini-first-panel/comparison/latest/report.md`
- Source-of-truth benchmark assets: `src/`, `src/frozen_splits/`, `reports/`, `packaging/kaggle/`
- Explicitly deferred outside the active gate: Anthropic live evidence, OpenAI live evidence, cross-provider comparison, broader run-store expansion beyond the current provenance contract, human pilot, independent rerun, and protocol extensions for adaptation-lag or recovery claims

## Checklist

| Item | Status | Evidence and reason |
| --- | --- | --- |
| Frozen assets intact and versioned | PASS | `src/frozen_splits/dev.json`, `src/frozen_splits/public_leaderboard.json`, and `src/frozen_splits/private_leaderboard.json` are present with `manifest_version = R14`, `spec_version = v1`, `generator_version = R12`, `template_set_version = v1`, and `difficulty_version = R12`. `packaging/kaggle/frozen_artifacts_manifest.json` records matching hashes and versions. |
| Split manifests intact | PASS | The only split names in the implemented surface are `dev`, `public_leaderboard`, and `private_leaderboard`. The three frozen manifests exist, keep those exact benchmark-facing names, and are referenced by both Gemini metadata files and the Kaggle manifest. |
| Template family intact and unchanged across compared runs | PASS | The implemented template family remains `T1` and `T2` in `src/tasks/iron_find_electric/protocol.py`. Both Gemini paired-run metadata files and `reports/live/gemini-first-panel/comparison/latest/metadata.json` record `template_family_version = v1`, with matching frozen split hashes across Flash and Flash-Lite. |
| Binary-only Post-shift Probe Accuracy as the sole headline metric | PASS | `README.md`, `packaging/kaggle/BENCHMARK_CARD.md`, the paired Gemini reports, and the comparison report all keep Binary as the only leaderboard-primary path and Post-shift Probe Accuracy as the only headline metric. |
| Narrative present as required same-episode robustness evidence | PASS | Both canonical paired Gemini runs list `prompt_modes = [binary, narrative]` in metadata and include paired Binary/Narrative robustness tables. The comparison report also keeps Narrative as required same-episode robustness evidence, not a second headline path. |
| Canonical report layout present and functioning | PASS | The live evidence surface uses `latest/`, `history/`, and `samples/` under `reports/live/gemini-first-panel/`. `tests/test_report_outputs.py` covers the canonical latest/history/samples path behavior. |
| Canonical run provenance present | PASS WITH NOTE | Both canonical Gemini runs include `metadata.json` with release, provider, invocation, benchmark versions, and frozen split hashes. The Flash anchor preserves provenance for the requested model and frozen assets, but its legacy-resynced capture does not record served model, token usage, or duration fields. |
| Current Gemini Flash run present and canonical | PASS WITH NOTE | The current Flash anchor run is present at `reports/live/gemini-first-panel/binary-vs-narrative/history/report__20260323_120000.md` with matching metadata and artifact snapshots. It is canonical as the committed anchor referenced by `reports/live/gemini-first-panel/comparison/latest/metadata.json`, even though the paired `latest/` surface now points to Flash-Lite. |
| Current Gemini Flash-Lite run present and canonical | PASS | The current Flash-Lite paired run is the canonical `latest/` run at `reports/live/gemini-first-panel/binary-vs-narrative/latest/` with report, artifact, metadata, and sample capture. |
| Flash vs Flash-Lite comparison present and directly comparable | PASS | `reports/live/gemini-first-panel/comparison/latest/report.md` and `metadata.json` exist and explicitly verify matching schema, generator, template-family, parser, metric, difficulty, and artifact-schema versions plus shared split hashes. |
| Heuristic baselines and anti-shortcut checks documented | PASS | The canonical reports compare model Binary accuracy against `random`, `never_update`, `last_evidence`, `physics_prior`, and `template_position`. `README.md` and `packaging/kaggle/BENCHMARK_CARD.md` also preserve the R13 anti-shortcut gate and R15 re-audit context. |
| Benchmark-facing docs synchronized | PASS | `README.md`, `iron_find_electric_improved_plan.md`, `src/README.md`, `reports/README.md`, `packaging/kaggle/BENCHMARK_CARD.md`, and `packaging/kaggle/PACKAGING_NOTE.md` now describe the implemented Gemini-only gate as current Flash anchor plus current Flash-Lite canonical latest run plus direct comparison output. |
| Deferred-work boundary explicit and non-blocking | PASS | `README.md`, `iron_find_electric_improved_plan.md`, and `packaging/kaggle/BENCHMARK_CARD.md` now separate current Gemini-only readiness from deferred post-v1 empirical expansion and longer-term scientific-validity strengthening. Anthropic and OpenAI remain preserved local-only integrations outside the active readiness gate. |
| `packaging/kaggle/` path present and aligned to frozen assets | PASS | `packaging/kaggle/` contains the staging notebook, benchmark card, README, packaging note, and frozen-artifact manifest. Those surfaces explicitly point back to `src/frozen_splits/` and preserve local code and reports as source of truth. |
| Packaging manifest present and validated if supported | PASS | `packaging/kaggle/frozen_artifacts_manifest.json` exists, and the repo implements `validate_kaggle_staging_manifest()` in `src/core/kaggle.py`. `tests/test_packaging.py` exercises the validation path against current artifacts. |
| Anthropic/OpenAI explicitly marked as available local-only integrations outside the current v1 gate | PASS | `README.md`, `src/README.md`, `iron_find_electric_improved_plan.md`, `reports/README.md`, and `packaging/kaggle/BENCHMARK_CARD.md` explicitly state that Anthropic and OpenAI integrations exist locally but are outside the active Gemini-only v1 readiness gate. |

## Readiness Decision

READY

## Minimal Blockers Or Follow-Ups

None.
