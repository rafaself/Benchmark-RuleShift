# Reports Layout

> **Status: SUPPORTING EVIDENCE INDEX**
> Report markdown captures evidence about the benchmark.
> It does not define benchmark rules or Kaggle operating procedure.

`reports/` is the implemented evidence surface for benchmark audits and live provider evidence. It is organized by context and target.

Canonical pattern:

- `reports/<context>/<target>/latest/<stable-name>.<ext>`
- `reports/<context>/<target>/latest/metadata.json` for canonical run provenance
- `reports/<context>/<target>/history/<stable-name>__<YYYYMMDD_HHMMSS>.<ext>`
- `reports/<context>/<target>/history/metadata__<YYYYMMDD_HHMMSS>.json` for immutable provenance snapshots
- `reports/<context>/<target>/samples/`

Interpretation rules:

- files under `latest/` are the current canonical evidence surfaces for that report family;
- files under `history/` are archive snapshots;
- flat top-level report aliases such as `reports/m1_binary_vs_narrative_robustness_report.md` are supporting convenience mirrors only and are not independent authority.

Current top-level groupings:

- `audit/`: deterministic benchmark audits, repairs, optimization summaries, and evidence-pass artifacts.
- `live/`: real-provider execution reports and provider-side diagnostic captures.

Current readiness interpretation:

- current benchmark-status statements live in the root `README.md`, `packaging/kaggle/BENCHMARK_CARD.md`, `packaging/kaggle/frozen_artifacts_manifest.json`, and the bundled audit fixtures rather than in any single live report;
- the active v1 live-evidence path is Gemini;
- the current paired live-report family surface is `reports/live/gemini-first-panel/binary-vs-narrative/latest/`, and the flat top-level alias `reports/m1_binary_vs_narrative_robustness_report.md` mirrors that surface;
- the earlier paired `gemini-2.5-flash` report at `reports/live/gemini-first-panel/binary-vs-narrative/history/report__20260323_120000.md` is historical provenance material;
- the direct Flash vs Flash-Lite comparison remains supplemental comparison evidence under `reports/live/gemini-first-panel/comparison/latest/`;
- Anthropic and OpenAI local report paths exist, but they are outside the current v1 readiness gate.

Use `latest/` first when you want the current file for a live-report family. Use `history/` when you need archived snapshots. Live reports are supporting evidence surfaces, not the source of truth for benchmark scope, Binary-vs-Narrative role, or current emitted difficulty labels.

Examples:

- `reports/live/gemini-first-panel/binary-only/history/report__legacy.md`
- `reports/live/gemini-first-panel/binary-vs-narrative/latest/report.md`
- `reports/live/gemini-first-panel/binary-vs-narrative/history/artifact__20260322_201900.json`
- `reports/live/gemini-first-panel/binary-vs-narrative/samples/raw_capture__20260323_120000.json`
- `reports/audit/evidence-pass/history/01_current_report.md`
