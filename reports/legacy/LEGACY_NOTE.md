# Legacy Artifacts

The files in this directory are archival model run outputs. They are preserved for historical reference and must not be treated as the current benchmark truth.

## Why these are legacy

- **Split version mismatch**: These artifacts were generated against `seed_bank_version` `R14-dev-3` / `R14-public-3` (16 episodes per split). The current frozen splits are `R14-dev-4` / `R14-public-4` (18 episodes per split).
- **Pre-hardening Narrative contract**: The `m1_*` report predates the Narrative format hardening. The current Narrative contract uses a 4-line structured format; the legacy report reflects the older long JSON format.

## What is here

| Path | Description |
|---|---|
| `gemini-first-panel/binary-only/latest/` | R18 Gemini 2.5 Flash binary-only run metadata |
| `gemini-first-panel/binary-vs-narrative/latest/` | R18 Gemini 2.5 Flash Lite binary+narrative run artifact and metadata |
| `gemini-first-panel/comparison/latest/` | R18 intra-family comparison metadata |
| `m1_binary_vs_narrative_robustness_report.json` | R18 Gemini 2.5 Flash Lite full run artifact (pre-hardening) |
| `m1_binary_vs_narrative_robustness_report.md` | R18 pre-hardening narrative robustness report |

## Internal path references

Some files inside this directory contain path strings that reference `reports/live/...`. Those references reflect the path at the time of generation and are now stale. The authoritative location is `reports/legacy/`.
