# Iron Find Electric

Iron Find Electric is benchmark infrastructure for the Iron Find Electric v1 task in the Executive Functions track of the Measuring Progress Toward AGI challenge. The repository contains the implemented local benchmark environment, frozen split assets, and task-specific logic for a narrow cognitive-flexibility benchmark.

Iron Find Electric v1 is a narrow Executive Functions benchmark for cognitive flexibility. It uses electrostatics only as a controlled substrate for evaluating final post-shift rule application after sparse contradictory evidence.

A high v1 Binary score is evidence that a model correctly applied the post-shift rule to the final probes after sparse contradictory evidence in the frozen episodes. It is not evidence of physics skill, broad adaptation ability, or general reasoning ability.

## Current State

The repository currently implements:

- rules: task protocol constants, enum parsing, and rule evaluation for `R_std` and `R_inv`;
- schema: canonical episode dataclasses and derived metadata;
- generator: deterministic episode generation from seeds;
- render: Binary and Narrative prompt rendering;
- parser: prediction parsing for both task modes;
- metrics: benchmark scoring and summary helpers;
- baselines: heuristic benchmark baselines, including the recency shortcut baseline `last_evidence`;
- validation: episode, dataset, and regeneration checks;
- splits: frozen split loading, deterministic replay, and overlap checks;
- audit: split- and baseline-level audit reporting.

Current blockers and known limitations:

- the current frozen benchmark now clears the local R13 anti-shortcut validity gate and keeps the recency shortcut materially bounded in the current R15 re-audit surface;
- `hard` remains a reserved difficulty label and is not currently emitted by the R3 generator. No packaging claim depends on emitted `hard` slices;
- M1 live Gemini panel evidence now confirms model-vs-heuristic separation (Binary accuracy = 0.781250 vs best baseline = 0.546875). Model predictions are not committed as repo artifacts;
- the Kaggle staging bundle under [`packaging/kaggle/`](./packaging/kaggle/) mirrors the repaired local benchmark state, but local validation remains the source of truth.

Task and metric boundaries:

- Binary is the only leaderboard-primary task and the scored evaluation path for the v1 claim.
- Narrative is required non-leaderboard robustness evidence. It uses the same frozen episodes and probe targets as Binary, and only the final four labels are scored.
- The primary metric is Binary-only Post-shift Probe Accuracy.

Current evidence status:

- **M1 (live Gemini panel)**: Binary accuracy = 0.781250, Narrative accuracy = 0.458333 (delta = 0.322917), Binary parse-valid = 1.000000, Narrative parse-valid = 0.937500. Binary substantially exceeds all heuristic baselines. Narrative is meaningfully lower than Binary on the same frozen episodes, indicating a real surface-form robustness gap. A small Narrative provider/runtime contamination note (overall rate = 0.041667) must be disclosed separately from parse/format and adaptation outcomes.
- **M2 (staging dry-run readiness)**: Packaged frozen artifacts load, manifest validation passes, and the staging notebook runs end to end in both Binary and Narrative modes. M2 is packaging-validation evidence only, not live model-evaluation evidence.
- **M3**: Not started.

This benchmark does not claim to measure physics skill, broad AGI capability, broad executive-function coverage, switch cost, recovery length, immediate post-shift drop, or online change-detection latency.

Optional real-model execution is available locally through the Gemini first-panel runner. It is not part of the deterministic main test gate, requires the optional Gemini SDK extra, and only runs when `GEMINI_API_KEY` is configured, either in the shell environment or in a repo-root `.env` file. Benchmark runs now require pinned model IDs; the default Gemini panel model is pinned. The current M1 evidence was produced via this runner.

## Benchmark Shape

Each episode contains:

- 5 labeled items;
- 4 unlabeled probes;
- a pre-shift segment explained by `rule_A`;
- a post-shift segment explained by `rule_B`.

The benchmark allows exactly two rules over charges from `{-3, -2, -1, +1, +2, +3}`:

- `R_std`: same-sign charges repel, opposite-sign charges attract.
- `R_inv`: same-sign charges attract, opposite-sign charges repel.

Labels depend only on charge sign, not magnitude, and pair order must not affect the outcome.

## Structure

Canonical code now lives in two package areas:

- [`src/core/`](./src/core): benchmark infrastructure shared across parsing, metrics, validation, audits, and split management.
- [`src/tasks/iron_find_electric/`](./src/tasks/iron_find_electric): Iron Find Electric task logic, including protocol, schema, generation, rendering, and baselines.

Compatibility wrapper modules remain at the top level under [`src/`](./src/) so existing imports like `from generator import generate_episode` still work during the transition.

```text
src/
├── core/
│   ├── audit.py
│   ├── metrics.py
│   ├── parser.py
│   ├── splits.py
│   └── validate.py
├── frozen_splits/
├── tasks/
│   └── iron_find_electric/
│       ├── baselines.py
│       ├── generator.py
│       ├── protocol.py
│       ├── render.py
│       ├── rules.py
│       └── schema.py
└── *.py compatibility wrappers
```

## Public Interfaces

Primary task-facing interfaces:

- [`src/tasks/iron_find_electric/protocol.py`](./src/tasks/iron_find_electric/protocol.py): benchmark vocabulary, enums, and template metadata.
- [`src/tasks/iron_find_electric/schema.py`](./src/tasks/iron_find_electric/schema.py): `EpisodeItem`, `ProbeMetadata`, and `Episode`.
- [`src/tasks/iron_find_electric/generator.py`](./src/tasks/iron_find_electric/generator.py): `generate_episode`.
- [`src/tasks/iron_find_electric/render.py`](./src/tasks/iron_find_electric/render.py): Binary and Narrative renderers.
- [`src/tasks/iron_find_electric/baselines.py`](./src/tasks/iron_find_electric/baselines.py): heuristic baselines and baseline-run results.

Primary benchmark-infrastructure interfaces:

- [`src/core/parser.py`](./src/core/parser.py): `ParsedPrediction`, `ParseStatus`, and task output parsers.
- [`src/core/metrics.py`](./src/core/metrics.py): `MetricSummary` and scoring helpers.
- [`src/core/validate.py`](./src/core/validate.py): validation result dataclasses plus `validate_episode`, `validate_dataset`, and `normalize_episode_payload`.
- [`src/core/splits.py`](./src/core/splits.py): `FrozenSplitManifest`, frozen split loading, regeneration, and partition audits.
- [`src/core/audit.py`](./src/core/audit.py): audit summaries and heuristic alignment analysis.

## Frozen Splits

The repository includes frozen split JSON files under [`src/frozen_splits/`](./src/frozen_splits/). These remain the source for deterministic dev/public/private partitions.

The split utilities support:

- loading manifest-backed frozen partitions;
- regenerating episodes from stored seed banks;
- checking partition overlap;
- auditing within-partition and cross-partition distribution gaps.

## Local Setup

Create and activate a local virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-dev.txt
```

If you want to run an optional local provider benchmark panel, install the relevant provider extra as well:

```bash
python3 -m pip install -e ".[gemini]"
python3 -m pip install -e ".[anthropic]"
python3 -m pip install -e ".[openai]"
```

Run the test suite:

```bash
python3 -m pytest
```

## Convenience Commands

For local use, the repository now includes thin command wrappers over the existing
benchmark functions.

Without installing the project:

```bash
make test
make validity
make reaudit
make integrity
make evidence-pass
```

Or run the script dispatcher directly:

```bash
.venv/bin/python scripts/ife.py validity
.venv/bin/python scripts/ife.py reaudit
.venv/bin/python scripts/ife.py gemini-first-panel
.venv/bin/python scripts/evidence_pass.py
```

To run the Gemini first panel (paired Binary and Narrative), which writes the canonical latest report under `reports/live/gemini-first-panel/binary-vs-narrative/latest/report.md`:

```bash
export GEMINI_API_KEY=your_api_key_here
.venv/bin/python scripts/ife.py gemini-first-panel
```

Or place `GEMINI_API_KEY=...` in a repo-root `.env` file and run the same command.

OpenAI local-only runs follow the same paired Binary/Narrative reporting surface and require the pinned snapshot model:

```bash
export OPENAI_API_KEY=your_api_key_here
.venv/bin/python scripts/ife.py openai-panel --include-narrative --model gpt-5-mini-2025-08-07
```

OpenAI support is optional local-only execution. It is not part of the Kaggle staging path.

## Reports Layout

Generated and archived evidence under `reports/` is grouped by context and target instead of using a flat directory.

Preferred pattern for new report writers:

- `reports/<context>/<target>/latest/<stable-name>.<ext>` for the current canonical file.
- `reports/<context>/<target>/history/<stable-name>__<YYYYMMDD_HHMMSS>.<ext>` for immutable snapshots.
- `reports/<context>/<target>/samples/` for raw provider captures or other diagnostic-only artifacts.

Current examples:

- `reports/live/gemini-first-panel/binary-only/` (historical Binary-only evidence)
- `reports/live/gemini-first-panel/binary-vs-narrative/` (current paired evidence)
- `reports/m1_binary_vs_narrative_robustness_report.md` (current M1 evidence reference)
- `reports/audit/evidence-pass/`

If you want stable shell commands, install the repo in editable mode from a venv
that includes the standard packaging backend (`setuptools`):

```bash
python3 -m pip install -e .
```

Then use:

```bash
ife-test
ife-validity
ife-reaudit
ife-integrity
ife-evidence-pass
```

## Source of Truth

The source of truth for the current project state is the implemented local benchmark stack in [`src/`](./src/), the frozen assets in [`src/frozen_splits/`](./src/frozen_splits/), and the local validation, audit, and test suite results. Supporting documents should describe that implemented state honestly; they are not a substitute for the code, frozen assets, and local validity checks.

Supporting documents:

- [`iron_find_electric_implementation_spec.md`](./iron_find_electric_implementation_spec.md): behavior and contract reference for the implemented local benchmark pipeline.
- [`iron_find_electric_improved_plan.md`](./iron_find_electric_improved_plan.md): current repair roadmap and status notes.
- [`src/README.md`](./src/README.md): source-tree overview and canonical package layout.
- [`benchmark_design_section_cognitive_flexibility.md`](./benchmark_design_section_cognitive_flexibility.md): benchmark framing and explicit v1 limitations.
- [`packaging/kaggle/BENCHMARK_CARD.md`](./packaging/kaggle/BENCHMARK_CARD.md): Kaggle-facing benchmark card tied to the current repaired implementation and bundled evidence.
- [`packaging/kaggle/README.md`](./packaging/kaggle/README.md): concise Kaggle staging flow and reproducibility notes.
