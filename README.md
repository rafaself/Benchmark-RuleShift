# RuleShift Benchmark

> **Status: SUPPORTING OVERVIEW**
> This README is a project guide, not a normative benchmark specification.
> For the frozen benchmark definition, use [`packaging/kaggle/FROZEN_BENCHMARK_SPEC.md`](./packaging/kaggle/FROZEN_BENCHMARK_SPEC.md).
> For benchmark description and current evidence summary, use [`packaging/kaggle/BENCHMARK_CARD.md`](./packaging/kaggle/BENCHMARK_CARD.md).
> For Kaggle submission and staging steps, use [`packaging/kaggle/README.md`](./packaging/kaggle/README.md).

RuleShift Benchmark is benchmark infrastructure for the implemented RuleShift Benchmark v1 task in the Executive Functions track of the Measuring Progress Toward AGI challenge. The repository already contains the local benchmark code, frozen split assets, Kaggle staging layer, reports tree, and current Gemini evidence for this narrow cognitive-flexibility benchmark.

RuleShift Benchmark v1 is a targeted Executive Functions benchmark for cognitive flexibility. It uses electrostatics only as a controlled substrate for evaluating final post-shift rule application after sparse contradictory evidence.

A high v1 Binary score is evidence that a model correctly applied the post-shift rule to the final probes after sparse contradictory evidence in the frozen episodes. It is not evidence of physics skill, broad adaptation ability, broad AGI capability, or general reasoning ability.

## Current Benchmark Status

- Scope: RuleShift Benchmark v1 is a narrow cognitive-flexibility benchmark in the Executive Functions track. Electrostatics remains only the controlled substrate.
- Leaderboard-primary path: Binary (`ruleshift_benchmark_v1_binary`) only.
- Supplemental evidence: Narrative is same-episode robustness/audit evidence only. Only the final four labels are scored, and Narrative never changes the leaderboard score.
- Current emitted difficulty labels in the shipped manifest and current audit fixtures: `easy`, `medium`, and `hard`. `reserved_difficulty_labels` is empty.
- Report status: benchmark-state statements live in this README and in [`packaging/kaggle/BENCHMARK_CARD.md`](./packaging/kaggle/BENCHMARK_CARD.md). Preserved Gemini live reports under [`reports/live/gemini-first-panel/`](./reports/live/gemini-first-panel/) are supporting evidence captures; `history/` paths are archival, and some retained live-report tables publish only `easy`/`medium` slices because they reflect older captured runs rather than current benchmark governance.

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
- audit: split- and baseline-level audit reporting;
- packaging: Kaggle packaging assets under [`packaging/kaggle/`](./packaging/kaggle/), with staging-only and archive-only material isolated under subdirectories;
- reports: current audit and live-evidence artifacts under [`reports/`](./reports/).

Current implementation notes:

- the current frozen benchmark clears the local R13 anti-shortcut validity gate and keeps the recency shortcut materially bounded in the current R15 re-audit surface;
- the current emitted difficulty labels are `easy`, `medium`, and `hard`, and `reserved_difficulty_labels` is empty in the shipped manifest surface;
- the frozen prompt/report contract includes the `template_family` axis (`canonical`, `observation_log`) and keeps invariance reporting diagnostic-only;
- the repo contains the current public paired Gemini report mirror at [`reports/m1_binary_vs_narrative_robustness_report.md`](./reports/m1_binary_vs_narrative_robustness_report.md), which currently mirrors [`reports/live/gemini-first-panel/binary-vs-narrative/latest/report.md`](./reports/live/gemini-first-panel/binary-vs-narrative/latest/report.md); the earlier paired Gemini Flash report is preserved as historical evidence at [`reports/live/gemini-first-panel/binary-vs-narrative/history/report__20260323_120000.md`](./reports/live/gemini-first-panel/binary-vs-narrative/history/report__20260323_120000.md);
- the Kaggle packaging bundle under [`packaging/kaggle/`](./packaging/kaggle/) mirrors the current local benchmark state, but it is downstream of the local benchmark and does not redefine benchmark governance.

Task and metric boundaries:

- Binary (`ruleshift_benchmark_v1_binary`) is the only leaderboard-primary path.
- Narrative is supplementary same-episode robustness evidence only. It is structured audit output over the same frozen episodes and probe targets as Binary, and only the final four labels are scored. Narrative results do not contribute to the leaderboard score.
- Post-shift Probe Accuracy is the sole headline metric.
- Aggregate accuracy remains available in the canonical payload `primary_result`.
- The frozen template-family axis is `canonical` and `observation_log`.
- Invariance reporting is diagnostic-only, reproducible when emitted, and does not change the Binary headline metric.
- Split names are exactly `dev`, `public_leaderboard`, and `private_leaderboard`. `dev` is local-only and is never included in the official leaderboard evaluation. `private_leaderboard` is held out and loaded only from an authorized private dataset mount.

Current v1 readiness status:

- the active v1 readiness evidence path is Gemini;
- the current public paired Gemini report surface is [`reports/m1_binary_vs_narrative_robustness_report.md`](./reports/m1_binary_vs_narrative_robustness_report.md), a convenience mirror of [`reports/live/gemini-first-panel/binary-vs-narrative/latest/report.md`](./reports/live/gemini-first-panel/binary-vs-narrative/latest/report.md), which currently contains the committed paired `gemini-2.5-flash-lite` run;
- the earlier paired `gemini-2.5-flash` report at [`reports/live/gemini-first-panel/binary-vs-narrative/history/report__20260323_120000.md`](./reports/live/gemini-first-panel/binary-vs-narrative/history/report__20260323_120000.md) is retained as historical live evidence and provenance material;
- the direct Flash vs Flash-Lite comparison remains supplemental comparison material under [`reports/live/gemini-first-panel/comparison/latest/report.md`](./reports/live/gemini-first-panel/comparison/latest/report.md), not an official Kaggle submission path;
- Anthropic and OpenAI integrations already exist locally, but they are outside the current v1 readiness gate and are not required for current v1 readiness;
- current v1 readiness does not require cross-provider evidence.

Deferred-work boundary:

- current v1 readiness remains the Gemini-only gate above and is not blocked by deferred empirical or scientific-validity work;
- Anthropic and OpenAI integrations are preserved local-only assets for later empirical expansion, not dead ends and not part of the active readiness gate;
- post-v1 empirical expansion is deferred: Anthropic live evidence, OpenAI live evidence, cross-provider comparison, and broader run-store expansion beyond the current provenance contract;
- longer-term scientific-validity strengthening is deferred: human pilot, independent rerun, and protocol extensions needed for adaptation-lag or recovery claims.

Current evidence status:

- **M1 (historical paired Gemini Flash run)**: Binary accuracy = 0.781250, Narrative accuracy = 0.458333 (delta = 0.322917), Binary parse-valid = 1.000000, Narrative parse-valid = 0.937500. Binary substantially exceeds all heuristic baselines. Narrative is meaningfully lower than Binary on the same frozen episodes, indicating a real surface-form robustness gap. A small Narrative provider/runtime contamination note (overall rate = 0.041667) must be disclosed separately from parse/format and adaptation outcomes. The committed historical report preserves the original requested model label `gemini-2.5-flash`.
- **M1b (current public paired Gemini Flash-Lite report surface)**: Binary accuracy = 0.687500, Narrative accuracy = 0.276042 (delta = 0.411458), Binary parse-valid = 0.958333, Narrative parse-valid = 0.520833. The current public paired-report mirror lives at [`reports/m1_binary_vs_narrative_robustness_report.md`](./reports/m1_binary_vs_narrative_robustness_report.md) and mirrors [`reports/live/gemini-first-panel/binary-vs-narrative/latest/report.md`](./reports/live/gemini-first-panel/binary-vs-narrative/latest/report.md).
- **M1c (intra-Gemini comparison)**: The Flash vs Flash-Lite comparison confirms matching benchmark versions and frozen split hashes, so the current Gemini anchor run and Flash-Lite run are directly comparable without widening the claim beyond the Gemini readiness path.
- **M2 (staging dry-run readiness)**: Packaged frozen artifacts load, manifest validation passes, and the staging notebook runs end to end in both Binary and Narrative modes. M2 is packaging-validation evidence only, not live model-evaluation evidence.
- **Local provider surfaces**: Gemini, Anthropic, and OpenAI runners exist locally. Only Gemini is part of the current v1 readiness path.
- **M6**: The v1.1 optimization pass tightened diagnostic reporting, live-artifact discipline, and release hygiene without changing benchmark behavior, frozen artifacts, or the headline metric.

This benchmark does not claim to measure physics skill, broad AGI capability, broad executive-function coverage, switch cost, recovery length, immediate post-shift drop, or online change-detection latency.

Optional real-model execution is available locally through the provider panel runners. It is not part of the deterministic main test gate, requires the matching optional provider SDK extra, and only runs when the relevant API key is configured in the shell environment or in a repo-root `.env` file. Benchmark runs require pinned model IDs. The current committed readiness evidence is Gemini-only.

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

Canonical code lives in two package areas:

- [`src/core/`](./src/core): benchmark infrastructure shared across parsing, metrics, validation, audits, providers, panel runners, and split management.
- [`src/tasks/ruleshift_benchmark/`](./src/tasks/ruleshift_benchmark): RuleShift Benchmark task logic, including protocol, schema, generation, rendering, and baselines.

Compatibility wrapper modules remain at the top level under [`src/`](./src/) so existing imports like `from generator import generate_episode` still work during the transition.

```text
src/
├── core/
│   ├── audit.py
│   ├── metrics.py
│   ├── panel_runner.py
│   ├── parser.py
│   ├── providers/
│   ├── report_outputs.py
│   ├── splits.py
│   └── validate.py
├── frozen_splits/
├── tasks/
│   └── ruleshift_benchmark/
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

- [`src/tasks/ruleshift_benchmark/protocol.py`](./src/tasks/ruleshift_benchmark/protocol.py): benchmark vocabulary, enums, and template metadata.
- [`src/tasks/ruleshift_benchmark/schema.py`](./src/tasks/ruleshift_benchmark/schema.py): `EpisodeItem`, `ProbeMetadata`, and `Episode`.
- [`src/tasks/ruleshift_benchmark/generator.py`](./src/tasks/ruleshift_benchmark/generator.py): `generate_episode`.
- [`src/tasks/ruleshift_benchmark/render.py`](./src/tasks/ruleshift_benchmark/render.py): Binary and Narrative renderers.
- [`src/tasks/ruleshift_benchmark/baselines.py`](./src/tasks/ruleshift_benchmark/baselines.py): heuristic baselines and baseline-run results.

Primary benchmark-infrastructure interfaces:

- [`src/core/parser.py`](./src/core/parser.py): `ParsedPrediction`, `ParseStatus`, and task output parsers.
- [`src/core/metrics.py`](./src/core/metrics.py): `MetricSummary` and scoring helpers.
- [`src/core/validate.py`](./src/core/validate.py): validation result dataclasses plus `validate_episode`, `validate_dataset`, and `normalize_episode_payload`.
- [`src/core/splits.py`](./src/core/splits.py): `FrozenSplitManifest`, frozen split loading, regeneration, and partition audits.
- [`src/core/audit.py`](./src/core/audit.py): audit summaries and heuristic alignment analysis.
- [`src/core/gemini_panel.py`](./src/core/gemini_panel.py), [`src/core/anthropic_panel.py`](./src/core/anthropic_panel.py), and [`src/core/openai_panel.py`](./src/core/openai_panel.py): local provider panel runners.

## Frozen Splits

The repository includes frozen split JSON files under [`src/frozen_splits/`](./src/frozen_splits/) for the public runtime partitions `dev` and `public_leaderboard`. The held-out `private_leaderboard` partition is resolved only through the private split loader and an authorized private dataset mount.

The split utilities support:

- loading manifest-backed frozen partitions;
- regenerating the public partitions from stored seed banks;
- generating the held-out private artifact offline for a fixed benchmark version;
- checking partition overlap;
- auditing within-partition and cross-partition distribution gaps.

The held-out private artifact is generated offline with `scripts/generate_private_split_artifact.py` and then mounted into the private runtime as `private_episodes.json`. Runtime code does not regenerate `private_leaderboard`.

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

For local use, the repository includes thin command wrappers over the implemented benchmark functions.

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
.venv/bin/python scripts/ruleshift_benchmark.py validity
.venv/bin/python scripts/ruleshift_benchmark.py reaudit
.venv/bin/python scripts/ruleshift_benchmark.py gemini-first-panel
.venv/bin/python scripts/evidence_pass.py
```

To run the current Gemini readiness path (paired Binary and Narrative), which writes the canonical latest report under `reports/live/gemini-first-panel/binary-vs-narrative/latest/report.md`:

```bash
export GEMINI_API_KEY=your_api_key_here
.venv/bin/python scripts/ruleshift_benchmark.py gemini-first-panel
```

Or place the local provider keys you need in a repo-root `.env` file:

```dotenv
GEMINI_API_KEY=your_gemini_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

Then run the matching local panel command.

OpenAI local-only runs follow the same paired Binary/Narrative reporting surface and require the pinned snapshot model:

```bash
export OPENAI_API_KEY=your_api_key_here
.venv/bin/python scripts/ruleshift_benchmark.py openai-panel --include-narrative --model gpt-5-mini-2025-08-07
```

Anthropic and OpenAI support are optional local-only execution surfaces. They are not part of the current v1 readiness gate.

## Reports Layout

Generated and archived evidence under `reports/` is grouped by context and target instead of using a flat directory.

Preferred pattern for new report writers:

- `reports/<context>/<target>/latest/<stable-name>.<ext>` for the current canonical file.
- `reports/<context>/<target>/history/<stable-name>__<YYYYMMDD_HHMMSS>.<ext>` for immutable snapshots.
- `reports/<context>/<target>/samples/` for raw provider captures or other diagnostic-only artifacts.

Current examples:

- `reports/live/gemini-first-panel/binary-only/` (historical Binary-only evidence)
- `reports/live/gemini-first-panel/binary-vs-narrative/` (current paired Gemini evidence history and supporting `latest/` material)
- `reports/m1_binary_vs_narrative_robustness_report.md` (single packaged readiness anchor, synced to the committed paired M1 report)
- `reports/audit/evidence-pass/`

If you want stable shell commands, install the repo in editable mode from a venv that includes the standard packaging backend (`setuptools`):

```bash
python3 -m pip install -e .
```

Then use:

```bash
ruleshift-benchmark-test
ruleshift-benchmark-validity
ruleshift-benchmark-reaudit
ruleshift-benchmark-integrity
ruleshift-benchmark-evidence-pass
```

## Source of Truth

The repository governance model is:

- frozen benchmark specification: [`packaging/kaggle/FROZEN_BENCHMARK_SPEC.md`](./packaging/kaggle/FROZEN_BENCHMARK_SPEC.md) is the single normative benchmark-definition document for the current cognitive-flexibility benchmark;
- benchmark definition: [`src/`](./src/) and the frozen manifests under [`src/frozen_splits/`](./src/frozen_splits/) are the single executable source of truth for benchmark behavior;
- benchmark card: [`packaging/kaggle/BENCHMARK_CARD.md`](./packaging/kaggle/BENCHMARK_CARD.md) is the descriptive benchmark summary and current evidence record;
- Kaggle runbook: [`packaging/kaggle/README.md`](./packaging/kaggle/README.md) is the single authoritative operational path description for Kaggle packaging, staging, and submission;
- Kaggle leaderboard entry point: [`packaging/kaggle/ruleshift_notebook_task.ipynb`](./packaging/kaggle/ruleshift_notebook_task.ipynb) is the single official Kaggle leaderboard notebook, wired by [`packaging/kaggle/kernel-metadata.json`](./packaging/kaggle/kernel-metadata.json);
- minimum Kaggle runtime package: the official notebook, [`packaging/kaggle/kernel-metadata.json`](./packaging/kaggle/kernel-metadata.json), [`packaging/kaggle/frozen_artifacts_manifest.json`](./packaging/kaggle/frozen_artifacts_manifest.json) as the Kaggle runtime-contract manifest, [`src/`](./src/), and the public frozen manifests under [`src/frozen_splits/`](./src/frozen_splits/);
- Kaggle staging materials: the rest of [`packaging/kaggle/`](./packaging/kaggle/) is packaging support only and does not redefine benchmark semantics or runtime behavior;
- evidence and audits: [`reports/`](./reports/) and the bundled validation/audit fixtures record evidence about the implemented benchmark.

Supporting documents describe that implemented state; they do not override the code, frozen assets, or local validity checks.

Key documents:

- [`packaging/kaggle/FROZEN_BENCHMARK_SPEC.md`](./packaging/kaggle/FROZEN_BENCHMARK_SPEC.md): frozen benchmark methodology and benchmark-facing scope for the current cognitive-flexibility benchmark.
- [`src/README.md`](./src/README.md): source-tree overview and canonical package layout.
- [`packaging/kaggle/BENCHMARK_CARD.md`](./packaging/kaggle/BENCHMARK_CARD.md): Kaggle-facing benchmark card tied to the current implemented benchmark and bundled evidence.
- [`packaging/kaggle/README.md`](./packaging/kaggle/README.md): Kaggle packaging governance, official leaderboard notebook, and staging-only path labels.
