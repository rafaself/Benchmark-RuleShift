# Iron Find Electric

Iron Find Electric is benchmark infrastructure for the Iron Find Electric v1 task in the Executive Functions track of the Measuring Progress Toward AGI challenge. The repository contains the implemented local benchmark environment, frozen split assets, and task-specific logic for the two-charge rule-update benchmark.

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

- the recency shortcut baseline `last_evidence` was materially reduced in the current re-audit surface, but private-leaderboard subset separation remains too weak for a clean validity pass;
- `hard` remains a reserved difficulty label and is not currently emitted by the R3 generator;
- no real-model runs are bundled in-repo, so model-vs-heuristic separation remains unverified locally;
- the Kaggle staging bundle under [`packaging/kaggle/`](./packaging/kaggle/) mirrors the repaired local benchmark state, but local validation remains the source of truth.

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

Run the test suite:

```bash
python3 -m pytest
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
