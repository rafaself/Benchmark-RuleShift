# `src` Overview

This directory contains the implemented benchmark code for Iron Find Electric v1.

## Canonical Layout

- `core/`: generic benchmark infrastructure used by this repository.
- `tasks/iron_find_electric/`: task-specific protocol, schema, generation, rendering, and baseline logic.
- top-level `*.py` modules: compatibility wrappers that re-export the canonical package modules.

## Core Infrastructure

- `core/parser.py`: `ParsedPrediction`, `ParseStatus`, `parse_binary_output`, and `parse_narrative_output`.
- `core/metrics.py`: `MetricSummary`, `compute_post_shift_probe_accuracy`, and `compute_metrics`.
- `core/validate.py`: validation result dataclasses, `validate_episode`, `validate_dataset`, and `normalize_episode_payload`.
- `core/splits.py`: `FrozenSplitManifest`, frozen split loading/generation, overlap checks, and split audits.
- `core/audit.py`: source-level audit summaries, mode comparisons, and heuristic alignment reporting.

## Task Logic

- `tasks/iron_find_electric/protocol.py`: shared task vocabulary, enums, template metadata, and parsing helpers.
- `tasks/iron_find_electric/rules.py`: charge-sign rule engine for `R_std` and `R_inv`.
- `tasks/iron_find_electric/schema.py`: canonical episode dataclasses and invariants.
- `tasks/iron_find_electric/generator.py`: deterministic seed-based episode generation.
- `tasks/iron_find_electric/render.py`: Binary and Narrative prompt renderers.
- `tasks/iron_find_electric/baselines.py`: heuristic baselines and baseline-run helpers.

## Frozen Assets

- `frozen_splits/dev.json`: frozen development partition.
- `frozen_splits/public_leaderboard.json`: frozen public leaderboard partition.
- `frozen_splits/private_leaderboard.json`: frozen private leaderboard partition.

## Local Entry Points

- `core.cli`: thin CLI wrapper around the existing benchmark functions.
- `scripts/ife.py`: repo-local dispatcher that works without installing the project.
- `Makefile`: shortcut targets for test, validity, re-audit, integrity, and evidence pass.

## Current Notes

- `hard` is still part of the protocol vocabulary but is reserved and not emitted by the current generator.
- The R13 anti-shortcut validity gate now passes; `last_evidence` is bounded at 0.500000 on leaderboard splits. M1 live Gemini evidence confirms model-vs-heuristic separation.
- Compatibility wrappers are intentionally kept to avoid breaking existing imports while the canonical package paths settle.
