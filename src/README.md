# `src` Overview

> **Status: SUPPORTING IMPLEMENTATION MAP**
> This file describes the source tree only.
> It does not define benchmark contract terms or Kaggle operating procedure.

This directory contains the implemented benchmark code for Iron Find Electric v1.

## Canonical Layout

- `core/`: generic benchmark infrastructure used by this repository, including metrics, validation, audits, split handling, providers, and panel runners.
- `tasks/iron_find_electric/`: task-specific protocol, schema, generation, rendering, and baseline logic.
- top-level `*.py` modules: compatibility wrappers that re-export the canonical package modules.

## Core Infrastructure

- `core/parser.py`: `ParsedPrediction`, `ParseStatus`, `parse_binary_output`, and `parse_narrative_output`.
- `core/metrics.py`: `MetricSummary`, `compute_post_shift_probe_accuracy`, and scoring helpers.
- `core/validate.py`: validation result dataclasses, `validate_episode`, `validate_dataset`, and `normalize_episode_payload`.
- `core/splits.py`: `FrozenSplitManifest`, frozen split loading/generation, overlap checks, and split audits.
- `core/audit.py`: source-level audit summaries, mode comparisons, and heuristic alignment reporting.
- `core/gemini_panel.py`: active v1 readiness evidence runner for the Gemini path.
- `core/anthropic_panel.py` and `core/openai_panel.py`: local-only provider runners that exist in-repo but are outside the current v1 readiness gate.

## Task Logic

- `tasks/iron_find_electric/protocol.py`: shared task vocabulary, enums, template metadata, and parsing helpers.
- `tasks/iron_find_electric/rules.py`: charge-sign rule engine for `R_std` and `R_inv`.
- `tasks/iron_find_electric/schema.py`: canonical episode dataclasses and invariants.
- `tasks/iron_find_electric/generator.py`: deterministic seed-based episode generation.
- `tasks/iron_find_electric/render.py`: Binary and Narrative prompt renderers over the same frozen episodes.
- `tasks/iron_find_electric/baselines.py`: heuristic baselines and baseline-run helpers.

## Frozen Assets

- `frozen_splits/dev.json`: frozen development partition.
- `frozen_splits/public_leaderboard.json`: frozen `public_leaderboard` partition.
- `frozen_splits/private_leaderboard.json`: frozen `private_leaderboard` partition.

These exact split names are part of the benchmark-facing contract.

## Local Entry Points

- Official runtime implementation path: `src/` is the runtime source of truth for benchmark behavior.
- `core.cli`: thin CLI wrapper around the implemented benchmark functions.
- `scripts/ife.py`: repo-local dispatcher that works without installing the project.
- `Makefile`: shortcut targets for test, validity, re-audit, integrity, and evidence pass.

## Current Notes

- Binary is the only leaderboard-primary path.
- Narrative is the required same-episode robustness companion.
- Post-shift Probe Accuracy is the sole headline metric.
- `hard` is still part of the protocol vocabulary but is reserved and not emitted by the current generator.
- The R13 anti-shortcut validity gate passes; `last_evidence` is bounded at `0.500000` on leaderboard splits.
- Current committed readiness evidence is Gemini-only, anchored by the paired `gemini-2.5-flash` report and the canonical paired `gemini-2.5-flash-lite` run plus direct intra-Gemini comparison.
- Anthropic and OpenAI integrations already exist locally, but they are outside the current v1 readiness gate.
