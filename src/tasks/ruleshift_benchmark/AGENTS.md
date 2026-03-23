# RuleShift Benchmark Task Guidance

`src/tasks/ruleshift_benchmark/` owns task-specific logic for RuleShift Benchmark v1: protocol, schema, rules, generation, rendering, and baselines.

## Current State

- This task is the implemented local benchmark for RuleShift Benchmark v1.
- Keep the benchmark deterministic and replayable from seeds.
- `hard` remains part of the vocabulary but is currently reserved and not emitted.

## Boundaries

- Change task semantics here, not in `src/core/`.
- Do not silently alter rule meaning, difficulty semantics, template behavior, emitted metadata, or baseline definitions unless explicitly requested.
- When modifying generator or protocol behavior, preserve compatibility with frozen splits and local validity expectations unless the user asks for a benchmark change.

## Working Style

- Prefer narrow fixes tied to the affected module.
- Preserve benchmark semantics first; avoid “cleanup” edits that can shift outputs.
- Keep docs and report language honest about the current state: local validation is authoritative, Kaggle is downstream staging, and no claim should depend on emitted `hard` episodes.

## Validation

- Run targeted tests first, for example: `python -m pytest tests/test_generator.py tests/test_rules.py tests/test_schema.py tests/test_render.py tests/test_baselines.py`
- Run `make validity` and `make reaudit` for changes that can affect task outputs.
- Run `make evidence-pass` for changes that may alter benchmark claims, frozen compatibility, or audit surfaces.
