# Core Guidance

`src/core/` is generic benchmark infrastructure. Keep task-specific behavior out of this directory unless it is truly cross-benchmark plumbing.

## Owns

- CLI orchestration
- parsing, metrics, validation
- frozen split loading and audits
- integrity checks and Kaggle manifest validation

## Boundaries

- Put RuleShift Benchmark semantics in `src/tasks/ruleshift_benchmark/`.
- If a change needs task vocabulary, rule logic, or episode-specific invariants, prefer changing the task package and only thread the minimal interface through `core/`.
- Top-level modules in `src/*.py` are compatibility wrappers; keep canonical logic in `src/core/` and `src/tasks/`.

## Working Style

- Preserve deterministic behavior and stable serialized outputs.
- Prefer additive or narrowly-scoped interface changes over broad API reshaping.
- Be careful with report text and validation/audit outputs; they should describe the current local benchmark honestly, including that `hard` is reserved and not emitted.

## Validation

- Run focused tests first, for example: `python -m pytest tests/test_cli.py tests/test_validation.py tests/test_audit.py`
- Run `make evidence-pass` before finishing changes that touch validation, audits, splits, CLI, or integrity behavior.
