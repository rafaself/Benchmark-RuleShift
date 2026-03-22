# Repository Guidance

This repo is benchmark infrastructure for Iron Find Electric v1. Keep instructions and changes aligned with the implemented local benchmark, not aspirational designs.

## Source of Truth

- Local code in `src/`, frozen assets in `src/frozen_splits/`, and local validation/audit results are authoritative.
- Kaggle under `packaging/kaggle/` is a staging and packaging layer after local validity is established. Do not treat Kaggle artifacts as the benchmark source of truth.
- `hard` is reserved in the protocol and reports, but the current generator does not emit it.

## Main Commands

- `make test`
- `make validity`
- `make reaudit`
- `make integrity`
- `make evidence-pass`

Direct entry point if needed: `.venv/bin/python scripts/ife.py <command>`.

## Change Rules

- Prefer narrow fixes. Avoid repo-wide refactors unless explicitly requested.
- Preserve determinism, replayability, and frozen split compatibility.
- Do not change benchmark semantics, validity thresholds, split composition, or task rules unless the user explicitly asks for that.
- Update tests for behavior changes. Prefer the smallest relevant pytest target first, then run `make evidence-pass` when the change could affect benchmark validity claims.
- Keep `reports/` organized by context and target, not as a flat dump.
- Use this storage pattern for newly introduced report writers:
  `reports/<context>/<target>/latest/<stable-name>.<ext>`
  `reports/<context>/<target>/history/<stable-name>__<YYYYMMDD_HHMMSS>.<ext>`
- Use `latest/` only for the current canonical file that downstream docs or commands should point to.
- Use `history/` for immutable snapshots that preserve prior evidence for later comparison.
- Group raw provider samples or one-off diagnostic captures under a contextual `samples/` directory instead of mixing them with canonical reports.
- When reorganizing prior reports, prefer moving them into the appropriate contextual `history/` or `samples/` directory rather than leaving duplicated flat copies behind.
- Do not replace the only copy of a past run if a comparison against future runs may be needed.

## Pointers

- `README.md`: current project state and command surface.
- `src/README.md`: canonical code layout.
- `iron_find_electric_implementation_spec.md`: task and benchmark contract.
