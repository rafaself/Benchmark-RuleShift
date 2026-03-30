#!/usr/bin/env python3
"""Fast preflight validation for the local Kaggle-like benchmark runtime."""

from __future__ import annotations

import json
from pathlib import Path
import platform
import sys


class PreflightFailure(RuntimeError):
    """Raised when a preflight stage fails."""


def _run_stage(name: str, fn):
    print(f"[preflight] {name}...")
    try:
        result = fn()
    except Exception as exc:
        raise PreflightFailure(f"{name} failed: {exc}") from exc
    print(f"[preflight] {name}: ok")
    return result


def _print_runtime_banner() -> None:
    print("=== RuleShift Kaggle Preflight ===")
    print(f"python={platform.python_version()}")
    print(f"cwd={Path.cwd()}")


def _import_runtime_modules() -> dict[str, object]:
    from core.kaggle import (
        build_kaggle_payload,
        load_leaderboard_dataframe,
        run_binary_task,
        validate_kaggle_payload,
    )
    from core.kaggle.runner import BinaryResponse, KaggleExecutionError, Label, normalize_binary_response
    from tasks.ruleshift_benchmark.protocol import PROBE_COUNT
    from tasks.ruleshift_benchmark.render import render_binary_prompt

    return {
        "BinaryResponse": BinaryResponse,
        "KaggleExecutionError": KaggleExecutionError,
        "Label": Label,
        "PROBE_COUNT": PROBE_COUNT,
        "build_kaggle_payload": build_kaggle_payload,
        "load_leaderboard_dataframe": load_leaderboard_dataframe,
        "normalize_binary_response": normalize_binary_response,
        "render_binary_prompt": render_binary_prompt,
        "run_binary_task": run_binary_task,
        "validate_kaggle_payload": validate_kaggle_payload,
    }


def _load_leaderboard(load_leaderboard_dataframe):
    private_root, frozen_splits, leaderboard_df = load_leaderboard_dataframe()
    if leaderboard_df.empty:
        raise ValueError("leaderboard dataframe is empty")
    required_columns = {"episode_id", "split", "prompt_binary", "probe_targets"}
    missing_columns = required_columns - set(leaderboard_df.columns)
    if missing_columns:
        raise ValueError(f"leaderboard dataframe is missing required columns: {sorted(missing_columns)}")
    return private_root, frozen_splits, leaderboard_df


def _validate_schema_path(BinaryResponse, Label, normalize_binary_response, probe_targets: tuple[str, ...]) -> None:
    response = BinaryResponse(*(Label(value) for value in probe_targets))
    normalized = normalize_binary_response(response)
    if normalized != probe_targets:
        raise ValueError("BinaryResponse normalization did not preserve probe targets")


def _exercise_minimal_task_path(
    *,
    BinaryResponse,
    build_kaggle_payload,
    probe_targets: tuple[str, ...],
    row: dict[str, object],
    run_binary_task,
    validate_kaggle_payload,
) -> dict[str, object]:
    import pandas as pd

    class StubLLM:
        def __init__(self, targets: tuple[str, ...]) -> None:
            self._targets = targets
            self.calls = 0

        def prompt(self, text: str, *, schema: object = None) -> dict[str, str]:
            if not text:
                raise ValueError("received empty prompt")
            if schema is not BinaryResponse:
                raise TypeError(f"expected BinaryResponse schema, got {schema!r}")
            self.calls += 1
            return {
                "probe_6": self._targets[0],
                "probe_7": self._targets[1],
                "probe_8": self._targets[2],
                "probe_9": self._targets[3],
            }

    llm = StubLLM(probe_targets)
    result = run_binary_task(
        llm=llm,
        prompt_binary=str(row["prompt_binary"]),
        probe_targets=probe_targets,
    )
    if llm.calls != 1:
        raise ValueError(f"expected exactly one stub prompt call, got {llm.calls}")

    binary_df = pd.DataFrame(
        [
            {
                "num_correct": result[0],
                "total": result[1],
                "split": row["split"],
            }
        ]
    )
    payload = build_kaggle_payload(binary_df)
    validate_kaggle_payload(payload)
    return payload


def _validate_structural_exception_surface(KaggleExecutionError, run_binary_task, row: dict[str, object]) -> None:
    class RaisingLLM:
        def prompt(self, *_args, **_kwargs):
            raise TimeoutError("preflight stub failure")

    try:
        run_binary_task(
            llm=RaisingLLM(),
            prompt_binary=str(row["prompt_binary"]),
            probe_targets=tuple(row["probe_targets"]),
        )
    except KaggleExecutionError as exc:
        if "llm.prompt failed" not in str(exc):
            raise ValueError(f"unexpected wrapped exception message: {exc}") from exc
        return
    raise ValueError("structural provider exception was not surfaced as KaggleExecutionError")


def main() -> int:
    _print_runtime_banner()

    runtime = _run_stage("import runtime modules", _import_runtime_modules)
    private_root, frozen_splits, leaderboard_df = _run_stage(
        "load leaderboard dataframe",
        lambda: _load_leaderboard(runtime["load_leaderboard_dataframe"]),
    )
    row = _run_stage(
        "select benchmark sample",
        lambda: leaderboard_df.iloc[0].to_dict(),
    )
    probe_targets = tuple(row["probe_targets"])

    _run_stage(
        "validate BinaryResponse schema path",
        lambda: _validate_schema_path(
            runtime["BinaryResponse"],
            runtime["Label"],
            runtime["normalize_binary_response"],
            probe_targets,
        ),
    )
    payload = _run_stage(
        "exercise minimal binary task path",
        lambda: _exercise_minimal_task_path(
            BinaryResponse=runtime["BinaryResponse"],
            build_kaggle_payload=runtime["build_kaggle_payload"],
            probe_targets=probe_targets,
            row=row,
            run_binary_task=runtime["run_binary_task"],
            validate_kaggle_payload=runtime["validate_kaggle_payload"],
        ),
    )
    _run_stage(
        "surface structural provider failures",
        lambda: _validate_structural_exception_surface(
            runtime["KaggleExecutionError"],
            runtime["run_binary_task"],
            row,
        ),
    )

    summary = {
        "private_dataset_root": None if private_root is None else str(private_root),
        "sample_episode_id": row["episode_id"],
        "sample_split": row["split"],
        "leaderboard_rows": len(leaderboard_df),
        "loaded_splits": sorted(frozen_splits),
        "payload": payload,
    }
    print("=== Preflight Summary ===")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("=== Preflight Passed ===")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PreflightFailure as exc:
        print(f"PRECHECK FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
