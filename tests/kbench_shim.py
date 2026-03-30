"""Minimal kaggle_benchmarks shim for local pre-deploy validation.

Provides just enough of the @kbench.task / kbench.llm / .evaluate() surface
to verify that the official notebook's task registration, function signatures,
and return shapes are compatible with the kaggle-benchmarks framework —
without requiring the real kaggle_benchmarks package or any LLM calls.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
import pandas as pd


@dataclass(frozen=True, slots=True)
class _EvalRow:
    """One scored row returned by TaskHandle.evaluate()."""
    score: tuple[int, int]
    kwargs: dict[str, Any]


class _ResultSet:
    """Mimics the result object returned by task.evaluate()."""

    def __init__(self, rows: list[_EvalRow]) -> None:
        self._rows = rows

    def as_dataframe(self) -> pd.DataFrame:
        rows = [
            {"run_id": f"Run #{idx + 1}", "result": r.score, **r.kwargs}
            for idx, r in enumerate(self._rows)
        ]
        return pd.DataFrame(rows).set_index("run_id")


class _LLMStub:
    """Stub LLM that returns None for any prompt — exercising the task's
    error-handling path and validating the (0, N) fallback shape."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def prompt(self, text: str, *, schema: Any = None) -> None:
        self.calls.append({"text": text, "schema": schema})
        return None

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def reset(self) -> None:
        self.calls.clear()


llm = _LLMStub()


@dataclass
class _TaskHandle:
    """Handle returned by @kbench.task — records registration metadata and
    provides an evaluate() method that calls the task function per row."""

    name: str
    description: str
    fn: Callable[..., Any]
    store_task: bool = True
    _registered: bool = field(default=True, init=False)
    evaluate_call_count: int = field(default=0, init=False)
    last_evaluation_data: pd.DataFrame | None = field(default=None, init=False)
    call_count: int = field(default=0, init=False)
    last_result: Any = field(default=None, init=False)

    def evaluate(
        self,
        *,
        llm: list[_LLMStub],
        evaluation_data: pd.DataFrame,
    ) -> _ResultSet:
        self.evaluate_call_count += 1
        self.last_evaluation_data = evaluation_data.copy()
        rows: list[_EvalRow] = []
        stub = llm[0]
        for _, row in evaluation_data.iterrows():
            # Build kwargs from the task function's signature (excluding 'llm')
            import inspect
            sig = inspect.signature(self.fn)
            kwargs: dict[str, Any] = {}
            for param_name in sig.parameters:
                if param_name == "llm":
                    continue
                if param_name in row.index:
                    kwargs[param_name] = row[param_name]
            score = self.fn(stub, **kwargs)
            rows.append(_EvalRow(score=score, kwargs=kwargs))
        return _ResultSet(rows)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.call_count += 1
        result = self.fn(*args, **kwargs)
        self.last_result = result
        if self.store_task:
            _write_task_artifact(self, published=False)
            _write_run_artifact(self, result=result, published=False)
        return result


# Registry of all tasks decorated with @kbench.task in this session.
_registry: dict[str, _TaskHandle] = {}


def task(
    *,
    name: str,
    description: str,
    store_task: bool = True,
) -> Callable[[Callable[..., Any]], _TaskHandle]:
    """Decorator matching the kaggle_benchmarks @kbench.task API."""

    def decorator(fn: Callable[..., Any]) -> _TaskHandle:
        handle = _TaskHandle(name=name, description=description, fn=fn, store_task=store_task)
        if store_task:
            _registry[name] = handle
            _write_task_artifact(handle, published=False)
        return handle

    return decorator


def get_registry() -> dict[str, _TaskHandle]:
    return dict(_registry)


def choose(name: str) -> Any:
    handle = _registry[name]
    result = handle(llm)
    _cleanup_published_artifacts()
    _write_task_artifact(handle, published=True)
    _write_run_artifact(handle, result=result, published=True)
    return result


def list_artifacts() -> list[Path]:
    output_dir = _artifact_dir()
    if output_dir is None:
        return []
    return sorted(output_dir.glob("*.json"))


def reset_registry() -> None:
    _registry.clear()
    llm.reset()
    output_dir = _artifact_dir()
    if output_dir is not None:
        for path in output_dir.glob("*.json"):
            path.unlink()


def _artifact_dir() -> Path | None:
    raw = os.environ.get("RULESHIFT_RUN_OUTPUT_DIR")
    if not raw:
        return None
    path = Path(raw).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cleanup_published_artifacts() -> None:
    output_dir = _artifact_dir()
    if output_dir is None:
        return
    for path in output_dir.glob("published.*.json"):
        path.unlink()


def _write_task_artifact(handle: _TaskHandle, *, published: bool) -> None:
    output_dir = _artifact_dir()
    if output_dir is None:
        return
    filename = f"{'published' if published else handle.name}.task.json"
    payload = {
        "task_name": handle.name,
        "description": handle.description,
        "store_task": handle.store_task,
        "artifact_scope": "published" if published else "intermediate",
    }
    (output_dir / filename).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_run_artifact(handle: _TaskHandle, *, result: Any, published: bool) -> None:
    output_dir = _artifact_dir()
    if output_dir is None:
        return
    filename = f"{'published' if published else handle.name}.run.json"
    payload: dict[str, Any] = {
        "task_name": handle.name,
        "artifact_scope": "published" if published else "intermediate",
    }
    if isinstance(result, dict):
        payload["run_type"] = "dataset_evaluation"
        payload["result"] = result
        payload["total_episodes"] = result.get("total_episodes")
        payload["split"] = result.get("split")
    else:
        payload["run_type"] = "single_invocation"
        payload["result"] = result
    (output_dir / filename).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
