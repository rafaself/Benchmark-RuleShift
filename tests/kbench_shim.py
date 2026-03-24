"""Minimal kaggle_benchmarks shim for local pre-deploy validation.

Provides just enough of the @kbench.task / kbench.llm / .evaluate() surface
to verify that the official notebook's task registration, function signatures,
and return shapes are compatible with the kaggle-benchmarks framework —
without requiring the real kaggle_benchmarks package or any LLM calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
        return pd.DataFrame(
            [{"num_correct": r.score[0], "total": r.score[1], **r.kwargs} for r in self._rows]
        )


class _LLMStub:
    """Stub LLM that returns None for any prompt — exercising the task's
    error-handling path and validating the (0, N) fallback shape."""

    def prompt(self, text: str, *, schema: Any = None) -> None:
        return None


llm = _LLMStub()


@dataclass
class _TaskHandle:
    """Handle returned by @kbench.task — records registration metadata and
    provides an evaluate() method that calls the task function per row."""

    name: str
    description: str
    fn: Callable[..., tuple[int, int]]
    _registered: bool = field(default=True, init=False)

    def evaluate(
        self,
        *,
        llm: list[_LLMStub],
        evaluation_data: pd.DataFrame,
    ) -> _ResultSet:
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

    def __call__(self, *args: Any, **kwargs: Any) -> tuple[int, int]:
        return self.fn(*args, **kwargs)


# Registry of all tasks decorated with @kbench.task in this session.
_registry: dict[str, _TaskHandle] = {}


def task(
    *,
    name: str,
    description: str,
) -> Callable[[Callable[..., tuple[int, int]]], _TaskHandle]:
    """Decorator matching the kaggle_benchmarks @kbench.task API."""

    def decorator(fn: Callable[..., tuple[int, int]]) -> _TaskHandle:
        handle = _TaskHandle(name=name, description=description, fn=fn)
        _registry[name] = handle
        return handle

    return decorator


def get_registry() -> dict[str, _TaskHandle]:
    return dict(_registry)


def reset_registry() -> None:
    _registry.clear()
