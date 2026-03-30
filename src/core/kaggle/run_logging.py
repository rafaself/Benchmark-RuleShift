from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from core.kaggle.run_context import (
    RUN_ID_ENV_VAR,
    RUN_OUTPUT_DIR_ENV_VAR,
    BenchmarkRunContext,
    build_run_context,
)
from core.kaggle.run_log_io import (
    DurableFileHandler,
    ExceptionSummary,
    JsonLinesFormatter,
    summarize_exception_log,
    timestamp_utc,
)

__all__ = [
    "BENCHMARK_LOG_FILENAME",
    "EXCEPTIONS_LOG_FILENAME",
    "LIFECYCLE_EVENTS",
    "RUN_ID_ENV_VAR",
    "RUN_OUTPUT_DIR_ENV_VAR",
    "BenchmarkRunContext",
    "BenchmarkRunLogger",
    "ExceptionSummary",
    "build_run_context",
]

BENCHMARK_LOG_FILENAME = "benchmark_log.jsonl"
EXCEPTIONS_LOG_FILENAME = "exceptions.jsonl"
LIFECYCLE_EVENTS = frozenset(
    {
        "run_started",
        "bootstrap_started",
        "bootstrap_finished",
        "phase_started",
        "episode_started",
        "provider_call_started",
        "provider_call_succeeded",
        "provider_call_failed",
        "response_parsed",
        "response_parse_failed",
        "episode_scored",
        "phase_finished",
        "payload_built",
        "run_finished",
        "run_invalidated",
    }
)

_LOGGER_BASE_NAME = "ruleshift"


class _ContextAdapter(logging.LoggerAdapter):
    def bind(self, **extra: Any) -> "_ContextAdapter":
        return _ContextAdapter(self.logger, {**self.extra, **extra})

    def process(self, msg: Any, kwargs: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        payload = dict(self.extra)
        payload.update(kwargs.pop("payload", {}))
        payload.update(kwargs.pop("extra", {}))
        kwargs["extra"] = {"payload": payload}
        return msg, kwargs


def _logger_name(*, channel: str, path: Path) -> str:
    return f"{_LOGGER_BASE_NAME}.{channel}.{path.resolve()}"


def _configure_jsonl_logger(
    *,
    channel: str,
    path: Path,
    include_traceback: bool = False,
) -> logging.Logger:
    logger = logging.getLogger(_logger_name(channel=channel, path=path))
    logger.setLevel(logging.INFO)
    logger.propagate = False

    resolved_path = path.resolve()
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename).resolve() == resolved_path:
            return logger

    handler = DurableFileHandler(resolved_path, mode="a", encoding="utf-8", delay=True)
    handler.setLevel(logging.INFO)
    handler.setFormatter(JsonLinesFormatter(include_traceback=include_traceback))
    logger.addHandler(handler)
    return logger


def _levelno(level: str) -> int:
    levelno = logging.getLevelName(level.upper())
    if isinstance(levelno, int):
        return levelno
    raise ValueError(f"unsupported log level: {level}")


class BenchmarkRunLogger:
    def __init__(self, context: BenchmarkRunContext) -> None:
        self.context = context
        self.context.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.context.output_dir / BENCHMARK_LOG_FILENAME
        self.exceptions_path = self.context.output_dir / EXCEPTIONS_LOG_FILENAME

        base_context = {
            "run_id": self.context.run_id,
            "provider": self.context.provider,
            "model": self.context.model,
        }
        self._benchmark_adapter = _ContextAdapter(
            _configure_jsonl_logger(channel="benchmark", path=self.log_path),
            base_context,
        )
        self._exceptions_adapter = _ContextAdapter(
            _configure_jsonl_logger(
                channel="exceptions",
                path=self.exceptions_path,
                include_traceback=True,
            ),
            base_context,
        )

    def _adapter(
        self,
        *,
        phase: str,
        task_mode: str,
        episode_id: str | None,
        status: str,
        exceptions: bool = False,
    ) -> _ContextAdapter:
        adapter = self._exceptions_adapter if exceptions else self._benchmark_adapter
        return adapter.bind(
            phase=phase,
            episode_id=episode_id,
            task_mode=task_mode,
            status=status,
        )

    def _emit(
        self,
        adapter: _ContextAdapter,
        *,
        event: str,
        level: str,
        exc_info: Any = None,
        **extra_fields: Any,
    ) -> dict[str, Any]:
        timestamp = timestamp_utc()
        payload = {
            **adapter.extra,
            "timestamp": timestamp,
            "event": event,
            "level": level,
            **extra_fields,
        }
        adapter.log(
            _levelno(level),
            event,
            payload={
                "timestamp": timestamp,
                "event": event,
                "level": level,
                **extra_fields,
            },
            exc_info=exc_info,
        )
        return payload

    def log(
        self,
        *,
        phase: str,
        event: str,
        level: str,
        status: str,
        task_mode: str,
        episode_id: str | None,
        **extra_fields: Any,
    ) -> dict[str, Any]:
        return self._emit(
            self._adapter(
                phase=phase,
                task_mode=task_mode,
                episode_id=episode_id,
                status=status,
            ),
            event=event,
            level=level,
            **extra_fields,
        )

    def log_lifecycle(
        self,
        *,
        phase: str,
        event: str,
        status: str,
        task_mode: str,
        episode_id: str | None,
        level: str = "info",
        **extra_fields: Any,
    ) -> dict[str, Any]:
        if event not in LIFECYCLE_EVENTS:
            raise ValueError(f"unsupported lifecycle event: {event}")
        return self.log(
            phase=phase,
            event=event,
            level=level,
            status=status,
            task_mode=task_mode,
            episode_id=episode_id,
            **extra_fields,
        )

    def log_run_started(self, **extra_fields: Any) -> dict[str, Any]:
        return self.log_lifecycle(phase="run", event="run_started", status="started", task_mode="notebook", episode_id=None, **extra_fields)

    def log_bootstrap_started(self, **extra_fields: Any) -> dict[str, Any]:
        return self.log_lifecycle(phase="bootstrap", event="bootstrap_started", status="started", task_mode="notebook", episode_id=None, **extra_fields)

    def log_bootstrap_finished(
        self,
        *,
        status: str = "completed",
        level: str = "info",
        **extra_fields: Any,
    ) -> dict[str, Any]:
        return self.log_lifecycle(phase="bootstrap", event="bootstrap_finished", level=level, status=status, task_mode="notebook", episode_id=None, **extra_fields)

    def log_phase_started(
        self,
        *,
        phase: str,
        task_mode: str = "notebook",
        **extra_fields: Any,
    ) -> dict[str, Any]:
        return self.log_lifecycle(phase=phase, event="phase_started", status="started", task_mode=task_mode, episode_id=None, **extra_fields)

    def log_phase_finished(
        self,
        *,
        phase: str,
        task_mode: str = "notebook",
        status: str = "completed",
        level: str = "info",
        **extra_fields: Any,
    ) -> dict[str, Any]:
        return self.log_lifecycle(phase=phase, event="phase_finished", level=level, status=status, task_mode=task_mode, episode_id=None, **extra_fields)

    def log_episode_started(
        self,
        *,
        phase: str,
        task_mode: str,
        episode_id: str | None,
        **extra_fields: Any,
    ) -> dict[str, Any]:
        return self.log_lifecycle(phase=phase, event="episode_started", status="started", task_mode=task_mode, episode_id=episode_id, **extra_fields)

    def log_provider_call_started(
        self,
        *,
        phase: str,
        task_mode: str,
        episode_id: str | None,
        **extra_fields: Any,
    ) -> dict[str, Any]:
        return self.log_lifecycle(phase=phase, event="provider_call_started", status="started", task_mode=task_mode, episode_id=episode_id, **extra_fields)

    def log_provider_call_succeeded(
        self,
        *,
        phase: str,
        task_mode: str,
        episode_id: str | None,
        **extra_fields: Any,
    ) -> dict[str, Any]:
        return self.log_lifecycle(phase=phase, event="provider_call_succeeded", status="completed", task_mode=task_mode, episode_id=episode_id, **extra_fields)

    def log_provider_call_failed(
        self,
        *,
        phase: str,
        task_mode: str,
        episode_id: str | None,
        level: str = "error",
        status: str = "failed",
        **extra_fields: Any,
    ) -> dict[str, Any]:
        return self.log_lifecycle(phase=phase, event="provider_call_failed", level=level, status=status, task_mode=task_mode, episode_id=episode_id, **extra_fields)

    def log_response_parsed(
        self,
        *,
        phase: str,
        task_mode: str,
        episode_id: str | None,
        status: str = "valid",
        **extra_fields: Any,
    ) -> dict[str, Any]:
        return self.log_lifecycle(phase=phase, event="response_parsed", status=status, task_mode=task_mode, episode_id=episode_id, **extra_fields)

    def log_response_parse_failed(
        self,
        *,
        phase: str,
        task_mode: str,
        episode_id: str | None,
        status: str,
        level: str = "warning",
        **extra_fields: Any,
    ) -> dict[str, Any]:
        return self.log_lifecycle(phase=phase, event="response_parse_failed", level=level, status=status, task_mode=task_mode, episode_id=episode_id, **extra_fields)

    def log_episode_scored(
        self,
        *,
        phase: str,
        task_mode: str,
        episode_id: str | None,
        status: str,
        level: str = "info",
        **extra_fields: Any,
    ) -> dict[str, Any]:
        return self.log_lifecycle(phase=phase, event="episode_scored", level=level, status=status, task_mode=task_mode, episode_id=episode_id, **extra_fields)

    def log_payload_built(self, **extra_fields: Any) -> dict[str, Any]:
        return self.log_lifecycle(phase="canonical_payload", event="payload_built", status="completed", task_mode="notebook", episode_id=None, **extra_fields)

    def log_run_finished(
        self,
        *,
        status: str = "completed",
        level: str = "info",
        **extra_fields: Any,
    ) -> dict[str, Any]:
        return self.log_lifecycle(phase="run", event="run_finished", level=level, status=status, task_mode="notebook", episode_id=None, **extra_fields)

    def log_run_invalidated(
        self,
        *,
        phase: str,
        status: str = "invalidated",
        level: str = "error",
        **extra_fields: Any,
    ) -> dict[str, Any]:
        return self.log_lifecycle(phase=phase, event="run_invalidated", level=level, status=status, task_mode="notebook", episode_id=None, **extra_fields)

    def log_exception(
        self,
        exc: BaseException,
        *,
        phase: str,
        task_mode: str,
        episode_id: str | None = None,
        **extra_fields: Any,
    ) -> dict[str, Any]:
        exception_type = type(exc).__name__
        exception_message = str(exc)
        exception_record = self._emit(
            self._adapter(
                phase=phase,
                task_mode=task_mode,
                episode_id=episode_id,
                status="exception",
                exceptions=True,
            ),
            event="exception",
            level="error",
            exc_info=(type(exc), exc, exc.__traceback__),
            exception_type=exception_type,
            exception_message=exception_message,
            **extra_fields,
        )
        self.log(
            phase=phase,
            event="exception",
            level="error",
            status="exception",
            task_mode=task_mode,
            episode_id=episode_id,
            exception_type=exception_type,
            exception_message=exception_message,
        )
        return exception_record

    def summarize_exceptions(self) -> ExceptionSummary:
        summary = summarize_exception_log(self.exceptions_path)
        self.log(
            phase="run",
            event="exception_summary",
            level="warning" if summary.total > 0 else "info",
            status="exceptions_found" if summary.total > 0 else "clean",
            task_mode="notebook",
            episode_id=None,
            total_exceptions=summary.total,
            by_phase=summary.by_phase,
        )
        return summary
