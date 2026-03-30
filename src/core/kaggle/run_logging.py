from __future__ import annotations

import json
import logging
import os
import traceback as _tb
from pathlib import Path
from typing import Any

from core.kaggle.run_context import (
    RUN_ID_ENV_VAR,
    RUN_OUTPUT_DIR_ENV_VAR,
    BenchmarkRunContext,
    build_run_context,
)
from core.kaggle.run_log_io import (
    ExceptionSummary,
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
_RESERVED_LOG_RECORD_FIELDS = frozenset(
    logging.makeLogRecord({}).__dict__.keys()
)
_RESERVED_LOG_RECORD_FIELDS |= frozenset({"message", "asctime"})


class _JsonLinesFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _RESERVED_LOG_RECORD_FIELDS and not key.startswith("_")
        }
        payload.setdefault("timestamp", timestamp_utc())
        payload.setdefault("level", record.levelname.lower())
        return json.dumps(payload, ensure_ascii=True, sort_keys=True)


class _FsyncFileHandler(logging.FileHandler):
    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        if self.stream is None:
            return
        self.flush()
        os.fsync(self.stream.fileno())


class _ContextAdapter(logging.LoggerAdapter):
    def bind(self, **extra: Any) -> "_ContextAdapter":
        merged = dict(self.extra)
        merged.update(extra)
        return _ContextAdapter(self.logger, merged)

    def process(self, msg: Any, kwargs: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        merged = dict(self.extra)
        merged.update(kwargs.pop("extra", {}))
        kwargs["extra"] = merged
        return msg, kwargs


def _logger_key(path: Path) -> str:
    return str(path.resolve()).replace(os.sep, "_")


def _configure_file_logger(*, suffix: str, path: Path) -> logging.Logger:
    base_logger = logging.getLogger(_LOGGER_BASE_NAME)
    logger = base_logger.getChild(f"{suffix}.{_logger_key(path)}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    resolved_path = path.resolve()
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename).resolve() == resolved_path:
            return logger

    handler = _FsyncFileHandler(resolved_path, mode="a", encoding="utf-8", delay=True)
    handler.setLevel(logging.INFO)
    handler.setFormatter(_JsonLinesFormatter())
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
        self._benchmark_adapter = _ContextAdapter(
            _configure_file_logger(suffix="benchmark", path=self.log_path),
            {
                "run_id": self.context.run_id,
                "provider": self.context.provider,
                "model": self.context.model,
            },
        )
        self._exceptions_adapter = _ContextAdapter(
            _configure_file_logger(suffix="exceptions", path=self.exceptions_path),
            {
                "run_id": self.context.run_id,
                "provider": self.context.provider,
                "model": self.context.model,
            },
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
        record: dict[str, Any] = {
            "timestamp": timestamp_utc(),
            "run_id": self.context.run_id,
            "phase": phase,
            "event": event,
            "level": level,
            "episode_id": episode_id,
            "task_mode": task_mode,
            "provider": self.context.provider,
            "model": self.context.model,
            "status": status,
        }
        record.update(extra_fields)
        self._adapter(
            phase=phase,
            task_mode=task_mode,
            episode_id=episode_id,
            status=status,
        ).log(
            _levelno(level),
            event,
            extra={"event": event, "timestamp": record["timestamp"], **extra_fields},
        )
        return record

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
        return self.log_lifecycle(
            phase="run",
            event="run_started",
            level="info",
            status="started",
            task_mode="notebook",
            episode_id=None,
            **extra_fields,
        )

    def log_bootstrap_started(self, **extra_fields: Any) -> dict[str, Any]:
        return self.log_lifecycle(
            phase="bootstrap",
            event="bootstrap_started",
            level="info",
            status="started",
            task_mode="notebook",
            episode_id=None,
            **extra_fields,
        )

    def log_bootstrap_finished(
        self,
        *,
        status: str = "completed",
        level: str = "info",
        **extra_fields: Any,
    ) -> dict[str, Any]:
        return self.log_lifecycle(
            phase="bootstrap",
            event="bootstrap_finished",
            level=level,
            status=status,
            task_mode="notebook",
            episode_id=None,
            **extra_fields,
        )

    def log_phase_started(
        self,
        *,
        phase: str,
        task_mode: str = "notebook",
        **extra_fields: Any,
    ) -> dict[str, Any]:
        return self.log_lifecycle(
            phase=phase,
            event="phase_started",
            level="info",
            status="started",
            task_mode=task_mode,
            episode_id=None,
            **extra_fields,
        )

    def log_phase_finished(
        self,
        *,
        phase: str,
        task_mode: str = "notebook",
        status: str = "completed",
        level: str = "info",
        **extra_fields: Any,
    ) -> dict[str, Any]:
        return self.log_lifecycle(
            phase=phase,
            event="phase_finished",
            level=level,
            status=status,
            task_mode=task_mode,
            episode_id=None,
            **extra_fields,
        )

    def log_episode_started(
        self,
        *,
        phase: str,
        task_mode: str,
        episode_id: str | None,
        **extra_fields: Any,
    ) -> dict[str, Any]:
        return self.log_lifecycle(
            phase=phase,
            event="episode_started",
            level="info",
            status="started",
            task_mode=task_mode,
            episode_id=episode_id,
            **extra_fields,
        )

    def log_provider_call_started(
        self,
        *,
        phase: str,
        task_mode: str,
        episode_id: str | None,
        **extra_fields: Any,
    ) -> dict[str, Any]:
        return self.log_lifecycle(
            phase=phase,
            event="provider_call_started",
            level="info",
            status="started",
            task_mode=task_mode,
            episode_id=episode_id,
            **extra_fields,
        )

    def log_provider_call_succeeded(
        self,
        *,
        phase: str,
        task_mode: str,
        episode_id: str | None,
        **extra_fields: Any,
    ) -> dict[str, Any]:
        return self.log_lifecycle(
            phase=phase,
            event="provider_call_succeeded",
            level="info",
            status="completed",
            task_mode=task_mode,
            episode_id=episode_id,
            **extra_fields,
        )

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
        return self.log_lifecycle(
            phase=phase,
            event="provider_call_failed",
            level=level,
            status=status,
            task_mode=task_mode,
            episode_id=episode_id,
            **extra_fields,
        )

    def log_response_parsed(
        self,
        *,
        phase: str,
        task_mode: str,
        episode_id: str | None,
        status: str = "valid",
        **extra_fields: Any,
    ) -> dict[str, Any]:
        return self.log_lifecycle(
            phase=phase,
            event="response_parsed",
            level="info",
            status=status,
            task_mode=task_mode,
            episode_id=episode_id,
            **extra_fields,
        )

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
        return self.log_lifecycle(
            phase=phase,
            event="response_parse_failed",
            level=level,
            status=status,
            task_mode=task_mode,
            episode_id=episode_id,
            **extra_fields,
        )

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
        return self.log_lifecycle(
            phase=phase,
            event="episode_scored",
            level=level,
            status=status,
            task_mode=task_mode,
            episode_id=episode_id,
            **extra_fields,
        )

    def log_payload_built(self, **extra_fields: Any) -> dict[str, Any]:
        return self.log_lifecycle(
            phase="canonical_payload",
            event="payload_built",
            level="info",
            status="completed",
            task_mode="notebook",
            episode_id=None,
            **extra_fields,
        )

    def log_run_finished(
        self,
        *,
        status: str = "completed",
        level: str = "info",
        **extra_fields: Any,
    ) -> dict[str, Any]:
        return self.log_lifecycle(
            phase="run",
            event="run_finished",
            level=level,
            status=status,
            task_mode="notebook",
            episode_id=None,
            **extra_fields,
        )

    def log_run_invalidated(
        self,
        *,
        phase: str,
        status: str = "invalidated",
        level: str = "error",
        **extra_fields: Any,
    ) -> dict[str, Any]:
        return self.log_lifecycle(
            phase=phase,
            event="run_invalidated",
            level=level,
            status=status,
            task_mode="notebook",
            episode_id=None,
            **extra_fields,
        )

    def log_exception(
        self,
        exc: BaseException,
        *,
        phase: str,
        task_mode: str,
        episode_id: str | None = None,
        **extra_fields: Any,
    ) -> dict[str, Any]:
        tb_text = "".join(_tb.format_exception(type(exc), exc, exc.__traceback__))
        full_record: dict[str, Any] = {
            "timestamp": timestamp_utc(),
            "run_id": self.context.run_id,
            "phase": phase,
            "event": "exception",
            "level": "error",
            "episode_id": episode_id,
            "task_mode": task_mode,
            "provider": self.context.provider,
            "model": self.context.model,
            "status": "exception",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": tb_text,
        }
        full_record.update(extra_fields)
        self._adapter(
            phase=phase,
            task_mode=task_mode,
            episode_id=episode_id,
            status="exception",
            exceptions=True,
        ).error(
            "exception",
            extra={
                "event": "exception",
                "timestamp": full_record["timestamp"],
                "exception_type": full_record["exception_type"],
                "exception_message": full_record["exception_message"],
                "traceback": tb_text,
                **extra_fields,
            },
        )
        self.log(
            phase=phase,
            event="exception",
            level="error",
            status="exception",
            task_mode=task_mode,
            episode_id=episode_id,
            exception_type=type(exc).__name__,
            exception_message=str(exc),
        )
        return full_record

    def summarize_exceptions(self) -> ExceptionSummary:
        """Read exceptions.jsonl, count by phase, and emit a summary event."""
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
