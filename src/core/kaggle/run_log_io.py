from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ExceptionSummary:
    total: int
    by_phase: dict[str, int]


class JsonLinesFormatter(logging.Formatter):
    def __init__(self, *, include_traceback: bool = False) -> None:
        super().__init__()
        self._include_traceback = include_traceback

    def format(self, record: logging.LogRecord) -> str:
        record_payload = getattr(record, "payload", {})
        payload = dict(record_payload) if isinstance(record_payload, dict) else {}
        payload.setdefault("timestamp", timestamp_utc())
        payload.setdefault("level", record.levelname.lower())
        if self._include_traceback and record.exc_info:
            payload.setdefault("traceback", self.formatException(record.exc_info))
        return json.dumps(payload, ensure_ascii=True, sort_keys=True)


class DurableFileHandler(logging.FileHandler):
    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        if self.stream is None:
            return
        self.flush()
        os.fsync(self.stream.fileno())


def append_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    serialized = json.dumps(record, ensure_ascii=True, sort_keys=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(serialized)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def summarize_exception_log(path: Path) -> ExceptionSummary:
    by_phase: dict[str, int] = {}
    total = 0
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    phase_key = "_unparseable"
                else:
                    phase_key = record.get("phase", "_unknown")
                by_phase[phase_key] = by_phase.get(phase_key, 0) + 1
                total += 1
    return ExceptionSummary(total=total, by_phase=by_phase)


def timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00",
        "Z",
    )
