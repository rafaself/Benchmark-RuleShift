from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ExceptionSummary:
    total: int
    by_phase: dict[str, int]


def append_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    serialized = json.dumps(record, ensure_ascii=True, sort_keys=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(serialized)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def summarize_exception_log(path: Path) -> ExceptionSummary:
    by_phase: dict[str, int] = {}
    total = 0
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
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
