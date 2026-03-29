from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from core.kaggle.run_logging import BenchmarkRunContext

__all__ = [
    "EPISODE_RESULTS_FILENAME",
    "EpisodeResultLedgerWriter",
]

EPISODE_RESULTS_FILENAME = "episode_results.jsonl"


class EpisodeResultLedgerWriter:
    """Append-only JSONL writer for per-episode primary execution evidence."""

    def __init__(self, context: BenchmarkRunContext) -> None:
        self.context = context
        self.context.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.context.output_dir / EPISODE_RESULTS_FILENAME
        self.path.touch(exist_ok=True)

    def write_record(
        self,
        *,
        episode_id: str | None,
        split: str | None,
        task_mode: str,
        call_status: str,
        parse_status: str | None,
        latency_ms: int | None,
        prediction: list[str] | None,
        target: list[str] | None,
        score: dict[str, int] | None,
        exception_ref: str | None,
    ) -> dict[str, Any]:
        record = {
            "run_id": self.context.run_id,
            "episode_id": episode_id,
            "split": split,
            "task_mode": task_mode,
            "provider": self.context.provider,
            "model": self.context.model,
            "call_status": call_status,
            "parse_status": parse_status,
            "latency_ms": latency_ms,
            "prediction": prediction,
            "target": target,
            "score": score,
            "exception_ref": exception_ref,
        }
        serialized = json.dumps(record, ensure_ascii=True, sort_keys=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return record
