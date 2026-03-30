from __future__ import annotations

from typing import Any

from core.kaggle.run_logging import BenchmarkRunContext
from core.kaggle.run_log_io import append_jsonl_record

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
        outcome_kind: str,
        failure_category: str | None,
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
            "outcome_kind": outcome_kind,
            "failure_category": failure_category,
            "latency_ms": latency_ms,
            "prediction": prediction,
            "target": target,
            "score": score,
            "exception_ref": exception_ref,
        }
        append_jsonl_record(self.path, record)
        return record
