from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.kaggle.episode_ledger import EPISODE_RESULTS_FILENAME
from core.kaggle.run_log_io import read_jsonl_records
from core.kaggle.run_logging import (
    BENCHMARK_LOG_FILENAME,
    EXCEPTIONS_LOG_FILENAME,
    BenchmarkRunContext,
)

__all__ = [
    "DIAGNOSTICS_SUMMARY_FILENAME",
    "build_diagnostics_summary",
    "write_diagnostics_summary",
]

DIAGNOSTICS_SUMMARY_FILENAME = "diagnostics_summary.json"


def build_diagnostics_summary(
    *,
    context: BenchmarkRunContext,
    binary_parse_valid_rate: float | None = None,
    narrative_schema_valid_rate: float | None = None,
) -> dict[str, Any]:
    """Build a compact, reviewer-friendly run-health summary.

    Parse-validity rates are accepted as optional inputs because the canonical
    metric computation already happens in the notebook runtime. When those
    values are unavailable at finalization time, later releases can backfill a
    durable source of truth without changing the artifact shape.
    """
    log_records = read_jsonl_records(context.output_dir / BENCHMARK_LOG_FILENAME)
    exception_records = read_jsonl_records(context.output_dir / EXCEPTIONS_LOG_FILENAME)
    episode_records = read_jsonl_records(context.output_dir / EPISODE_RESULTS_FILENAME)
    invalidation_reasons = _collect_invalidation_reasons(log_records)

    return {
        "run_id": context.run_id,
        "run_valid": len(invalidation_reasons) == 0,
        "invalidation_reasons": invalidation_reasons,
        "binary_parse_valid_rate": binary_parse_valid_rate,
        "narrative_schema_valid_rate": narrative_schema_valid_rate,
        "provider_failure_count": _provider_failure_count(
            episode_records=episode_records,
            log_records=log_records,
        ),
        "failure_category_counts": _collect_failure_category_counts(
            episode_records=episode_records,
            log_records=log_records,
        ),
        "total_exception_count": len(exception_records),
        "total_logged_events": len(log_records),
        "started_at": _find_event_timestamp(log_records, event="run_started"),
        "finished_at": _find_event_timestamp(log_records, event="run_finished", last=True),
    }


def write_diagnostics_summary(
    *,
    context: BenchmarkRunContext,
    binary_parse_valid_rate: float | None = None,
    narrative_schema_valid_rate: float | None = None,
) -> Path:
    summary = build_diagnostics_summary(
        context=context,
        binary_parse_valid_rate=binary_parse_valid_rate,
        narrative_schema_valid_rate=narrative_schema_valid_rate,
    )
    summary_path = context.output_dir / DIAGNOSTICS_SUMMARY_FILENAME
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary_path


def _collect_invalidation_reasons(log_records: list[dict[str, Any]]) -> list[str]:
    ordered_reasons: list[str] = []
    seen_reasons: set[str] = set()
    for record in log_records:
        if record.get("event") != "run_invalidated":
            continue
        reason = record.get("reason")
        if not isinstance(reason, str) or not reason:
            continue
        if reason in seen_reasons:
            continue
        seen_reasons.add(reason)
        ordered_reasons.append(reason)
    return ordered_reasons


def _collect_failure_category_counts(
    *,
    episode_records: list[dict[str, Any]],
    log_records: list[dict[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    if episode_records:
        for record in episode_records:
            category = record.get("failure_category")
            if not isinstance(category, str) or not category:
                continue
            counts[category] = counts.get(category, 0) + 1
        return counts

    for record in log_records:
        if record.get("event") != "response_parse_failed":
            continue
        category = _failure_category(record)
        counts[category] = counts.get(category, 0) + 1
    return counts


def _provider_failure_count(
    *,
    episode_records: list[dict[str, Any]],
    log_records: list[dict[str, Any]],
) -> int:
    provider_side_categories = {
        "provider_failure",
        "timeout",
        "transport_failure",
    }
    if episode_records:
        return sum(
            1
            for record in episode_records
            if record.get("failure_category") in provider_side_categories
        )
    return sum(
        1 for record in log_records
        if record.get("failure_category") in provider_side_categories
        or record.get("event") == "provider_call_failed"
    )


def _failure_category(record: dict[str, Any]) -> str:
    for key in ("failure_category", "failure_stage", "parse_status", "status"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    return "unknown"


def _find_event_timestamp(
    log_records: list[dict[str, Any]],
    *,
    event: str,
    last: bool = False,
) -> str | None:
    selected_timestamp: str | None = None
    for record in log_records:
        if record.get("event") != event:
            continue
        timestamp = record.get("timestamp")
        if isinstance(timestamp, str) and timestamp:
            selected_timestamp = timestamp
            if not last:
                return selected_timestamp
    return selected_timestamp
