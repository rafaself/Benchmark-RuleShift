from __future__ import annotations

import json

from core.kaggle import (
    EPISODE_RESULTS_FILENAME,
    EpisodeResultLedgerWriter,
    build_run_context,
)


def test_episode_ledger_appends_valid_jsonl_records_incrementally(tmp_path):
    context = build_run_context(
        repo_root=tmp_path,
        run_id="run-ledger-001",
        output_dir=tmp_path / "ledger-run",
    )
    ledger = EpisodeResultLedgerWriter(context)

    ledger.write_record(
        episode_id="ep-001",
        split="dev",
        task_mode="binary",
        call_status="completed",
        parse_status="valid",
        latency_ms=12,
        prediction=["attract", "repel", "attract", "repel"],
        target=["attract", "repel", "attract", "repel"],
        score={"num_correct": 4, "total": 4},
        exception_ref=None,
    )
    first_pass = [
        json.loads(line)
        for line in (tmp_path / "ledger-run" / EPISODE_RESULTS_FILENAME)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(first_pass) == 1
    assert first_pass[0]["episode_id"] == "ep-001"

    ledger.write_record(
        episode_id="ep-002",
        split="public_leaderboard",
        task_mode="narrative",
        call_status="failed",
        parse_status="operational_failure",
        latency_ms=7,
        prediction=None,
        target=["attract", "repel", "attract", "repel"],
        score={"num_correct": 0, "total": 4},
        exception_ref="exceptions.jsonl#2026-03-29T12:00:00Z",
    )

    second_pass = [
        json.loads(line)
        for line in (tmp_path / "ledger-run" / EPISODE_RESULTS_FILENAME)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(second_pass) == 2
    assert second_pass[1]["episode_id"] == "ep-002"
    assert second_pass[1]["exception_ref"] == "exceptions.jsonl#2026-03-29T12:00:00Z"
