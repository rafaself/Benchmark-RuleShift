from __future__ import annotations

import json

from core.kaggle import (
    BENCHMARK_LOG_FILENAME,
    EPISODE_RESULTS_FILENAME,
    EXCEPTIONS_LOG_FILENAME,
    FAILURE_CATEGORY_INVALID_PREDICTION_SHAPE,
    FAILURE_CATEGORY_TIMEOUT,
    OPERATIONAL_FAILURE_STATUS,
    OUTCOME_KIND_MODEL_PARSE_FAILURE,
    OUTCOME_KIND_OPERATIONAL_FAILURE,
    BenchmarkRunLogger,
    EpisodeResultLedgerWriter,
    build_run_context,
    run_binary_episode,
    run_narrative_episode,
)
from core.parser import NarrativeParseStatus, ParseStatus


class _RaisingLLM:
    provider_name = "shim-provider"
    model_name = "shim-model"

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def prompt(self, *_args, **_kwargs):
        raise self._exc


class _ValidBinaryLLM:
    provider_name = "shim-provider"
    model_name = "shim-model"

    def prompt(self, *_args, **_kwargs):
        return "attract, repel, attract, repel"


def test_run_binary_episode_records_successful_prediction_payloads(tmp_path):
    context = build_run_context(
        repo_root=tmp_path,
        llm=_ValidBinaryLLM(),
        run_id="run-binary-success",
        output_dir=tmp_path / "binary-success",
    )
    logger = BenchmarkRunLogger(context)
    ledger = EpisodeResultLedgerWriter(context)

    result = run_binary_episode(
        llm=_ValidBinaryLLM(),
        prompt_binary="prompt",
        probe_targets=("attract", "repel", "attract", "repel"),
        logger=logger,
        ledger=ledger,
        phase="official_binary_evaluation",
        task_mode="binary",
        episode_id="ep-000",
        split="dev",
    )

    assert result.status == ParseStatus.VALID.value
    assert result.score == (4, 4)
    assert result.parsed_prediction.status is ParseStatus.VALID

    ledger_records = [
        json.loads(line)
        for line in (tmp_path / "binary-success" / EPISODE_RESULTS_FILENAME)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert ledger_records == [
        {
            "run_id": "run-binary-success",
            "episode_id": "ep-000",
            "split": "dev",
            "task_mode": "binary",
            "provider": "shim-provider",
            "model": "shim-model",
            "call_status": "completed",
            "parse_status": "valid",
            "outcome_kind": "scored_model_result",
            "failure_category": None,
            "latency_ms": ledger_records[0]["latency_ms"],
            "prediction": ["attract", "repel", "attract", "repel"],
            "target": ["attract", "repel", "attract", "repel"],
            "score": {"num_correct": 4, "total": 4},
            "exception_ref": None,
        }
    ]
    assert isinstance(ledger_records[0]["latency_ms"], int)
    assert ledger_records[0]["latency_ms"] >= 0


def test_run_binary_episode_marks_provider_exceptions_as_operational_failures(tmp_path):
    context = build_run_context(
        repo_root=tmp_path,
        llm=_RaisingLLM(RuntimeError("provider timeout")),
        run_id="run-binary-opfail",
        output_dir=tmp_path / "binary-opfail",
    )
    logger = BenchmarkRunLogger(context)
    ledger = EpisodeResultLedgerWriter(context)

    result = run_binary_episode(
        llm=_RaisingLLM(RuntimeError("provider timeout")),
        prompt_binary="prompt",
        probe_targets=("attract", "repel", "attract", "repel"),
        logger=logger,
        ledger=ledger,
        phase="official_binary_evaluation",
        task_mode="binary",
        episode_id="ep-001",
        split="public_leaderboard",
    )

    assert result.status == OPERATIONAL_FAILURE_STATUS
    assert result.score == (0, 4)
    assert result.parsed_prediction.status is ParseStatus.SKIPPED_PROVIDER_FAILURE

    log_records = [
        json.loads(line)
        for line in (tmp_path / "binary-opfail" / BENCHMARK_LOG_FILENAME)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["event"] for record in log_records] == [
        "episode_started",
        "provider_call_started",
        "provider_call_failed",
        "exception",
        "response_parse_failed",
        "episode_scored",
    ]
    assert log_records[2]["status"] == "failed"
    assert log_records[2]["failure_stage"] == "provider_call"
    assert log_records[2]["failure_category"] == FAILURE_CATEGORY_TIMEOUT
    assert log_records[4]["status"] == OPERATIONAL_FAILURE_STATUS
    assert log_records[4]["detail"].startswith("Operational failure during provider_call")
    assert log_records[4]["failure_category"] == FAILURE_CATEGORY_TIMEOUT
    assert log_records[5]["status"] == OPERATIONAL_FAILURE_STATUS
    assert log_records[5]["level"] == "error"
    assert log_records[5]["failure_category"] == FAILURE_CATEGORY_TIMEOUT

    exception_records = [
        json.loads(line)
        for line in (tmp_path / "binary-opfail" / EXCEPTIONS_LOG_FILENAME)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(exception_records) == 1
    assert exception_records[0]["episode_id"] == "ep-001"
    assert exception_records[0]["failure_stage"] == "provider_call"
    assert exception_records[0]["exception_type"] == "RuntimeError"
    assert exception_records[0]["failure_category"] == FAILURE_CATEGORY_TIMEOUT

    ledger_records = [
        json.loads(line)
        for line in (tmp_path / "binary-opfail" / EPISODE_RESULTS_FILENAME)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(ledger_records) == 1
    assert ledger_records[0] == {
        "run_id": "run-binary-opfail",
        "episode_id": "ep-001",
        "split": "public_leaderboard",
        "task_mode": "binary",
        "provider": "shim-provider",
        "model": "shim-model",
        "call_status": "failed",
        "parse_status": OPERATIONAL_FAILURE_STATUS,
        "outcome_kind": OUTCOME_KIND_OPERATIONAL_FAILURE,
        "failure_category": FAILURE_CATEGORY_TIMEOUT,
        "latency_ms": ledger_records[0]["latency_ms"],
        "prediction": None,
        "target": ["attract", "repel", "attract", "repel"],
        "score": {"num_correct": 0, "total": 4},
        "exception_ref": ledger_records[0]["exception_ref"],
    }
    assert isinstance(ledger_records[0]["latency_ms"], int)
    assert ledger_records[0]["latency_ms"] >= 0
    assert ledger_records[0]["exception_ref"].startswith("exceptions.jsonl#")


def test_run_narrative_episode_marks_scoring_exceptions_as_operational_failures(tmp_path):
    logger = BenchmarkRunLogger(
        build_run_context(
            repo_root=tmp_path,
            llm=_ValidBinaryLLM(),
            run_id="run-narrative-opfail",
            output_dir=tmp_path / "narrative-opfail",
        )
    )

    result = run_narrative_episode(
        llm=_ValidBinaryLLM(),
        prompt_narrative=(
            "rule_before: old\n"
            "shift_evidence: new\n"
            "rule_after: switched\n"
            "final_decision: attract, repel, attract, repel"
        ),
        probe_targets=("attract", "repel"),
        logger=logger,
        phase="official_narrative_evaluation",
        task_mode="narrative",
        episode_id="ep-002",
    )

    assert result.status == OPERATIONAL_FAILURE_STATUS
    assert result.score == (0, 4)
    assert result.parsed_result.status is NarrativeParseStatus.SKIPPED_PROVIDER_FAILURE

    log_records = [
        json.loads(line)
        for line in (tmp_path / "narrative-opfail" / BENCHMARK_LOG_FILENAME)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["event"] for record in log_records] == [
        "episode_started",
        "provider_call_started",
        "provider_call_succeeded",
        "response_parse_failed",
        "exception",
        "response_parse_failed",
        "episode_scored",
    ]
    assert log_records[3]["status"] == "invalid_format"
    assert log_records[3]["failure_category"] == "narrative_parse_failure"
    assert log_records[3]["outcome_kind"] == OUTCOME_KIND_MODEL_PARSE_FAILURE
    assert log_records[5]["failure_stage"] == "episode_scoring"
    assert log_records[5]["status"] == OPERATIONAL_FAILURE_STATUS
    assert log_records[5]["failure_category"] == FAILURE_CATEGORY_INVALID_PREDICTION_SHAPE
    assert log_records[6]["status"] == OPERATIONAL_FAILURE_STATUS
    assert log_records[6]["failure_category"] == FAILURE_CATEGORY_INVALID_PREDICTION_SHAPE

    exception_records = [
        json.loads(line)
        for line in (tmp_path / "narrative-opfail" / EXCEPTIONS_LOG_FILENAME)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(exception_records) == 1
    assert exception_records[0]["failure_stage"] == "episode_scoring"
    assert exception_records[0]["failure_category"] == FAILURE_CATEGORY_INVALID_PREDICTION_SHAPE
