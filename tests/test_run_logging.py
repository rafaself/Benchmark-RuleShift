from __future__ import annotations

import json

from core.kaggle import (
    BENCHMARK_LOG_FILENAME,
    EXCEPTIONS_LOG_FILENAME,
    BenchmarkRunLogger,
    ExceptionSummary,
    build_run_context,
)


class _LLMIdentityStub:
    provider_name = "shim-provider"
    model_name = "shim-model"


def test_benchmark_run_logger_writes_reconstructable_lifecycle_records(tmp_path):
    context = build_run_context(
        repo_root=tmp_path,
        llm=_LLMIdentityStub(),
        run_id="run-life-001",
        output_dir=tmp_path / "life-run",
    )
    logger = BenchmarkRunLogger(context)

    logger.log_run_started(output_dir=str(context.output_dir))
    logger.log_bootstrap_started(detail="loading splits", total=2)
    logger.log_bootstrap_finished(detail="splits ready", processed=2, total=2)
    logger.log_phase_started(phase="official_binary_evaluation", task_mode="binary", total=1)
    logger.log_episode_started(phase="official_binary_evaluation", task_mode="binary", episode_id="ep-001")
    logger.log_provider_call_started(phase="official_binary_evaluation", task_mode="binary", episode_id="ep-001")
    logger.log_provider_call_succeeded(phase="official_binary_evaluation", task_mode="binary", episode_id="ep-001")
    logger.log_response_parsed(
        phase="official_binary_evaluation",
        task_mode="binary",
        episode_id="ep-001",
        parse_status="valid",
    )
    logger.log_episode_scored(
        phase="official_binary_evaluation",
        task_mode="binary",
        episode_id="ep-001",
        status="valid",
        num_correct=4,
        total=4,
    )
    logger.log_phase_finished(phase="official_binary_evaluation", task_mode="binary", processed=1, total=1)
    logger.log_payload_built(total_episodes=1)
    logger.log_run_finished(total_episodes=1)

    records = [
        json.loads(line)
        for line in (tmp_path / "life-run" / BENCHMARK_LOG_FILENAME).read_text(encoding="utf-8").splitlines()
    ]

    assert [record["event"] for record in records] == [
        "run_started",
        "bootstrap_started",
        "bootstrap_finished",
        "phase_started",
        "episode_started",
        "provider_call_started",
        "provider_call_succeeded",
        "response_parsed",
        "episode_scored",
        "phase_finished",
        "payload_built",
        "run_finished",
    ]
    for record in records:
        assert record["run_id"] == "run-life-001"
        assert record["provider"] == "shim-provider"
        assert record["model"] == "shim-model"


def test_log_exception_writes_compact_and_detailed_artifacts(tmp_path):
    context = build_run_context(
        repo_root=tmp_path,
        llm=_LLMIdentityStub(),
        run_id="run-exc-001",
        output_dir=tmp_path / "exc-run",
    )
    logger = BenchmarkRunLogger(context)

    try:
        raise RuntimeError("provider timeout")
    except RuntimeError as exc:
        logger.log_exception(
            exc,
            phase="official_binary_evaluation",
            task_mode="binary",
            episode_id="ep-abc",
        )

    benchmark_record = json.loads(
        (tmp_path / "exc-run" / BENCHMARK_LOG_FILENAME).read_text(encoding="utf-8").splitlines()[0]
    )
    exception_record = json.loads(
        (tmp_path / "exc-run" / EXCEPTIONS_LOG_FILENAME).read_text(encoding="utf-8").splitlines()[0]
    )

    assert benchmark_record["event"] == "exception"
    assert benchmark_record["exception_type"] == "RuntimeError"
    assert "traceback" not in benchmark_record

    assert exception_record["event"] == "exception"
    assert exception_record["episode_id"] == "ep-abc"
    assert exception_record["exception_type"] == "RuntimeError"
    assert "RuntimeError" in exception_record["traceback"]


def test_summarize_exceptions_reports_clean_and_non_clean_runs(tmp_path):
    clean_logger = BenchmarkRunLogger(
        build_run_context(repo_root=tmp_path, run_id="run-clean", output_dir=tmp_path / "clean-run")
    )
    clean_summary = clean_logger.summarize_exceptions()

    assert isinstance(clean_summary, ExceptionSummary)
    assert clean_summary.total == 0
    assert clean_summary.by_phase == {}

    noisy_context = build_run_context(
        repo_root=tmp_path,
        llm=_LLMIdentityStub(),
        run_id="run-noisy",
        output_dir=tmp_path / "noisy-run",
    )
    noisy_logger = BenchmarkRunLogger(noisy_context)
    for phase, exc in [
        ("official_binary_evaluation", ValueError("bad response")),
        ("official_binary_evaluation", RuntimeError("timeout")),
        ("official_narrative_evaluation", OSError("connection lost")),
    ]:
        try:
            raise exc
        except Exception as error:
            noisy_logger.log_exception(error, phase=phase, task_mode="binary")

    noisy_summary = noisy_logger.summarize_exceptions()
    assert noisy_summary.total == 3
    assert noisy_summary.by_phase == {
        "official_binary_evaluation": 2,
        "official_narrative_evaluation": 1,
    }


def test_reinitializing_logger_for_same_run_does_not_duplicate_handler_writes(tmp_path):
    context = build_run_context(
        repo_root=tmp_path,
        llm=_LLMIdentityStub(),
        run_id="run-reinit-001",
        output_dir=tmp_path / "reinit-run",
    )
    BenchmarkRunLogger(context)
    logger = BenchmarkRunLogger(context)

    logger.log_run_started(output_dir=str(context.output_dir))

    records = [
        json.loads(line)
        for line in (tmp_path / "reinit-run" / BENCHMARK_LOG_FILENAME).read_text(encoding="utf-8").splitlines()
    ]

    assert len(records) == 1
    assert records[0]["event"] == "run_started"
