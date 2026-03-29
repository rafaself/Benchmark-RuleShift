from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from core.kaggle.episode_ledger import EpisodeResultLedgerWriter
from core.kaggle.run_logging import EXCEPTIONS_LOG_FILENAME
from core.kaggle.run_logging import BenchmarkRunLogger
from core.kaggle.types import (
    BinaryResponse,
    parse_binary_response,
    parse_narrative_response,
    score_episode,
)
from core.parser import (
    NarrativeParsedResult,
    NarrativeParseStatus,
    ParseStatus,
    ParsedPrediction,
)
from tasks.ruleshift_benchmark.protocol import PROBE_COUNT

__all__ = [
    "OPERATIONAL_FAILURE_STATUS",
    "BinaryEpisodeExecution",
    "NarrativeEpisodeExecution",
    "run_binary_episode",
    "run_narrative_episode",
]

OPERATIONAL_FAILURE_STATUS = "operational_failure"


@dataclass(frozen=True, slots=True)
class BinaryEpisodeExecution:
    parsed_prediction: ParsedPrediction
    score: tuple[int, int]
    status: str


@dataclass(frozen=True, slots=True)
class NarrativeEpisodeExecution:
    parsed_result: NarrativeParsedResult
    score: tuple[int, int]
    status: str


def run_binary_episode(
    *,
    llm: object,
    prompt_binary: str,
    probe_targets: tuple,
    logger: BenchmarkRunLogger,
    ledger: EpisodeResultLedgerWriter | None = None,
    phase: str,
    task_mode: str,
    episode_id: str | None,
    split: str | None = None,
) -> BinaryEpisodeExecution:
    logger.log_episode_started(
        phase=phase,
        task_mode=task_mode,
        episode_id=episode_id,
    )
    logger.log_provider_call_started(
        phase=phase,
        task_mode=task_mode,
        episode_id=episode_id,
    )
    call_started_at = monotonic()

    try:
        response = llm.prompt(prompt_binary, schema=BinaryResponse)
    except Exception as exc:
        return _binary_operational_failure(
            exc,
            logger=logger,
            ledger=ledger,
            phase=phase,
            task_mode=task_mode,
            episode_id=episode_id,
            split=split,
            target=probe_targets,
            latency_ms=_duration_ms(call_started_at),
            failure_stage="provider_call",
        )

    logger.log_provider_call_succeeded(
        phase=phase,
        task_mode=task_mode,
        episode_id=episode_id,
    )

    try:
        parsed_prediction = parse_binary_response(response)
    except Exception as exc:
        return _binary_operational_failure(
            exc,
            logger=logger,
            ledger=ledger,
            phase=phase,
            task_mode=task_mode,
            episode_id=episode_id,
            split=split,
            target=probe_targets,
            latency_ms=_duration_ms(call_started_at),
            failure_stage="response_parse",
        )

    if parsed_prediction.status is ParseStatus.VALID:
        logger.log_response_parsed(
            phase=phase,
            task_mode=task_mode,
            episode_id=episode_id,
            status=parsed_prediction.status.value,
            parse_status=parsed_prediction.status.value,
        )
    else:
        logger.log_response_parse_failed(
            phase=phase,
            task_mode=task_mode,
            episode_id=episode_id,
            status=parsed_prediction.status.value,
            parse_status=parsed_prediction.status.value,
        )

    try:
        predictions = (
            tuple(label.value for label in parsed_prediction.labels)
            if parsed_prediction.status is ParseStatus.VALID
            else None
        )
        score = score_episode(predictions, probe_targets)
    except Exception as exc:
        return _binary_operational_failure(
            exc,
            logger=logger,
            ledger=ledger,
            phase=phase,
            task_mode=task_mode,
            episode_id=episode_id,
            split=split,
            target=probe_targets,
            latency_ms=_duration_ms(call_started_at),
            failure_stage="episode_scoring",
        )

    logger.log_episode_scored(
        phase=phase,
        task_mode=task_mode,
        episode_id=episode_id,
        level="info" if parsed_prediction.status is ParseStatus.VALID else "warning",
        status=parsed_prediction.status.value,
        num_correct=score[0],
        total=score[1],
    )
    if ledger is not None:
        ledger.write_record(
            episode_id=episode_id,
            split=split,
            task_mode=task_mode,
            call_status="completed",
            parse_status=parsed_prediction.status.value,
            latency_ms=_duration_ms(call_started_at),
            prediction=_binary_prediction_payload(parsed_prediction),
            target=_labels_payload(probe_targets),
            score=_score_payload(score),
            exception_ref=None,
        )
    return BinaryEpisodeExecution(
        parsed_prediction=parsed_prediction,
        score=score,
        status=parsed_prediction.status.value,
    )


def run_narrative_episode(
    *,
    llm: object,
    prompt_narrative: str,
    probe_targets: tuple,
    logger: BenchmarkRunLogger,
    ledger: EpisodeResultLedgerWriter | None = None,
    phase: str,
    task_mode: str,
    episode_id: str | None,
    split: str | None = None,
) -> NarrativeEpisodeExecution:
    logger.log_episode_started(
        phase=phase,
        task_mode=task_mode,
        episode_id=episode_id,
    )
    logger.log_provider_call_started(
        phase=phase,
        task_mode=task_mode,
        episode_id=episode_id,
    )
    call_started_at = monotonic()

    try:
        response = llm.prompt(prompt_narrative)
    except Exception as exc:
        return _narrative_operational_failure(
            exc,
            logger=logger,
            ledger=ledger,
            phase=phase,
            task_mode=task_mode,
            episode_id=episode_id,
            split=split,
            target=probe_targets,
            latency_ms=_duration_ms(call_started_at),
            failure_stage="provider_call",
        )

    logger.log_provider_call_succeeded(
        phase=phase,
        task_mode=task_mode,
        episode_id=episode_id,
    )

    try:
        parsed_result = parse_narrative_response(response)
    except Exception as exc:
        return _narrative_operational_failure(
            exc,
            logger=logger,
            ledger=ledger,
            phase=phase,
            task_mode=task_mode,
            episode_id=episode_id,
            split=split,
            target=probe_targets,
            latency_ms=_duration_ms(call_started_at),
            failure_stage="response_parse",
        )

    if parsed_result.status is NarrativeParseStatus.VALID:
        logger.log_response_parsed(
            phase=phase,
            task_mode=task_mode,
            episode_id=episode_id,
            status=parsed_result.status.value,
            parse_status=parsed_result.status.value,
        )
    else:
        logger.log_response_parse_failed(
            phase=phase,
            task_mode=task_mode,
            episode_id=episode_id,
            status=parsed_result.status.value,
            parse_status=parsed_result.status.value,
            failure_detail=parsed_result.failure_detail,
        )

    try:
        predictions = (
            tuple(label.value for label in parsed_result.output.final_decision)
            if parsed_result.status is NarrativeParseStatus.VALID and parsed_result.output is not None
            else None
        )
        score = score_episode(predictions, probe_targets)
    except Exception as exc:
        return _narrative_operational_failure(
            exc,
            logger=logger,
            ledger=ledger,
            phase=phase,
            task_mode=task_mode,
            episode_id=episode_id,
            split=split,
            target=probe_targets,
            latency_ms=_duration_ms(call_started_at),
            failure_stage="episode_scoring",
        )

    logger.log_episode_scored(
        phase=phase,
        task_mode=task_mode,
        episode_id=episode_id,
        level="info" if parsed_result.status is NarrativeParseStatus.VALID else "warning",
        status=parsed_result.status.value,
        num_correct=score[0],
        total=score[1],
    )
    if ledger is not None:
        ledger.write_record(
            episode_id=episode_id,
            split=split,
            task_mode=task_mode,
            call_status="completed",
            parse_status=parsed_result.status.value,
            latency_ms=_duration_ms(call_started_at),
            prediction=_narrative_prediction_payload(parsed_result),
            target=_labels_payload(probe_targets),
            score=_score_payload(score),
            exception_ref=None,
        )
    return NarrativeEpisodeExecution(
        parsed_result=parsed_result,
        score=score,
        status=parsed_result.status.value,
    )


def _binary_operational_failure(
    exc: BaseException,
    *,
    logger: BenchmarkRunLogger,
    ledger: EpisodeResultLedgerWriter | None,
    phase: str,
    task_mode: str,
    episode_id: str | None,
    split: str | None,
    target: tuple,
    latency_ms: int | None,
    failure_stage: str,
) -> BinaryEpisodeExecution:
    exception_ref = _log_operational_failure(
        exc,
        logger=logger,
        phase=phase,
        task_mode=task_mode,
        episode_id=episode_id,
        failure_stage=failure_stage,
    )
    score = (0, PROBE_COUNT)
    logger.log_episode_scored(
        phase=phase,
        task_mode=task_mode,
        episode_id=episode_id,
        level="error",
        status=OPERATIONAL_FAILURE_STATUS,
        num_correct=score[0],
        total=score[1],
        failure_stage=failure_stage,
    )
    if ledger is not None:
        ledger.write_record(
            episode_id=episode_id,
            split=split,
            task_mode=task_mode,
            call_status="failed" if failure_stage == "provider_call" else "completed",
            parse_status=OPERATIONAL_FAILURE_STATUS,
            latency_ms=latency_ms,
            prediction=None,
            target=_labels_payload(target),
            score=_score_payload(score),
            exception_ref=exception_ref,
        )
    return BinaryEpisodeExecution(
        parsed_prediction=ParsedPrediction.skipped_provider_failure(),
        score=score,
        status=OPERATIONAL_FAILURE_STATUS,
    )


def _narrative_operational_failure(
    exc: BaseException,
    *,
    logger: BenchmarkRunLogger,
    ledger: EpisodeResultLedgerWriter | None,
    phase: str,
    task_mode: str,
    episode_id: str | None,
    split: str | None,
    target: tuple,
    latency_ms: int | None,
    failure_stage: str,
) -> NarrativeEpisodeExecution:
    exception_ref = _log_operational_failure(
        exc,
        logger=logger,
        phase=phase,
        task_mode=task_mode,
        episode_id=episode_id,
        failure_stage=failure_stage,
    )
    score = (0, PROBE_COUNT)
    logger.log_episode_scored(
        phase=phase,
        task_mode=task_mode,
        episode_id=episode_id,
        level="error",
        status=OPERATIONAL_FAILURE_STATUS,
        num_correct=score[0],
        total=score[1],
        failure_stage=failure_stage,
    )
    if ledger is not None:
        ledger.write_record(
            episode_id=episode_id,
            split=split,
            task_mode=task_mode,
            call_status="failed" if failure_stage == "provider_call" else "completed",
            parse_status=OPERATIONAL_FAILURE_STATUS,
            latency_ms=latency_ms,
            prediction=None,
            target=_labels_payload(target),
            score=_score_payload(score),
            exception_ref=exception_ref,
        )
    return NarrativeEpisodeExecution(
        parsed_result=NarrativeParsedResult.skipped_provider_failure(),
        score=score,
        status=OPERATIONAL_FAILURE_STATUS,
    )


def _log_operational_failure(
    exc: BaseException,
    *,
    logger: BenchmarkRunLogger,
    phase: str,
    task_mode: str,
    episode_id: str | None,
    failure_stage: str,
) -> str:
    detail = _format_operational_failure_detail(failure_stage, exc)
    if failure_stage == "provider_call":
        logger.log_provider_call_failed(
            phase=phase,
            task_mode=task_mode,
            episode_id=episode_id,
            exception_type=type(exc).__name__,
            exception_message=str(exc),
            failure_stage=failure_stage,
            detail=detail,
        )
    exception_record = logger.log_exception(
        exc,
        phase=phase,
        task_mode=task_mode,
        episode_id=episode_id,
        failure_stage=failure_stage,
    )
    logger.log_response_parse_failed(
        phase=phase,
        task_mode=task_mode,
        episode_id=episode_id,
        status=OPERATIONAL_FAILURE_STATUS,
        level="error",
        parse_status=OPERATIONAL_FAILURE_STATUS,
        failure_stage=failure_stage,
        detail=detail,
    )
    return f"{EXCEPTIONS_LOG_FILENAME}#{exception_record['timestamp']}"


def _format_operational_failure_detail(
    failure_stage: str,
    exc: BaseException,
) -> str:
    return (
        f"Operational failure during {failure_stage}: "
        f"{type(exc).__name__}: {exc}"
    )


def _duration_ms(started_at: float) -> int:
    return max(0, int((monotonic() - started_at) * 1000))


def _binary_prediction_payload(
    parsed_prediction: ParsedPrediction,
) -> list[str] | None:
    if parsed_prediction.status is not ParseStatus.VALID:
        return None
    return [label.value for label in parsed_prediction.labels]


def _narrative_prediction_payload(
    parsed_result: NarrativeParsedResult,
) -> list[str] | None:
    if parsed_result.status is not NarrativeParseStatus.VALID:
        return None
    if parsed_result.output is None:
        return None
    return [label.value for label in parsed_result.output.final_decision]


def _labels_payload(labels: tuple | None) -> list[str] | None:
    if labels is None:
        return None
    return [
        str(getattr(label, "value", label))
        for label in labels
    ]


def _score_payload(score: tuple[int, int]) -> dict[str, int]:
    return {
        "num_correct": score[0],
        "total": score[1],
    }
