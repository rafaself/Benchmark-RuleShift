from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from core.kaggle.episode_ledger import EpisodeResultLedgerWriter
from core.kaggle.execution_artifacts import (
    binary_prediction_payload,
    duration_ms,
    labels_payload,
    narrative_prediction_payload,
    record_operational_failure,
    score_payload,
)
from core.kaggle.failure_categories import (
    classify_binary_parse_status,
    classify_narrative_parse_status,
)
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
            latency_ms=duration_ms(call_started_at),
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
            latency_ms=duration_ms(call_started_at),
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
        outcome_kind, failure_category = classify_binary_parse_status(
            parsed_prediction.status,
        )
        logger.log_response_parse_failed(
            phase=phase,
            task_mode=task_mode,
            episode_id=episode_id,
            status=parsed_prediction.status.value,
            parse_status=parsed_prediction.status.value,
            outcome_kind=outcome_kind,
            failure_category=failure_category,
        )

    outcome_kind, failure_category = classify_binary_parse_status(
        parsed_prediction.status,
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
            latency_ms=duration_ms(call_started_at),
            failure_stage="episode_scoring",
        )

    logger.log_episode_scored(
        phase=phase,
        task_mode=task_mode,
        episode_id=episode_id,
        level="info" if parsed_prediction.status is ParseStatus.VALID else "warning",
        status=parsed_prediction.status.value,
        outcome_kind=outcome_kind,
        failure_category=failure_category,
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
            outcome_kind=outcome_kind,
            failure_category=failure_category,
            latency_ms=duration_ms(call_started_at),
            prediction=binary_prediction_payload(parsed_prediction),
            target=labels_payload(probe_targets),
            score=score_payload(score),
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
            latency_ms=duration_ms(call_started_at),
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
            latency_ms=duration_ms(call_started_at),
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
        outcome_kind, failure_category = classify_narrative_parse_status(
            parsed_result.status,
        )
        logger.log_response_parse_failed(
            phase=phase,
            task_mode=task_mode,
            episode_id=episode_id,
            status=parsed_result.status.value,
            parse_status=parsed_result.status.value,
            failure_detail=parsed_result.failure_detail,
            outcome_kind=outcome_kind,
            failure_category=failure_category,
        )

    outcome_kind, failure_category = classify_narrative_parse_status(
        parsed_result.status,
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
            latency_ms=duration_ms(call_started_at),
            failure_stage="episode_scoring",
        )

    logger.log_episode_scored(
        phase=phase,
        task_mode=task_mode,
        episode_id=episode_id,
        level="info" if parsed_result.status is NarrativeParseStatus.VALID else "warning",
        status=parsed_result.status.value,
        outcome_kind=outcome_kind,
        failure_category=failure_category,
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
            outcome_kind=outcome_kind,
            failure_category=failure_category,
            latency_ms=duration_ms(call_started_at),
            prediction=narrative_prediction_payload(parsed_result),
            target=labels_payload(probe_targets),
            score=score_payload(score),
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
    failure = record_operational_failure(
        exc,
        logger=logger,
        ledger=ledger,
        phase=phase,
        task_mode=task_mode,
        episode_id=episode_id,
        split=split,
        target=target,
        latency_ms=latency_ms,
        failure_stage=failure_stage,
        operational_failure_status=OPERATIONAL_FAILURE_STATUS,
    )
    return BinaryEpisodeExecution(
        parsed_prediction=ParsedPrediction.skipped_provider_failure(),
        score=failure.score,
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
    failure = record_operational_failure(
        exc,
        logger=logger,
        ledger=ledger,
        phase=phase,
        task_mode=task_mode,
        episode_id=episode_id,
        split=split,
        target=target,
        latency_ms=latency_ms,
        failure_stage=failure_stage,
        operational_failure_status=OPERATIONAL_FAILURE_STATUS,
    )
    return NarrativeEpisodeExecution(
        parsed_result=NarrativeParsedResult.skipped_provider_failure(),
        score=failure.score,
        status=OPERATIONAL_FAILURE_STATUS,
    )
