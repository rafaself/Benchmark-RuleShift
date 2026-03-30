from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

from core.kaggle.episode_ledger import EpisodeResultLedgerWriter
from core.kaggle.failure_categories import (
    OUTCOME_KIND_OPERATIONAL_FAILURE,
    classify_operational_exception,
)
from core.kaggle.run_logging import EXCEPTIONS_LOG_FILENAME
from core.kaggle.run_logging import BenchmarkRunLogger
from core.parser import (
    NarrativeParsedResult,
    NarrativeParseStatus,
    ParseStatus,
    ParsedPrediction,
)
from tasks.ruleshift_benchmark.protocol import PROBE_COUNT


@dataclass(frozen=True, slots=True)
class OperationalFailureDetails:
    failure_category: str
    exception_ref: str
    score: tuple[int, int]


def record_operational_failure(
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
    operational_failure_status: str,
) -> OperationalFailureDetails:
    failure_category = classify_operational_exception(
        exc=exc,
        failure_stage=failure_stage,
    )
    detail = format_operational_failure_detail(failure_stage, exc)
    if failure_stage == "provider_call":
        logger.log_provider_call_failed(
            phase=phase,
            task_mode=task_mode,
            episode_id=episode_id,
            exception_type=type(exc).__name__,
            exception_message=str(exc),
            failure_stage=failure_stage,
            failure_category=failure_category,
            outcome_kind=OUTCOME_KIND_OPERATIONAL_FAILURE,
            detail=detail,
        )
    exception_record = logger.log_exception(
        exc,
        phase=phase,
        task_mode=task_mode,
        episode_id=episode_id,
        failure_stage=failure_stage,
        failure_category=failure_category,
        outcome_kind=OUTCOME_KIND_OPERATIONAL_FAILURE,
    )
    logger.log_response_parse_failed(
        phase=phase,
        task_mode=task_mode,
        episode_id=episode_id,
        status=operational_failure_status,
        level="error",
        parse_status=operational_failure_status,
        failure_stage=failure_stage,
        failure_category=failure_category,
        outcome_kind=OUTCOME_KIND_OPERATIONAL_FAILURE,
        detail=detail,
    )

    score = (0, PROBE_COUNT)
    logger.log_episode_scored(
        phase=phase,
        task_mode=task_mode,
        episode_id=episode_id,
        level="error",
        status=operational_failure_status,
        outcome_kind=OUTCOME_KIND_OPERATIONAL_FAILURE,
        failure_category=failure_category,
        num_correct=score[0],
        total=score[1],
        failure_stage=failure_stage,
    )
    exception_ref = f"{EXCEPTIONS_LOG_FILENAME}#{exception_record['timestamp']}"
    if ledger is not None:
        ledger.write_record(
            episode_id=episode_id,
            split=split,
            task_mode=task_mode,
            call_status="failed" if failure_stage == "provider_call" else "completed",
            parse_status=operational_failure_status,
            outcome_kind=OUTCOME_KIND_OPERATIONAL_FAILURE,
            failure_category=failure_category,
            latency_ms=latency_ms,
            prediction=None,
            target=labels_payload(target),
            score=score_payload(score),
            exception_ref=exception_ref,
        )
    return OperationalFailureDetails(
        failure_category=failure_category,
        exception_ref=exception_ref,
        score=score,
    )


def binary_prediction_payload(
    parsed_prediction: ParsedPrediction,
) -> list[str] | None:
    if parsed_prediction.status is not ParseStatus.VALID:
        return None
    return [label.value for label in parsed_prediction.labels]


def narrative_prediction_payload(
    parsed_result: NarrativeParsedResult,
) -> list[str] | None:
    if parsed_result.status is not NarrativeParseStatus.VALID:
        return None
    if parsed_result.output is None:
        return None
    return [label.value for label in parsed_result.output.final_decision]


def labels_payload(labels: tuple | None) -> list[str] | None:
    if labels is None:
        return None
    return [
        str(getattr(label, "value", label))
        for label in labels
    ]


def score_payload(score: tuple[int, int]) -> dict[str, int]:
    return {
        "num_correct": score[0],
        "total": score[1],
    }


def duration_ms(started_at: float) -> int:
    return max(0, int((monotonic() - started_at) * 1000))


def format_operational_failure_detail(
    failure_stage: str,
    exc: BaseException,
) -> str:
    return (
        f"Operational failure during {failure_stage}: "
        f"{type(exc).__name__}: {exc}"
    )
