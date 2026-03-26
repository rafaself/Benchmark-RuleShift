from core.metrics import MetricSummary, compute_metrics, compute_post_shift_probe_accuracy
from core.parser import (
    NarrativeAuditOutput,
    NarrativeParseStatus,
    NarrativeParsedResult,
    ParsedPrediction,
    ParseStatus,
)
from tasks.ruleshift_benchmark.protocol import InteractionLabel

ATTRACT = InteractionLabel.ATTRACT
REPEL = InteractionLabel.REPEL


def _valid_prediction(*labels: InteractionLabel) -> ParsedPrediction:
    return ParsedPrediction(labels=labels, status=ParseStatus.VALID)


def _invalid_prediction() -> ParsedPrediction:
    return ParsedPrediction(labels=(), status=ParseStatus.INVALID)


def _valid_narrative_result(*labels: InteractionLabel) -> NarrativeParsedResult:
    return NarrativeParsedResult(
        output=NarrativeAuditOutput(
            rule_before="rule A",
            shift_evidence="evidence",
            rule_after="rule B",
            final_decision=labels,
        ),
        status=NarrativeParseStatus.VALID,
    )


def _invalid_narrative_result() -> NarrativeParsedResult:
    return NarrativeParsedResult(
        output=None,
        status=NarrativeParseStatus.INVALID_FORMAT,
    )


def _skipped_narrative_result() -> NarrativeParsedResult:
    return NarrativeParsedResult.skipped_provider_failure()


# ---------------------------------------------------------------------------
# compute_post_shift_probe_accuracy tests
# ---------------------------------------------------------------------------


def test_post_shift_probe_accuracy_matches_hand_checked_fixture():
    predictions = (
        _valid_prediction(ATTRACT, REPEL, REPEL, ATTRACT),
        _valid_prediction(REPEL, ATTRACT, REPEL, ATTRACT),
    )
    targets = (
        (ATTRACT, REPEL, REPEL, ATTRACT),
        (ATTRACT, ATTRACT, REPEL, REPEL),
    )

    assert compute_post_shift_probe_accuracy(predictions, targets) == 0.75


def test_invalid_parses_contribute_zero_correct_probes():
    predictions = (
        _valid_prediction(ATTRACT, REPEL, REPEL, ATTRACT),
        _invalid_prediction(),
    )
    targets = (
        (ATTRACT, REPEL, REPEL, ATTRACT),
        (REPEL, ATTRACT, REPEL, ATTRACT),
    )

    assert compute_post_shift_probe_accuracy(predictions, targets) == 0.5


# ---------------------------------------------------------------------------
# compute_metrics tests
# ---------------------------------------------------------------------------


def test_compute_metrics_headline_is_binary_accuracy():
    binary_predictions = (_valid_prediction(ATTRACT, REPEL, REPEL, ATTRACT),)
    binary_targets = ((ATTRACT, REPEL, REPEL, ATTRACT),)
    narrative_results = (_valid_narrative_result(REPEL, REPEL, REPEL, REPEL),)

    summary = compute_metrics(
        binary_predictions=binary_predictions,
        binary_targets=binary_targets,
        narrative_results=narrative_results,
    )

    assert summary.post_shift_probe_accuracy == 1.0


def test_narrative_cannot_change_the_headline_binary_metric():
    binary_predictions = (_valid_prediction(ATTRACT, REPEL, REPEL, ATTRACT),)
    binary_targets = ((ATTRACT, REPEL, REPEL, ATTRACT),)
    narrative_results = (_invalid_narrative_result(),)

    summary = compute_metrics(
        binary_predictions=binary_predictions,
        binary_targets=binary_targets,
        narrative_results=narrative_results,
    )

    assert summary == MetricSummary(
        post_shift_probe_accuracy=1.0,
        binary_parse_valid_rate=1.0,
        narrative_schema_valid_rate=0.0,
        narrative_parse_failure_count=1,
    )


def test_compute_metrics_returns_correct_binary_parse_valid_rate():
    binary_predictions = (
        _valid_prediction(ATTRACT, REPEL, REPEL, ATTRACT),
        _invalid_prediction(),
    )
    binary_targets = (
        (ATTRACT, REPEL, REPEL, ATTRACT),
        (REPEL, ATTRACT, REPEL, ATTRACT),
    )
    narrative_results = (_valid_narrative_result(REPEL, ATTRACT, REPEL, ATTRACT),)

    summary = compute_metrics(
        binary_predictions=binary_predictions,
        binary_targets=binary_targets,
        narrative_results=narrative_results,
    )

    assert summary == MetricSummary(
        post_shift_probe_accuracy=0.5,
        binary_parse_valid_rate=0.5,
        narrative_schema_valid_rate=1.0,
        narrative_parse_failure_count=0,
    )


def test_compute_metrics_narrative_parse_failure_count():
    binary_predictions = (_valid_prediction(ATTRACT, REPEL, REPEL, ATTRACT),)
    binary_targets = ((ATTRACT, REPEL, REPEL, ATTRACT),)
    narrative_results = (
        _invalid_narrative_result(),
        NarrativeParsedResult(output=None, status=NarrativeParseStatus.MISSING_FIELD),
        _valid_narrative_result(REPEL, ATTRACT, REPEL, ATTRACT),
    )

    summary = compute_metrics(
        binary_predictions=binary_predictions,
        binary_targets=binary_targets,
        narrative_results=narrative_results,
    )

    assert summary.narrative_parse_failure_count == 2
    assert round(summary.narrative_schema_valid_rate, 6) == round(1 / 3, 6)


def test_compute_metrics_provider_failures_excluded_from_narrative_rate():
    binary_predictions = (_valid_prediction(ATTRACT, REPEL, REPEL, ATTRACT),)
    binary_targets = ((ATTRACT, REPEL, REPEL, ATTRACT),)
    # One provider failure (excluded from denominator), one format failure.
    narrative_results = (
        _skipped_narrative_result(),
        _invalid_narrative_result(),
    )

    summary = compute_metrics(
        binary_predictions=binary_predictions,
        binary_targets=binary_targets,
        narrative_results=narrative_results,
    )

    # Only 1 attempted (not skipped): 0 valid → rate = 0.0, failure_count = 1
    assert summary.narrative_schema_valid_rate == 0.0
    assert summary.narrative_parse_failure_count == 1


def test_compute_metrics_empty_narrative_results():
    binary_predictions = (_valid_prediction(ATTRACT, REPEL, REPEL, ATTRACT),)
    binary_targets = ((ATTRACT, REPEL, REPEL, ATTRACT),)

    summary = compute_metrics(
        binary_predictions=binary_predictions,
        binary_targets=binary_targets,
        narrative_results=(),
    )

    assert summary.post_shift_probe_accuracy == 1.0
    assert summary.binary_parse_valid_rate == 1.0
    assert summary.narrative_schema_valid_rate == 0.0
    assert summary.narrative_parse_failure_count == 0


def test_compute_metrics_provider_failures_excluded_from_binary_rate():
    from core.parser import ParsedPrediction, ParseStatus

    binary_predictions = (
        _valid_prediction(ATTRACT, REPEL, REPEL, ATTRACT),
        ParsedPrediction.skipped_provider_failure(),
    )
    binary_targets = (
        (ATTRACT, REPEL, REPEL, ATTRACT),
        (REPEL, ATTRACT, REPEL, REPEL),
    )

    summary = compute_metrics(
        binary_predictions=binary_predictions,
        binary_targets=binary_targets,
        narrative_results=(),
    )

    # Provider failure excluded: 1 attempted, 1 valid → rate = 1.0
    assert summary.binary_parse_valid_rate == 1.0
