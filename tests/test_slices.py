"""Tests for core.slices — error classification, slice data computation, and aggregation."""
from __future__ import annotations

import pytest

from core.slices import (
    SLICE_DIMENSIONS,
    ErrorType,
    EpisodeSliceData,
    SliceAccuracy,
    SliceReport,
    build_slice_report,
    classify_binary_error_type,
    compute_episode_slice_data,
)
from generator import generate_episode
from parser import (
    NarrativeParseStatus,
    NarrativeParsedResult,
    ParsedPrediction,
    ParseStatus,
)
from protocol import PROBE_COUNT, InteractionLabel

ATTRACT = InteractionLabel.ATTRACT
REPEL = InteractionLabel.REPEL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeMeta:
    """Duck-typed ProbeMetadata for classify_binary_error_type tests."""

    def __init__(
        self,
        old_rule_label: InteractionLabel,
        new_rule_label: InteractionLabel,
    ) -> None:
        self.old_rule_label = old_rule_label
        self.new_rule_label = new_rule_label


def _valid_prediction(*labels: InteractionLabel) -> ParsedPrediction:
    return ParsedPrediction(labels=labels, status=ParseStatus.VALID)


def _invalid_prediction() -> ParsedPrediction:
    return ParsedPrediction(labels=(), status=ParseStatus.INVALID)


def _skipped_prediction() -> ParsedPrediction:
    return ParsedPrediction.skipped_provider_failure()


def _invalid_narrative() -> NarrativeParsedResult:
    return NarrativeParsedResult(output=None, status=NarrativeParseStatus.INVALID_FORMAT)


def _skipped_narrative() -> NarrativeParsedResult:
    return NarrativeParsedResult.skipped_provider_failure()


def _make_meta_all_same(old: InteractionLabel, new: InteractionLabel) -> tuple:
    return tuple(_FakeMeta(old, new) for _ in range(PROBE_COUNT))


def _make_episode_slice(
    *,
    episode_id: str = "ep-1",
    template: str = "T1",
    difficulty: str = "easy",
    shift_position: str = "2",
    transition_type: str = "R_std_to_R_inv",
    error_type: ErrorType = ErrorType.UNKNOWN,
    correct_probes: int = 4,
    total_probes: int = PROBE_COUNT,
) -> EpisodeSliceData:
    return EpisodeSliceData(
        episode_id=episode_id,
        template=template,
        difficulty=difficulty,
        shift_position=shift_position,
        transition_type=transition_type,
        error_type=error_type,
        correct_probes=correct_probes,
        total_probes=total_probes,
    )


# ---------------------------------------------------------------------------
# classify_binary_error_type — UNKNOWN cases
# ---------------------------------------------------------------------------


def test_classify_unknown_when_prediction_is_invalid():
    pred = _invalid_prediction()
    targets = (ATTRACT, REPEL, ATTRACT, REPEL)
    meta = _make_meta_all_same(REPEL, ATTRACT)
    result = classify_binary_error_type(
        prediction=pred,
        targets=targets,
        probe_metadata=meta,
    )
    assert result is ErrorType.UNKNOWN


def test_classify_unknown_when_prediction_is_provider_failure():
    pred = _skipped_prediction()
    targets = (ATTRACT, REPEL, ATTRACT, REPEL)
    meta = _make_meta_all_same(REPEL, ATTRACT)
    result = classify_binary_error_type(
        prediction=pred,
        targets=targets,
        probe_metadata=meta,
    )
    assert result is ErrorType.UNKNOWN


def test_classify_unknown_when_all_probes_correct():
    targets = (ATTRACT, REPEL, ATTRACT, REPEL)
    pred = _valid_prediction(*targets)
    meta = _make_meta_all_same(REPEL, ATTRACT)
    result = classify_binary_error_type(
        prediction=pred,
        targets=targets,
        probe_metadata=meta,
    )
    assert result is ErrorType.UNKNOWN


def test_classify_unknown_when_wrong_probe_count():
    pred = ParsedPrediction(labels=(ATTRACT, REPEL), status=ParseStatus.VALID)
    targets = (ATTRACT, REPEL, ATTRACT, REPEL)
    meta = _make_meta_all_same(REPEL, ATTRACT)
    result = classify_binary_error_type(
        prediction=pred,
        targets=targets,
        probe_metadata=meta,
    )
    assert result is ErrorType.UNKNOWN


# ---------------------------------------------------------------------------
# classify_binary_error_type — INVALID_NARRATIVE
# ---------------------------------------------------------------------------


def test_classify_invalid_narrative_takes_priority_over_correct_prediction():
    """invalid_narrative must fire even when the binary prediction is correct."""
    targets = (ATTRACT, REPEL, ATTRACT, REPEL)
    pred = _valid_prediction(*targets)  # all correct
    meta = _make_meta_all_same(REPEL, ATTRACT)
    result = classify_binary_error_type(
        prediction=pred,
        targets=targets,
        probe_metadata=meta,
        narrative_result=_invalid_narrative(),
    )
    assert result is ErrorType.INVALID_NARRATIVE


def test_classify_invalid_narrative_takes_priority_over_old_rule_error():
    """invalid_narrative fires even when prediction would be old_rule_persistence."""
    targets = (ATTRACT, ATTRACT, ATTRACT, ATTRACT)
    pred = _valid_prediction(REPEL, REPEL, REPEL, REPEL)
    meta = _make_meta_all_same(old=REPEL, new=ATTRACT)
    result = classify_binary_error_type(
        prediction=pred,
        targets=targets,
        probe_metadata=meta,
        narrative_result=_invalid_narrative(),
    )
    assert result is ErrorType.INVALID_NARRATIVE


def test_classify_skipped_narrative_does_not_trigger_invalid_narrative():
    """SKIPPED_PROVIDER_FAILURE narrative must not be treated as invalid."""
    targets = (ATTRACT, ATTRACT, ATTRACT, ATTRACT)
    pred = _valid_prediction(REPEL, REPEL, REPEL, REPEL)
    meta = _make_meta_all_same(old=REPEL, new=ATTRACT)
    result = classify_binary_error_type(
        prediction=pred,
        targets=targets,
        probe_metadata=meta,
        narrative_result=_skipped_narrative(),
    )
    # Not INVALID_NARRATIVE because skip is excluded from error classification.
    assert result is ErrorType.OLD_RULE_PERSISTENCE


def test_classify_no_narrative_does_not_trigger_invalid_narrative():
    """No narrative result (None) must not trigger invalid_narrative."""
    targets = (ATTRACT, ATTRACT, ATTRACT, ATTRACT)
    pred = _valid_prediction(REPEL, REPEL, REPEL, REPEL)
    meta = _make_meta_all_same(old=REPEL, new=ATTRACT)
    result = classify_binary_error_type(
        prediction=pred,
        targets=targets,
        probe_metadata=meta,
        narrative_result=None,
    )
    assert result is ErrorType.OLD_RULE_PERSISTENCE


# ---------------------------------------------------------------------------
# classify_binary_error_type — OLD_RULE_PERSISTENCE
# ---------------------------------------------------------------------------


def test_classify_old_rule_persistence_all_errors_match_old_label():
    """Every error probe predicts the old-rule label."""
    targets = (ATTRACT, ATTRACT, ATTRACT, ATTRACT)
    # Predicting all REPEL is wrong; REPEL = old_rule_label
    pred = _valid_prediction(REPEL, REPEL, REPEL, REPEL)
    meta = _make_meta_all_same(old=REPEL, new=ATTRACT)
    result = classify_binary_error_type(
        prediction=pred,
        targets=targets,
        probe_metadata=meta,
    )
    assert result is ErrorType.OLD_RULE_PERSISTENCE


def test_classify_old_rule_persistence_partial_errors_all_match_old_label():
    """Only two probes are wrong; both match old_rule → still OLD_RULE_PERSISTENCE."""
    targets = (ATTRACT, ATTRACT, REPEL, REPEL)
    # probes 0,1 wrong: predicted REPEL where target is ATTRACT
    pred = _valid_prediction(REPEL, REPEL, REPEL, REPEL)
    meta = (
        _FakeMeta(old=REPEL, new=ATTRACT),
        _FakeMeta(old=REPEL, new=ATTRACT),
        _FakeMeta(old=ATTRACT, new=REPEL),
        _FakeMeta(old=ATTRACT, new=REPEL),
    )
    result = classify_binary_error_type(
        prediction=pred,
        targets=targets,
        probe_metadata=meta,
    )
    assert result is ErrorType.OLD_RULE_PERSISTENCE


# ---------------------------------------------------------------------------
# classify_binary_error_type — RECENCY_OVERWEIGHT
# ---------------------------------------------------------------------------


def test_classify_recency_overweight_all_errors_match_new_label():
    """Every error probe predicts the new-rule label (which disagrees with target)."""
    # Target is REPEL due to a local sign exception; new_rule=ATTRACT
    targets = (REPEL, REPEL, REPEL, REPEL)
    pred = _valid_prediction(ATTRACT, ATTRACT, ATTRACT, ATTRACT)
    # old_rule=REPEL (same as target), new_rule=ATTRACT (≠ target)
    meta = _make_meta_all_same(old=REPEL, new=ATTRACT)
    result = classify_binary_error_type(
        prediction=pred,
        targets=targets,
        probe_metadata=meta,
    )
    assert result is ErrorType.RECENCY_OVERWEIGHT


# ---------------------------------------------------------------------------
# classify_binary_error_type — PREMATURE_SWITCH
# ---------------------------------------------------------------------------


def test_classify_premature_switch_mixed_old_and_new_errors():
    """One probe error matches old rule, one matches new rule → PREMATURE_SWITCH."""
    targets = (ATTRACT, ATTRACT, ATTRACT, ATTRACT)
    # probe 0 wrong: predicted REPEL = old_rule_label
    # probe 1 wrong: predicted REPEL = new_rule_label (old=ATTRACT for this probe)
    # probes 2,3 correct
    pred = _valid_prediction(REPEL, REPEL, ATTRACT, ATTRACT)
    meta = (
        _FakeMeta(old=REPEL, new=ATTRACT),   # probe 0 error = old_rule
        _FakeMeta(old=ATTRACT, new=REPEL),   # probe 1 error = new_rule
        _FakeMeta(old=REPEL, new=ATTRACT),
        _FakeMeta(old=REPEL, new=ATTRACT),
    )
    result = classify_binary_error_type(
        prediction=pred,
        targets=targets,
        probe_metadata=meta,
    )
    assert result is ErrorType.PREMATURE_SWITCH


# ---------------------------------------------------------------------------
# SLICE_DIMENSIONS constant
# ---------------------------------------------------------------------------


def test_slice_dimensions_has_five_canonical_entries():
    assert SLICE_DIMENSIONS == (
        "template",
        "difficulty",
        "shift_position",
        "transition_type",
        "error_type",
    )


# ---------------------------------------------------------------------------
# compute_episode_slice_data
# ---------------------------------------------------------------------------


def test_compute_episode_slice_data_extracts_episode_metadata():
    """Metadata fields are copied directly from Episode attributes."""
    episode = generate_episode(0)
    pred = _valid_prediction(*episode.probe_targets)

    result = compute_episode_slice_data(episode=episode, prediction=pred)

    assert result.episode_id == episode.episode_id
    assert result.template == episode.template_id.value
    assert result.difficulty == episode.difficulty.value
    assert result.shift_position == str(episode.shift_after_position)
    assert result.transition_type == episode.transition.value
    assert result.total_probes == PROBE_COUNT


def test_compute_episode_slice_data_perfect_prediction_is_all_correct():
    episode = generate_episode(1)
    pred = _valid_prediction(*episode.probe_targets)

    result = compute_episode_slice_data(episode=episode, prediction=pred)

    assert result.correct_probes == PROBE_COUNT
    assert result.error_type is ErrorType.UNKNOWN  # all correct → unknown


def test_compute_episode_slice_data_invalid_prediction_gives_zero_correct():
    episode = generate_episode(2)
    pred = _invalid_prediction()

    result = compute_episode_slice_data(episode=episode, prediction=pred)

    assert result.correct_probes == 0
    assert result.error_type is ErrorType.UNKNOWN  # parse failure → unknown


def test_compute_episode_slice_data_passes_narrative_result_to_classifier():
    """An invalid narrative result must produce error_type = INVALID_NARRATIVE."""
    episode = generate_episode(3)
    # Use perfect prediction; error type should still be INVALID_NARRATIVE
    pred = _valid_prediction(*episode.probe_targets)

    result = compute_episode_slice_data(
        episode=episode,
        prediction=pred,
        narrative_result=_invalid_narrative(),
    )

    assert result.error_type is ErrorType.INVALID_NARRATIVE


# ---------------------------------------------------------------------------
# SliceAccuracy
# ---------------------------------------------------------------------------


def test_slice_accuracy_property():
    acc = SliceAccuracy(episode_count=4, correct_probes=10, total_probes=16)
    assert acc.accuracy == pytest.approx(10 / 16)


def test_slice_accuracy_zero_total_probes_returns_zero():
    acc = SliceAccuracy(episode_count=0, correct_probes=0, total_probes=0)
    assert acc.accuracy == 0.0


def test_slice_accuracy_to_dict_has_accuracy_key():
    acc = SliceAccuracy(episode_count=2, correct_probes=6, total_probes=8)
    d = acc.to_dict()
    assert d["episode_count"] == 2
    assert d["correct_probes"] == 6
    assert d["total_probes"] == 8
    assert d["accuracy"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# build_slice_report
# ---------------------------------------------------------------------------


def test_build_slice_report_empty_returns_empty_groups():
    report = build_slice_report([])
    assert report.template == ()
    assert report.difficulty == ()
    assert report.shift_position == ()
    assert report.transition_type == ()
    # error_type always has all 5 keys even when empty
    error_dict = dict(report.error_type)
    assert set(error_dict.keys()) == {et.value for et in ErrorType}
    assert all(v == 0 for v in error_dict.values())


def test_build_slice_report_aggregates_correct_probes_by_template():
    slices = [
        _make_episode_slice(template="T1", correct_probes=3),
        _make_episode_slice(template="T1", correct_probes=4),
        _make_episode_slice(template="T2", correct_probes=2),
    ]
    report = build_slice_report(slices)
    template_dict = dict(report.template)

    assert "T1" in template_dict
    assert "T2" in template_dict
    assert template_dict["T1"].episode_count == 2
    assert template_dict["T1"].correct_probes == 7
    assert template_dict["T1"].total_probes == 8
    assert template_dict["T2"].episode_count == 1
    assert template_dict["T2"].correct_probes == 2


def test_build_slice_report_aggregates_error_type_counts():
    """Only episodes with correct_probes < total_probes are counted as failures."""
    slices = [
        _make_episode_slice(error_type=ErrorType.OLD_RULE_PERSISTENCE, correct_probes=0),
        _make_episode_slice(error_type=ErrorType.OLD_RULE_PERSISTENCE, correct_probes=0),
        _make_episode_slice(error_type=ErrorType.PREMATURE_SWITCH, correct_probes=2),
        _make_episode_slice(error_type=ErrorType.UNKNOWN, correct_probes=4),  # all correct → not counted
    ]
    report = build_slice_report(slices)
    error_dict = dict(report.error_type)

    assert error_dict[ErrorType.OLD_RULE_PERSISTENCE.value] == 2
    assert error_dict[ErrorType.PREMATURE_SWITCH.value] == 1
    assert error_dict[ErrorType.UNKNOWN.value] == 0  # all-correct episode not counted


def test_build_slice_report_template_order_is_canonical():
    """T1 must appear before T2 in the output regardless of insertion order."""
    slices = [
        _make_episode_slice(template="T2", correct_probes=2),
        _make_episode_slice(template="T1", correct_probes=4),
    ]
    report = build_slice_report(slices)
    keys = [k for k, _ in report.template]
    assert keys == ["T1", "T2"]


def test_build_slice_report_difficulty_order_is_canonical():
    slices = [
        _make_episode_slice(difficulty="medium"),
        _make_episode_slice(difficulty="easy"),
    ]
    report = build_slice_report(slices)
    keys = [k for k, _ in report.difficulty]
    assert keys == ["easy", "medium"]


def test_build_slice_report_to_dict_has_all_five_dimensions():
    slices = [_make_episode_slice()]
    report = build_slice_report(slices)
    d = report.to_dict()
    for dim in SLICE_DIMENSIONS:
        assert dim in d, f"SliceReport.to_dict() missing dimension {dim!r}"


def test_build_slice_report_transition_type_order_is_canonical():
    slices = [
        _make_episode_slice(transition_type="R_inv_to_R_std"),
        _make_episode_slice(transition_type="R_std_to_R_inv"),
    ]
    report = build_slice_report(slices)
    keys = [k for k, _ in report.transition_type]
    assert keys == ["R_std_to_R_inv", "R_inv_to_R_std"]


def test_build_slice_report_shift_position_sorted_alphabetically_when_no_order():
    slices = [
        _make_episode_slice(shift_position="3"),
        _make_episode_slice(shift_position="2"),
    ]
    report = build_slice_report(slices)
    keys = [k for k, _ in report.shift_position]
    assert keys == ["2", "3"]  # sorted ascending as strings


def test_build_slice_report_is_frozen():
    report = build_slice_report([])
    with pytest.raises((AttributeError, TypeError)):
        report.template = ()  # type: ignore[misc]
