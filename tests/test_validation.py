from __future__ import annotations

import json
from dataclasses import is_dataclass
from pathlib import Path

import pytest

from baselines import (
    last_evidence_baseline,
    never_update_baseline,
    physics_prior_baseline,
    random_baseline,
    template_position_baseline,
)
from generator import generate_episode
from metrics import MetricSummary, compute_metrics
from parser import ParsedPrediction, ParseStatus, parse_binary_output, parse_narrative_output
from protocol import LABELED_ITEM_COUNT, InteractionLabel
from render import render_binary_prompt, render_narrative_prompt
from rules import label
from schema import Episode, EpisodeItem, ProbeMetadata
from validate import (
    BaselineAccuracySummary,
    BenchmarkValidityReport,
    DatasetDistributionSummary,
    DatasetValidationResult,
    EpisodeValidationResult,
    RegenerationCheck,
    SplitBaselineAccuracySummary,
    ValidationIssue,
    normalize_episode_payload,
    run_benchmark_validity_report,
    serialize_benchmark_validity_report,
    validate_benchmark_validity,
    validate_dataset,
    validate_episode,
)


_FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "validation_regression.json"
_VALIDITY_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "release_r13_validity_report.json"
)


def _load_fixture() -> dict[str, object]:
    return json.loads(_FIXTURE_PATH.read_text())


def _load_validity_fixture() -> dict[str, object]:
    return json.loads(_VALIDITY_FIXTURE_PATH.read_text())


def _episode_from_payload(payload: dict[str, object]) -> Episode:
    normalized_payload = dict(payload)
    normalized_payload["items"] = tuple(
        EpisodeItem(**item_payload) for item_payload in normalized_payload["items"]
    )
    normalized_payload["probe_targets"] = tuple(normalized_payload["probe_targets"])
    normalized_payload["probe_label_counts"] = tuple(
        tuple(pair) for pair in normalized_payload["probe_label_counts"]
    )
    normalized_payload["probe_sign_pattern_counts"] = tuple(
        tuple(pair) for pair in normalized_payload["probe_sign_pattern_counts"]
    )
    normalized_payload["probe_metadata"] = tuple(
        ProbeMetadata(**metadata_payload)
        for metadata_payload in normalized_payload["probe_metadata"]
    )
    return Episode(**normalized_payload)


def _regenerate_fixture_episode(episode_fixture: dict[str, object]) -> Episode:
    payload = episode_fixture["payload"]
    return generate_episode(episode_fixture["seed"], split=payload["split"])


def _parsed_payload(parsed_prediction: ParsedPrediction) -> dict[str, object]:
    return {
        "status": parsed_prediction.status.value,
        "labels": [label.value for label in parsed_prediction.labels],
    }


def _issue_codes(result: EpisodeValidationResult | DatasetValidationResult) -> set[str]:
    return {issue.code for issue in result.issues}


def _synthetic_split_summary(
    split_name: str,
    accuracy: float,
    *,
    template_scores: tuple[tuple[str, float], ...],
    difficulty_scores: tuple[tuple[str, float], ...],
) -> SplitBaselineAccuracySummary:
    return SplitBaselineAccuracySummary(
        split_name=split_name,
        accuracy=accuracy,
        by_template=template_scores,
        by_difficulty=difficulty_scores,
    )


def _synthetic_baseline_summary(
    baseline_name: str,
    overall_accuracy: float,
    split_scores: tuple[SplitBaselineAccuracySummary, ...],
    *,
    template_scores: tuple[tuple[str, float], ...] = (("T1", 0.5), ("T2", 0.5)),
    difficulty_scores: tuple[tuple[str, float], ...] = (
        ("easy", 0.5),
        ("medium", 0.5),
    ),
) -> BaselineAccuracySummary:
    return BaselineAccuracySummary(
        baseline_name=baseline_name,
        overall_accuracy=overall_accuracy,
        by_split=split_scores,
        by_template=template_scores,
        by_difficulty=difficulty_scores,
    )


def _synthetic_report(
    baseline_summaries: tuple[BaselineAccuracySummary, ...],
) -> BenchmarkValidityReport:
    return BenchmarkValidityReport(
        release_id="R13",
        random_baseline_seed=11,
        report_splits=("dev", "public_leaderboard", "private_leaderboard"),
        gate_splits=("public_leaderboard", "private_leaderboard"),
        audit_split="private_leaderboard",
        split_episode_counts=(
            ("dev", 4),
            ("public_leaderboard", 4),
            ("private_leaderboard", 4),
        ),
        difficulty_labels_present=("easy", "medium"),
        difficulty_labels_missing=("hard",),
        critical_baselines=(
            "never_update",
            "last_evidence",
            "physics_prior",
            "template_position",
        ),
        baseline_summaries=baseline_summaries,
        checks=(),
        passed=False,
        comparison_summary="provisional",
        validity_note="provisional",
        limitations=("No emitted hard episodes in supplied set; hard slice omitted.",),
    )


def test_validate_episode_accepts_frozen_valid_fixture():
    fixture = _load_fixture()

    for episode_fixture in fixture["episodes"]:
        episode = _episode_from_payload(episode_fixture["payload"])
        assert validate_episode(episode) == EpisodeValidationResult(
            episode_id=episode.episode_id,
            ok=True,
            issues=(),
            regeneration=RegenerationCheck(
                checked=True,
                passed=True,
                expected_seed=episode_fixture["seed"],
            ),
        )


@pytest.mark.parametrize(
    ("mutator", "expected_code"),
    (
        (
            lambda episode: (
                object.__setattr__(episode.items[8], "q1", episode.items[7].q1),
                object.__setattr__(episode.items[8], "q2", episode.items[7].q2),
            ),
            "duplicate_item_pairs",
        ),
        (
            lambda episode: object.__setattr__(
                episode,
                "shift_after_position",
                episode.pre_count + 1,
            ),
            "invalid_shift_boundary",
        ),
        (
            lambda episode: object.__setattr__(
                episode,
                "probe_targets",
                (
                    InteractionLabel.ATTRACT,
                    InteractionLabel.ATTRACT,
                    InteractionLabel.ATTRACT,
                    InteractionLabel.ATTRACT,
                ),
            ),
            "trivial_probe_block",
        ),
        (
            lambda episode: object.__setattr__(
                episode,
                "probe_targets",
                ("repel", "attract", "unknown", "attract"),
            ),
            "invalid_probe_targets",
        ),
    ),
)
def test_validate_episode_rejects_invalid_episode_with_specific_issue_codes(
    mutator,
    expected_code: str,
):
    fixture = _load_fixture()
    episode = _episode_from_payload(fixture["episodes"][0]["payload"])

    mutator(episode)
    result = validate_episode(episode)

    assert not result.ok
    assert expected_code in _issue_codes(result)


def test_validate_episode_is_deterministic_across_repeated_runs():
    episode = generate_episode(0)

    assert validate_episode(episode) == validate_episode(episode)


def test_same_seed_replays_to_same_episode_and_stays_valid():
    episode = generate_episode(7)
    result = validate_episode(episode, seed=7)

    assert generate_episode(7) == generate_episode(7)
    assert result.ok
    assert result.regeneration == RegenerationCheck(
        checked=True,
        passed=True,
        expected_seed=7,
    )


def test_validate_dataset_rejects_duplicate_episode_ids():
    episode_a = generate_episode(0)
    episode_b = generate_episode(1)
    object.__setattr__(episode_b, "episode_id", episode_a.episode_id)

    result = validate_dataset((episode_a, episode_b))

    assert not result.ok
    assert "duplicate_episode_id" in _issue_codes(result)


def test_validate_dataset_rejects_duplicate_episode_payloads():
    episode_a = generate_episode(0)
    episode_b = generate_episode(0)
    object.__setattr__(episode_b, "episode_id", "fixture-copy")

    result = validate_dataset((episode_a, episode_b))

    assert not result.ok
    assert "duplicate_episode_payload" in _issue_codes(result)


def test_validate_dataset_returns_deterministic_distribution_summary():
    fixture = _load_fixture()
    episodes = tuple(
        _episode_from_payload(episode_fixture["payload"])
        for episode_fixture in fixture["episodes"]
    )

    result = validate_dataset(episodes)

    assert result.summary == DatasetDistributionSummary(
        template_counts=(("T1", 1), ("T2", 1)),
        transition_counts=(("R_std_to_R_inv", 1), ("R_inv_to_R_std", 1)),
        probe_label_counts=(("attract", 4), ("repel", 4)),
        sign_pattern_counts=(("++", 2), ("--", 2), ("+-", 2), ("-+", 2)),
        version_values=(
            ("spec_version", ("v1",)),
            ("generator_version", ("R12",)),
            ("template_set_version", ("v1",)),
            ("difficulty_version", ("R12",)),
        ),
    )
    assert result.summary == validate_dataset(episodes).summary


def test_validate_episode_rejects_recency_and_persistence_collapsible_probe_blocks():
    fixture = _load_fixture()
    episode = _episode_from_payload(fixture["episodes"][0]["payload"])
    probe_items = episode.items[LABELED_ITEM_COUNT:]

    object.__setattr__(
        episode,
        "probe_targets",
        tuple(label(episode.rule_A, item.q1, item.q2) for item in probe_items),
    )
    persistence_result = validate_episode(episode)

    object.__setattr__(
        episode,
        "probe_targets",
        tuple(label(episode.rule_B, item.q1, item.q2) for item in probe_items),
    )
    recency_result = validate_episode(episode)

    assert "persistence_collapsible_probe_block" in _issue_codes(persistence_result)
    assert "recency_collapsible_probe_block" in _issue_codes(recency_result)


def test_validate_dataset_checks_regeneration_for_canonical_ids_and_skips_noncanonical_ids():
    canonical_episode = generate_episode(0)
    noncanonical_episode = generate_episode(1)
    object.__setattr__(noncanonical_episode, "episode_id", "fixture-1")

    result = validate_dataset((canonical_episode, noncanonical_episode))

    assert result.ok
    assert result.episode_results[0].regeneration == RegenerationCheck(
        checked=True,
        passed=True,
        expected_seed=0,
    )
    assert result.episode_results[1].regeneration == RegenerationCheck(
        checked=False,
        passed=None,
        expected_seed=None,
    )


def test_regression_fixture_blocks_schema_field_drift():
    fixture = _load_fixture()

    for episode_fixture in fixture["episodes"]:
        payload = normalize_episode_payload(_regenerate_fixture_episode(episode_fixture))
        assert list(payload.keys()) == list(episode_fixture["payload"].keys())
        assert payload == episode_fixture["payload"]


def test_regression_fixture_blocks_render_drift():
    fixture = _load_fixture()

    for episode_fixture in fixture["episodes"]:
        episode = _regenerate_fixture_episode(episode_fixture)
        assert render_binary_prompt(episode) == episode_fixture["render"]["binary"]
        assert render_narrative_prompt(episode) == episode_fixture["render"]["narrative"]


def test_regression_fixture_blocks_parser_drift():
    fixture = _load_fixture()

    for episode_fixture in fixture["episodes"]:
        parser_fixture = episode_fixture["parser"]
        assert _parsed_payload(
            parse_binary_output(parser_fixture["binary_text"])
        ) == parser_fixture["binary_expected"]
        assert _parsed_payload(
            parse_narrative_output(parser_fixture["narrative_text"])
        ) == parser_fixture["narrative_expected"]


def test_regression_fixture_blocks_metric_drift():
    fixture = _load_fixture()
    binary_predictions = []
    narrative_predictions = []
    targets = []
    for episode_fixture in fixture["episodes"]:
        episode = _regenerate_fixture_episode(episode_fixture)
        binary_predictions.append(
            parse_binary_output(episode_fixture["parser"]["binary_text"])
        )
        narrative_predictions.append(
            parse_narrative_output(episode_fixture["parser"]["narrative_text"])
        )
        targets.append(episode.probe_targets)

    summary = compute_metrics(
        binary_predictions=tuple(binary_predictions),
        binary_targets=tuple(targets),
        narrative_predictions=tuple(narrative_predictions),
        narrative_targets=tuple(targets),
    )

    assert summary == MetricSummary(**fixture["metrics"]["expected"])


def test_regression_fixture_blocks_baseline_drift():
    fixture = _load_fixture()
    random_seed = fixture["random_baseline_seed"]

    for episode_fixture in fixture["episodes"]:
        episode = _regenerate_fixture_episode(episode_fixture)
        assert [label.value for label in random_baseline(episode, seed=random_seed)] == episode_fixture["baselines"]["random"]
        assert [label.value for label in never_update_baseline(episode)] == episode_fixture["baselines"]["never_update"]
        assert [label.value for label in last_evidence_baseline(episode)] == episode_fixture["baselines"]["last_evidence"]
        assert [label.value for label in physics_prior_baseline(episode)] == episode_fixture["baselines"]["physics_prior"]
        assert [label.value for label in template_position_baseline(episode)] == episode_fixture["baselines"]["template_position"]


def test_validation_result_objects_are_stable_dataclasses():
    issue = ValidationIssue(code="duplicate_episode_id", message="duplicate")
    regeneration = RegenerationCheck(checked=True, passed=True, expected_seed=0)
    episode_result = EpisodeValidationResult(
        episode_id="ife-r12-0",
        ok=True,
        issues=(issue,),
        regeneration=regeneration,
    )
    summary = DatasetDistributionSummary(
        template_counts=(("T1", 1), ("T2", 1)),
        transition_counts=(("R_std_to_R_inv", 1), ("R_inv_to_R_std", 1)),
        probe_label_counts=(("attract", 4), ("repel", 4)),
        sign_pattern_counts=(("++", 2), ("--", 2), ("+-", 3), ("-+", 1)),
        version_values=(("spec_version", ("v1",)),),
    )
    dataset_result = DatasetValidationResult(
        ok=True,
        episode_results=(episode_result,),
        issues=(issue,),
        summary=summary,
    )

    assert is_dataclass(issue)
    assert is_dataclass(regeneration)
    assert is_dataclass(episode_result)
    assert is_dataclass(summary)
    assert is_dataclass(dataset_result)
    assert issue == ValidationIssue(code="duplicate_episode_id", message="duplicate")
    assert regeneration == RegenerationCheck(checked=True, passed=True, expected_seed=0)
    assert episode_result == EpisodeValidationResult(
        episode_id="ife-r12-0",
        ok=True,
        issues=(issue,),
        regeneration=regeneration,
    )
    assert "ValidationIssue(" in repr(issue)
    assert "DatasetValidationResult(" in repr(dataset_result)


def test_benchmark_validity_report_is_deterministic_for_frozen_splits():
    assert validate_benchmark_validity() == validate_benchmark_validity()
    assert run_benchmark_validity_report() == run_benchmark_validity_report()


def test_benchmark_validity_report_matches_current_frozen_fixture():
    report = validate_benchmark_validity()

    assert serialize_benchmark_validity_report(report) == _load_validity_fixture()


def test_benchmark_validity_report_honestly_limits_difficulty_reporting_to_emitted_labels():
    report = validate_benchmark_validity()
    template_position = next(
        summary
        for summary in report.baseline_summaries
        if summary.baseline_name == "template_position"
    )

    assert report.difficulty_labels_present == ("easy", "medium")
    assert report.difficulty_labels_missing == ("hard",)
    assert tuple(label for label, _ in template_position.by_difficulty) == (
        "easy",
        "medium",
    )
    assert "hard" not in {label for label, _ in template_position.by_difficulty}
    assert "Hard is still not emitted" in report.validity_note


def test_benchmark_validity_gate_fails_when_shortcut_baseline_remains_too_strong():
    split_scores = (
        _synthetic_split_summary(
            "dev",
            0.5,
            template_scores=(("T1", 0.5), ("T2", 0.5)),
            difficulty_scores=(("easy", 0.5), ("medium", 0.5)),
        ),
        _synthetic_split_summary(
            "public_leaderboard",
            0.5,
            template_scores=(("T1", 0.5), ("T2", 0.5)),
            difficulty_scores=(("easy", 0.5), ("medium", 0.5)),
        ),
        _synthetic_split_summary(
            "private_leaderboard",
            0.625,
            template_scores=(("T1", 0.625), ("T2", 0.625)),
            difficulty_scores=(("easy", 0.625), ("medium", 0.625)),
        ),
    )
    report = _synthetic_report(
        (
            _synthetic_baseline_summary("random", 0.5, split_scores),
            _synthetic_baseline_summary("never_update", 0.5, split_scores),
            _synthetic_baseline_summary("last_evidence", 0.625, split_scores),
            _synthetic_baseline_summary("physics_prior", 0.5, split_scores),
            _synthetic_baseline_summary("template_position", 0.5, split_scores),
        )
    )

    result = validate_benchmark_validity(report=report)
    check_map = {check.code: check for check in result.checks}

    assert not result.passed
    assert not check_map["last_evidence_bounded"].passed
    assert not check_map["no_dominant_trivial_heuristic"].passed


def test_benchmark_validity_gate_passes_for_handcrafted_repaired_report():
    random_splits = (
        _synthetic_split_summary(
            "dev",
            0.48,
            template_scores=(("T1", 0.5), ("T2", 0.46)),
            difficulty_scores=(("easy", 0.49), ("medium", 0.47)),
        ),
        _synthetic_split_summary(
            "public_leaderboard",
            0.5,
            template_scores=(("T1", 0.52), ("T2", 0.48)),
            difficulty_scores=(("easy", 0.51), ("medium", 0.49)),
        ),
        _synthetic_split_summary(
            "private_leaderboard",
            0.47,
            template_scores=(("T1", 0.53), ("T2", 0.41)),
            difficulty_scores=(("easy", 0.52), ("medium", 0.42)),
        ),
    )
    bounded_uniform_splits = (
        _synthetic_split_summary(
            "dev",
            0.48,
            template_scores=(("T1", 0.48), ("T2", 0.48)),
            difficulty_scores=(("easy", 0.48), ("medium", 0.48)),
        ),
        _synthetic_split_summary(
            "public_leaderboard",
            0.5,
            template_scores=(("T1", 0.5), ("T2", 0.5)),
            difficulty_scores=(("easy", 0.5), ("medium", 0.5)),
        ),
        _synthetic_split_summary(
            "private_leaderboard",
            0.52,
            template_scores=(("T1", 0.52), ("T2", 0.52)),
            difficulty_scores=(("easy", 0.52), ("medium", 0.52)),
        ),
    )
    separated_template_position_splits = (
        _synthetic_split_summary(
            "dev",
            0.49,
            template_scores=(("T1", 0.55), ("T2", 0.43)),
            difficulty_scores=(("easy", 0.56), ("medium", 0.42)),
        ),
        _synthetic_split_summary(
            "public_leaderboard",
            0.53,
            template_scores=(("T1", 0.58), ("T2", 0.48)),
            difficulty_scores=(("easy", 0.57), ("medium", 0.49)),
        ),
        _synthetic_split_summary(
            "private_leaderboard",
            0.54,
            template_scores=(("T1", 0.62), ("T2", 0.46)),
            difficulty_scores=(("easy", 0.64), ("medium", 0.44)),
        ),
    )
    report = _synthetic_report(
        (
            _synthetic_baseline_summary("random", 0.48333333333333334, random_splits),
            _synthetic_baseline_summary("never_update", 0.5, bounded_uniform_splits),
            _synthetic_baseline_summary("last_evidence", 0.5, bounded_uniform_splits),
            _synthetic_baseline_summary("physics_prior", 0.5, bounded_uniform_splits),
            _synthetic_baseline_summary(
                "template_position",
                0.52,
                separated_template_position_splits,
                template_scores=(("T1", 0.58), ("T2", 0.45)),
                difficulty_scores=(("easy", 0.59), ("medium", 0.45)),
            ),
        )
    )

    result = validate_benchmark_validity(report=report)

    assert result.passed
    assert all(check.passed for check in result.checks)
    assert "Current status: PASS." in result.validity_note
