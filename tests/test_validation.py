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
    DatasetDistributionSummary,
    DatasetValidationResult,
    EpisodeValidationResult,
    RegenerationCheck,
    ValidationIssue,
    normalize_episode_payload,
    validate_dataset,
    validate_episode,
)


_FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "validation_regression.json"


def _load_fixture() -> dict[str, object]:
    return json.loads(_FIXTURE_PATH.read_text())


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


def _parsed_payload(parsed_prediction: ParsedPrediction) -> dict[str, object]:
    return {
        "status": parsed_prediction.status.value,
        "labels": [label.value for label in parsed_prediction.labels],
    }


def _issue_codes(result: EpisodeValidationResult | DatasetValidationResult) -> set[str]:
    return {issue.code for issue in result.issues}


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
        payload = normalize_episode_payload(generate_episode(episode_fixture["seed"]))
        assert list(payload.keys()) == list(episode_fixture["payload"].keys())
        assert payload == episode_fixture["payload"]


def test_regression_fixture_blocks_render_drift():
    fixture = _load_fixture()

    for episode_fixture in fixture["episodes"]:
        episode = generate_episode(episode_fixture["seed"])
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
        episode = generate_episode(episode_fixture["seed"])
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
        episode = generate_episode(episode_fixture["seed"])
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
