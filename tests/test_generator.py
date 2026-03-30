from unittest.mock import patch

import pytest

import tasks.ruleshift_benchmark.generator as rsb_generator
from maintainer.baselines import last_evidence_baseline
from tasks.ruleshift_benchmark.generator import generate_episode
from tasks.ruleshift_benchmark.protocol import (
    LABELED_ITEM_COUNT,
    Difficulty,
    DifficultyProfileId,
    InteractionLabel,
    ItemKind,
    Phase,
    Split,
    TemplateFamily,
    TemplateId,
    Transition,
)
from tasks.ruleshift_benchmark.rules import label
from tasks.ruleshift_benchmark.schema import Episode


def _probe_sign_pattern(q1: int, q2: int) -> str:
    if q1 > 0 and q2 > 0:
        return "++"
    if q1 < 0 and q2 < 0:
        return "--"
    if q1 > 0 and q2 < 0:
        return "+-"
    return "-+"


def test_same_seed_regenerates_the_same_episode():
    assert generate_episode(7) == generate_episode(7)


def test_different_seeds_can_generate_different_episodes():
    episodes = {generate_episode(seed) for seed in range(6)}
    assert len(episodes) > 1


def test_all_templates_are_emitted():
    emitted_templates = {generate_episode(seed).template_id for seed in range(32)}
    assert emitted_templates == {TemplateId.T1, TemplateId.T2, TemplateId.T3}


def test_all_template_families_are_emitted():
    emitted_families = {generate_episode(seed).template_family for seed in range(32)}
    assert emitted_families == {
        TemplateFamily.CANONICAL,
        TemplateFamily.OBSERVATION_LOG,
        TemplateFamily.CASE_LEDGER,
    }


@pytest.mark.parametrize("seed", range(10))
def test_episode_length_is_always_nine(seed):
    assert len(generate_episode(seed).items) == 9


@pytest.mark.parametrize("seed", range(10))
def test_labeled_and_probe_boundaries_are_correct(seed):
    episode = generate_episode(seed)
    labeled_items = episode.items[:LABELED_ITEM_COUNT]
    probe_items = episode.items[LABELED_ITEM_COUNT:]

    assert all(item.kind is ItemKind.LABELED for item in labeled_items)
    assert all(item.kind is ItemKind.PROBE for item in probe_items)
    assert all(item.phase is Phase.PRE for item in labeled_items[: episode.pre_count])
    assert all(item.phase is Phase.POST for item in labeled_items[episode.pre_count :])
    assert all(item.phase is Phase.POST for item in probe_items)


@pytest.mark.parametrize("seed", range(10))
def test_shift_after_position_equals_pre_count(seed):
    episode = generate_episode(seed)
    assert episode.shift_after_position == episode.pre_count


@pytest.mark.parametrize("seed", range(10))
def test_rule_b_is_always_the_opposite_of_rule_a(seed):
    episode = generate_episode(seed)
    assert episode.rule_B is episode.rule_A.opposite


@pytest.mark.parametrize("seed", range(10))
def test_no_duplicate_q1_q2_pairs_exist_within_an_episode(seed):
    episode = generate_episode(seed)
    pairs = [(item.q1, item.q2) for item in episode.items]
    assert len(set(pairs)) == len(pairs)


def test_generator_populates_canonical_row_metadata():
    episode = generate_episode(3)
    probe_items = episode.items[LABELED_ITEM_COUNT:]
    target_matches_new_rule = sum(
        metadata.new_rule_label == target
        for metadata, target in zip(episode.probe_metadata, episode.probe_targets)
    )
    target_matches_old_rule = sum(
        metadata.old_rule_label == target
        for metadata, target in zip(episode.probe_metadata, episode.probe_targets)
    )

    assert episode.split is Split.DEV
    assert isinstance(episode.difficulty, Difficulty)
    assert episode.transition == Transition.from_rules(episode.rule_A, episode.rule_B)
    assert target_matches_new_rule == 2
    assert target_matches_old_rule == 2
    assert episode.probe_targets != tuple(
        label(episode.rule_A, item.q1, item.q2) for item in probe_items
    )
    assert episode.probe_targets != tuple(
        label(episode.rule_B, item.q1, item.q2) for item in probe_items
    )
    assert episode.probe_label_counts == (
        (
            InteractionLabel.ATTRACT,
            episode.probe_targets.count(InteractionLabel.ATTRACT),
        ),
        (
            InteractionLabel.REPEL,
            episode.probe_targets.count(InteractionLabel.REPEL),
        ),
    )


@pytest.mark.parametrize("seed", range(64))
def test_valid_generated_episodes_have_nontrivial_probe_blocks(seed):
    episode = generate_episode(seed)
    assert len(set(episode.probe_targets)) >= 2


@pytest.mark.parametrize("seed", range(64))
def test_valid_generated_episodes_have_post_shift_contradictions(seed):
    episode = generate_episode(seed)
    assert episode.contradiction_count_post >= 1


@pytest.mark.parametrize("seed", range(64))
def test_valid_generated_episodes_use_exactly_two_mixed_polarity_updated_sign_patterns(seed):
    episode = generate_episode(seed)
    updated_sign_patterns = {
        _probe_sign_pattern(item.q1, item.q2)
        for item in episode.items[episode.pre_count:LABELED_ITEM_COUNT]
    }

    assert len(updated_sign_patterns) == 2
    assert any(pattern in {"++", "--"} for pattern in updated_sign_patterns)
    assert any(pattern in {"+-", "-+"} for pattern in updated_sign_patterns)


@pytest.mark.parametrize("seed", range(64))
def test_valid_generated_episodes_cover_each_probe_sign_pattern_once(seed):
    episode = generate_episode(seed)

    assert episode.probe_sign_pattern_counts == (
        ("++", 1),
        ("--", 1),
        ("+-", 1),
        ("-+", 1),
    )


@pytest.mark.parametrize("seed", range(64))
def test_valid_generated_episodes_are_not_global_rule_probe_blocks(seed):
    episode = generate_episode(seed)
    probe_items = episode.items[LABELED_ITEM_COUNT:]

    assert episode.probe_targets != tuple(
        label(episode.rule_A, item.q1, item.q2) for item in probe_items
    )
    assert episode.probe_targets != tuple(
        label(episode.rule_B, item.q1, item.q2) for item in probe_items
    )


@pytest.mark.parametrize("seed", range(10))
def test_labeled_items_use_the_active_rule_engine_label(seed):
    episode = generate_episode(seed)
    for item in episode.items[:LABELED_ITEM_COUNT]:
        active_rule = episode.rule_A if item.position <= episode.pre_count else episode.rule_B
        assert item.label == label(active_rule, item.q1, item.q2)


def test_invalid_candidates_are_rejected_by_deterministic_resampling():
    invalid_candidate = (
        (1, 1),
        (-1, 1),
        (-2, -2),
        (2, -2),
        (1, 2),
        (2, 3),
        (-3, -2),
        (3, -1),
        (-2, 3),
    )
    valid_candidate = (
        (1, 1),
        (-1, 1),
        (-2, -2),
        (2, -2),
        (-1, -2),
        (2, 3),
        (-3, -2),
        (3, -1),
        (-2, 3),
    )

    with patch.object(rsb_generator, "_target_difficulty_for_seed", return_value=rsb_generator.Difficulty.HARD):
        with patch.object(rsb_generator, "_target_template_for_seed", return_value=TemplateId.T1):
            with patch.object(
                rsb_generator,
                "_target_template_family_for_seed",
                return_value=TemplateFamily.CANONICAL,
            ):
                with patch.object(
                    rsb_generator,
                    "_target_transition_for_seed",
                    return_value=Transition.R_STD_TO_R_INV,
                ):
                    with patch.object(
                        rsb_generator,
                        "_sample_pairs",
                        side_effect=(invalid_candidate, valid_candidate),
                    ) as sample_pairs:
                        episode = generate_episode(1)

    assert sample_pairs.call_count == 2
    assert tuple((item.q1, item.q2) for item in episode.items) == valid_candidate
    assert episode.contradiction_count_post >= 1
    assert episode.probe_sign_pattern_counts == (
        ("++", 1),
        ("--", 1),
        ("+-", 1),
        ("-+", 1),
    )


def test_difficulty_is_not_effectively_encoded_by_template_identity():
    episodes = tuple(generate_episode(seed) for seed in range(162))
    difficulty_to_templates: dict[Difficulty, set[TemplateId]] = {}
    for episode in episodes:
        difficulty_to_templates.setdefault(episode.difficulty, set()).add(episode.template_id)

    assert difficulty_to_templates == {
        Difficulty.EASY: {TemplateId.T1, TemplateId.T2, TemplateId.T3},
        Difficulty.MEDIUM: {TemplateId.T1, TemplateId.T2, TemplateId.T3},
        Difficulty.HARD: {TemplateId.T1, TemplateId.T2, TemplateId.T3},
    }


@pytest.mark.parametrize("seed", range(64))
def test_every_valid_episode_gets_exactly_one_difficulty_tier(seed):
    episode = generate_episode(seed)
    assert episode.difficulty is (
        Difficulty.EASY
        if seed % 3 == 0
        else Difficulty.MEDIUM
        if seed % 3 == 1
        else Difficulty.HARD
    )


@pytest.mark.parametrize("seed", range(24))
def test_difficulty_profile_is_intentional_and_seed_stable(seed):
    episode = generate_episode(seed)
    expected_profile = (
        DifficultyProfileId.EASY_ANCHORED
        if seed % 3 == 0
        else DifficultyProfileId.MEDIUM_BALANCED
        if seed % 3 == 1
        else DifficultyProfileId.HARD_INTERLEAVED
    )

    assert episode.difficulty_profile_id is expected_profile
    assert generate_episode(seed).difficulty_factors == episode.difficulty_factors


@pytest.mark.parametrize("seed", range(32))
def test_emitted_metadata_fields_are_consistent_with_episode_contents(seed):
    episode = generate_episode(seed)
    post_labeled_items = episode.items[episode.pre_count:LABELED_ITEM_COUNT]
    probe_items = episode.items[LABELED_ITEM_COUNT:]
    expected_contradictions = sum(
        label(episode.rule_A, item.q1, item.q2) != label(episode.rule_B, item.q1, item.q2)
        for item in post_labeled_items
    )
    expected_sign_pattern_counts = (
        ("++", sum(item.q1 > 0 and item.q2 > 0 for item in probe_items)),
        ("--", sum(item.q1 < 0 and item.q2 < 0 for item in probe_items)),
        ("+-", sum(item.q1 > 0 and item.q2 < 0 for item in probe_items)),
        ("-+", sum(item.q1 < 0 and item.q2 > 0 for item in probe_items)),
    )

    assert episode.contradiction_count_post == expected_contradictions
    assert episode.probe_label_counts == (
        (
            InteractionLabel.ATTRACT,
            episode.probe_targets.count(InteractionLabel.ATTRACT),
        ),
        (
            InteractionLabel.REPEL,
            episode.probe_targets.count(InteractionLabel.REPEL),
        ),
    )
    assert episode.probe_sign_pattern_counts == expected_sign_pattern_counts
    assert episode.probe_sign_pattern_counts == (
        ("++", 1),
        ("--", 1),
        ("+-", 1),
        ("-+", 1),
    )
    assert episode.difficulty_version == "R13"


def test_last_evidence_is_capped_on_representative_r12_sample():
    exact_match_count = 0
    total_probe_accuracy = 0.0

    for seed in range(64):
        episode = generate_episode(seed)
        prediction = last_evidence_baseline(episode)
        exact_match_count += int(prediction == episode.probe_targets)
        total_probe_accuracy += sum(
            label_value == target
            for label_value, target in zip(prediction, episode.probe_targets)
        ) / 4

    assert exact_match_count == 0
    assert total_probe_accuracy / 64 == 0.5
