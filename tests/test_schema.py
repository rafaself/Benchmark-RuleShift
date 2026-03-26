from dataclasses import fields, replace

import pytest

from generator import generate_episode
from protocol import (
    Difficulty,
    InteractionLabel,
    ItemKind,
    Phase,
    RuleName,
    Split,
    TemplateFamily,
    TemplateId,
    Transition,
)
from rules import label
from schema import (
    DIFFICULTY_VERSION,
    GENERATOR_VERSION,
    SPEC_VERSION,
    TEMPLATE_SET_VERSION,
    Episode,
    EpisodeItem,
    ProbeMetadata,
)


def make_valid_t1_episode() -> Episode:
    episode = generate_episode(0)
    assert episode.template_id is TemplateId.T1
    assert episode.difficulty is Difficulty.EASY
    return episode


def make_valid_t2_episode() -> Episode:
    episode = generate_episode(10)
    assert episode.template_id is TemplateId.T2
    assert episode.difficulty is Difficulty.MEDIUM
    return episode


def test_schema_accepts_valid_t1_episode():
    episode = make_valid_t1_episode()

    assert episode.template_id is TemplateId.T1
    assert len(episode.items) == 9
    assert episode.difficulty is Difficulty.EASY


def test_schema_accepts_valid_t2_episode():
    episode = make_valid_t2_episode()

    assert episode.template_id is TemplateId.T2
    assert len(episode.items) == 9
    assert episode.difficulty is Difficulty.MEDIUM


def test_schema_fields_are_present():
    expected_fields = (
        "episode_id",
        "split",
        "difficulty",
        "template_id",
        "template_family",
        "rule_A",
        "rule_B",
        "transition",
        "pre_count",
        "post_labeled_count",
        "shift_after_position",
        "contradiction_count_post",
        "difficulty_profile_id",
        "difficulty_factors",
        "items",
        "probe_targets",
        "probe_label_counts",
        "probe_sign_pattern_counts",
        "probe_metadata",
        "difficulty_version",
        "spec_version",
        "generator_version",
        "template_set_version",
    )

    assert tuple(field.name for field in fields(Episode)) == expected_fields

    episode = make_valid_t1_episode()
    assert episode.spec_version == SPEC_VERSION
    assert episode.generator_version == GENERATOR_VERSION
    assert episode.template_set_version == TEMPLATE_SET_VERSION
    assert episode.difficulty_version == DIFFICULTY_VERSION
    assert episode.split is Split.DEV
    assert episode.transition is Transition.from_rules(episode.rule_A, episode.rule_B)


def test_schema_requires_first_five_items_to_be_labeled():
    episode = make_valid_t1_episode()
    invalid_items = list(episode.items)
    invalid_items[4] = EpisodeItem(
        position=5,
        phase=Phase.POST,
        kind=ItemKind.PROBE,
        q1=-3,
        q2=3,
    )

    with pytest.raises(ValueError, match="first 5 items"):
        replace(episode, items=tuple(invalid_items))


def test_schema_requires_last_four_items_to_be_probes():
    episode = make_valid_t1_episode()
    invalid_items = list(episode.items)
    invalid_items[5] = EpisodeItem(
        position=6,
        phase=Phase.POST,
        kind=ItemKind.LABELED,
        q1=1,
        q2=3,
        label=InteractionLabel.ATTRACT,
    )

    with pytest.raises(ValueError, match="last 4 items"):
        replace(episode, items=tuple(invalid_items))


def test_schema_requires_shift_after_position_to_match_pre_count():
    episode = make_valid_t1_episode()

    with pytest.raises(ValueError, match="shift_after_position must equal pre_count"):
        replace(episode, shift_after_position=3)


def test_schema_rejects_duplicate_charge_pairs():
    episode = make_valid_t1_episode()
    invalid_items = list(episode.items)
    duplicate_q1 = episode.items[0].q1
    duplicate_q2 = episode.items[0].q2
    invalid_items[8] = EpisodeItem(
        position=9,
        phase=Phase.POST,
        kind=ItemKind.PROBE,
        q1=duplicate_q1,
        q2=duplicate_q2,
    )
    invalid_probe_metadata = list(episode.probe_metadata)
    invalid_probe_metadata[3] = ProbeMetadata(
        position=9,
        is_disagreement_probe=True,
        old_rule_label=label(episode.rule_A, duplicate_q1, duplicate_q2),
        new_rule_label=label(episode.rule_B, duplicate_q1, duplicate_q2),
    )

    with pytest.raises(ValueError, match="must not repeat"):
        replace(
            episode,
            items=tuple(invalid_items),
            probe_metadata=tuple(invalid_probe_metadata),
        )


def test_schema_rejects_invalid_total_item_count():
    episode = make_valid_t1_episode()

    with pytest.raises(ValueError, match="exactly 9"):
        replace(episode, items=episode.items[:-1])


def test_schema_requires_probe_targets_to_match_probe_items():
    episode = make_valid_t1_episode()

    with pytest.raises(ValueError, match="probe_targets must match slice-local derived labels"):
        replace(
            episode,
            probe_targets=(
                InteractionLabel.REPEL,
                InteractionLabel.REPEL,
                InteractionLabel.ATTRACT,
                InteractionLabel.ATTRACT,
            ),
        )


def test_schema_requires_probe_metadata_to_match_probe_items():
    episode = make_valid_t1_episode()
    invalid_probe_metadata = list(episode.probe_metadata)
    invalid_probe_metadata[0] = ProbeMetadata(
        position=6,
        is_disagreement_probe=True,
        old_rule_label=InteractionLabel.ATTRACT,
        new_rule_label=InteractionLabel.ATTRACT,
    )

    with pytest.raises(ValueError, match="probe_metadata must match"):
        replace(episode, probe_metadata=tuple(invalid_probe_metadata))


def test_schema_rejects_homogeneous_probe_targets():
    episode = make_valid_t1_episode()
    with pytest.raises(ValueError, match="slice-local derived labels"):
        replace(
            episode,
            probe_targets=(
                InteractionLabel.ATTRACT,
                InteractionLabel.ATTRACT,
                InteractionLabel.ATTRACT,
                InteractionLabel.ATTRACT,
            ),
            probe_label_counts=(
                (InteractionLabel.ATTRACT, 4),
                (InteractionLabel.REPEL, 0),
            ),
        )


def test_schema_requires_correct_contradiction_count_post():
    episode = make_valid_t1_episode()

    with pytest.raises(ValueError, match="contradiction_count_post"):
        replace(episode, contradiction_count_post=2)


def test_schema_requires_canonical_probe_label_counts():
    episode = make_valid_t1_episode()

    with pytest.raises(ValueError, match="probe_label_counts"):
        replace(
            episode,
            probe_label_counts=(
                (InteractionLabel.ATTRACT, 1),
                (InteractionLabel.REPEL, 3),
            ),
        )


def test_schema_requires_canonical_probe_sign_pattern_counts():
    episode = make_valid_t1_episode()

    with pytest.raises(ValueError, match="probe_sign_pattern_counts"):
        replace(
            episode,
            probe_sign_pattern_counts=(
                ("++", 2),
                ("--", 0),
                ("+-", 1),
                ("-+", 1),
            ),
        )


def test_schema_requires_difficulty_to_match_r13_rules():
    episode = make_valid_t1_episode()

    with pytest.raises(ValueError, match="difficulty must match"):
        replace(episode, difficulty=Difficulty.MEDIUM)


def test_schema_rejects_hard_difficulty_for_valid_r13_fixtures():
    episode = make_valid_t2_episode()

    with pytest.raises(ValueError, match="difficulty must match"):
        replace(episode, difficulty=Difficulty.HARD)


def test_schema_requires_difficulty_version_to_match_constant():
    episode = make_valid_t1_episode()

    with pytest.raises(ValueError, match="difficulty_version"):
        replace(episode, difficulty_version="R4")


def test_fixture_based_difficulty_rules_match_clarified_r13_behavior():
    assert make_valid_t1_episode().difficulty is Difficulty.EASY
    assert make_valid_t2_episode().difficulty is Difficulty.MEDIUM
    assert make_valid_t1_episode().difficulty is not Difficulty.HARD
    assert make_valid_t2_episode().difficulty is not Difficulty.HARD
