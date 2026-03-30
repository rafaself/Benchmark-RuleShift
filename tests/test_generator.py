from unittest.mock import patch

import pytest

import tasks.ruleshift_benchmark.generator as rsb_generator
from tasks.ruleshift_benchmark.generator import generate_episode
from tasks.ruleshift_benchmark.protocol import (
    LABELED_ITEM_COUNT,
    Difficulty,
    InteractionLabel,
    ItemKind,
    Phase,
    Split,
    TemplateFamily,
    TemplateId,
    Transition,
)
from tasks.ruleshift_benchmark.rules import label


def test_same_seed_regenerates_the_same_episode():
    assert generate_episode(7) == generate_episode(7)


def test_small_seed_range_covers_all_templates_and_families():
    episodes = [generate_episode(seed) for seed in range(32)]

    assert {episode.template_id for episode in episodes} == {
        TemplateId.T1,
        TemplateId.T2,
        TemplateId.T3,
    }
    assert {episode.template_family for episode in episodes} == {
        TemplateFamily.CANONICAL,
        TemplateFamily.OBSERVATION_LOG,
        TemplateFamily.CASE_LEDGER,
    }


@pytest.mark.parametrize("seed", range(9))
def test_generated_episode_preserves_runtime_semantics(seed):
    episode = generate_episode(seed)
    labeled_items = episode.items[:LABELED_ITEM_COUNT]
    probe_items = episode.items[LABELED_ITEM_COUNT:]

    assert episode.split is Split.DEV
    assert episode.transition == Transition.from_rules(episode.rule_A, episode.rule_B)
    assert episode.rule_B is episode.rule_A.opposite
    assert episode.shift_after_position == episode.pre_count
    assert len(episode.items) == 9
    assert all(item.kind is ItemKind.LABELED for item in labeled_items)
    assert all(item.phase is Phase.PRE for item in labeled_items[: episode.pre_count])
    assert all(item.phase is Phase.POST for item in labeled_items[episode.pre_count :])
    assert all(item.kind is ItemKind.PROBE and item.phase is Phase.POST for item in probe_items)
    assert len({(item.q1, item.q2) for item in episode.items}) == len(episode.items)
    assert len(set(episode.probe_targets)) >= 2
    assert episode.contradiction_count_post >= 1
    assert episode.probe_sign_pattern_counts == (
        ("++", 1),
        ("--", 1),
        ("+-", 1),
        ("-+", 1),
    )
    assert episode.probe_targets != tuple(label(episode.rule_A, item.q1, item.q2) for item in probe_items)
    assert episode.probe_targets != tuple(label(episode.rule_B, item.q1, item.q2) for item in probe_items)


def test_generator_populates_canonical_metadata_fields():
    episode = generate_episode(3)

    assert isinstance(episode.difficulty, Difficulty)
    assert episode.probe_label_counts == (
        (InteractionLabel.ATTRACT, episode.probe_targets.count(InteractionLabel.ATTRACT)),
        (InteractionLabel.REPEL, episode.probe_targets.count(InteractionLabel.REPEL)),
    )


def test_generator_raises_when_bounded_attempts_are_exhausted():
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
                        return_value=invalid_candidate,
                    ) as sample_pairs:
                        with pytest.raises(RuntimeError, match="exhausted max_attempts"):
                            generate_episode(1, max_attempts=2)

    assert sample_pairs.call_count == 2


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
