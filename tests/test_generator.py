from dataclasses import fields
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import generator
from generator import generate_episode
from protocol import (
    LABELED_ITEM_COUNT,
    Difficulty,
    InteractionLabel,
    ItemKind,
    Phase,
    Split,
    TemplateId,
    Transition,
)
from rules import label
from schema import Episode


class GeneratorTestCase(unittest.TestCase):
    def test_same_seed_regenerates_the_same_episode(self):
        self.assertEqual(generate_episode(7), generate_episode(7))

    def test_different_seeds_can_generate_different_episodes(self):
        episodes = {generate_episode(seed) for seed in range(6)}

        self.assertGreater(len(episodes), 1)

    def test_only_t1_and_t2_are_emitted(self):
        emitted_templates = {generate_episode(seed).template_id for seed in range(32)}

        self.assertEqual(emitted_templates, {TemplateId.T1, TemplateId.T2})

    def test_episode_length_is_always_nine(self):
        for seed in range(10):
            with self.subTest(seed=seed):
                self.assertEqual(len(generate_episode(seed).items), 9)

    def test_labeled_and_probe_boundaries_are_correct(self):
        for seed in range(10):
            with self.subTest(seed=seed):
                episode = generate_episode(seed)
                labeled_items = episode.items[:LABELED_ITEM_COUNT]
                probe_items = episode.items[LABELED_ITEM_COUNT:]

                self.assertTrue(all(item.kind is ItemKind.LABELED for item in labeled_items))
                self.assertTrue(all(item.kind is ItemKind.PROBE for item in probe_items))
                self.assertTrue(
                    all(item.phase is Phase.PRE for item in labeled_items[: episode.pre_count])
                )
                self.assertTrue(
                    all(item.phase is Phase.POST for item in labeled_items[episode.pre_count :])
                )
                self.assertTrue(all(item.phase is Phase.POST for item in probe_items))

    def test_shift_after_position_equals_pre_count(self):
        for seed in range(10):
            with self.subTest(seed=seed):
                episode = generate_episode(seed)
                self.assertEqual(episode.shift_after_position, episode.pre_count)

    def test_rule_b_is_always_the_opposite_of_rule_a(self):
        for seed in range(10):
            with self.subTest(seed=seed):
                episode = generate_episode(seed)
                self.assertIs(episode.rule_B, episode.rule_A.opposite)

    def test_no_duplicate_q1_q2_pairs_exist_within_an_episode(self):
        for seed in range(10):
            with self.subTest(seed=seed):
                episode = generate_episode(seed)
                pairs = [(item.q1, item.q2) for item in episode.items]
                self.assertEqual(len(set(pairs)), len(pairs))

    def test_schema_fields_are_always_present(self):
        expected_fields = (
            "episode_id",
            "split",
            "difficulty",
            "template_id",
            "rule_A",
            "rule_B",
            "transition",
            "pre_count",
            "post_labeled_count",
            "shift_after_position",
            "contradiction_count_post",
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

        episode = generate_episode(3)

        self.assertIsInstance(episode, Episode)
        self.assertEqual(tuple(field.name for field in fields(Episode)), expected_fields)
        for field_name in expected_fields:
            with self.subTest(field_name=field_name):
                self.assertTrue(hasattr(episode, field_name))

    def test_generator_populates_canonical_row_metadata(self):
        episode = generate_episode(3)

        self.assertIs(episode.split, Split.DEV)
        self.assertIsInstance(episode.difficulty, Difficulty)
        self.assertEqual(
            episode.transition, Transition.from_rules(episode.rule_A, episode.rule_B)
        )
        self.assertEqual(
            tuple(metadata.new_rule_label for metadata in episode.probe_metadata),
            episode.probe_targets,
        )
        self.assertEqual(
            episode.probe_label_counts,
            (
                (
                    InteractionLabel.ATTRACT,
                    episode.probe_targets.count(InteractionLabel.ATTRACT),
                ),
                (
                    InteractionLabel.REPEL,
                    episode.probe_targets.count(InteractionLabel.REPEL),
                ),
            ),
        )

    def test_valid_generated_episodes_have_nontrivial_probe_blocks(self):
        for seed in range(64):
            with self.subTest(seed=seed):
                episode = generate_episode(seed)
                self.assertGreaterEqual(len(set(episode.probe_targets)), 2)

    def test_valid_generated_episodes_have_post_shift_contradictions(self):
        for seed in range(64):
            with self.subTest(seed=seed):
                episode = generate_episode(seed)
                self.assertGreaterEqual(episode.contradiction_count_post, 1)

    def test_labeled_items_use_the_active_rule_engine_label(self):
        for seed in range(10):
            with self.subTest(seed=seed):
                episode = generate_episode(seed)
                for item in episode.items[:LABELED_ITEM_COUNT]:
                    active_rule = (
                        episode.rule_A
                        if item.position <= episode.pre_count
                        else episode.rule_B
                    )
                    self.assertEqual(item.label, label(active_rule, item.q1, item.q2))

    def test_invalid_candidates_are_rejected_by_deterministic_resampling(self):
        invalid_candidate = (
            (1, 1),
            (1, -1),
            (-1, 1),
            (-2, -2),
            (2, -2),
            (3, 3),
            (-3, -3),
            (2, 2),
            (-1, -1),
        )
        valid_candidate = (
            (1, 1),
            (1, -1),
            (-1, 1),
            (-2, -2),
            (2, -2),
            (3, 3),
            (3, -3),
            (-3, 3),
            (-1, -1),
        )

        with patch.object(
            generator,
            "_sample_pairs",
            side_effect=(invalid_candidate, valid_candidate),
        ) as sample_pairs:
            episode = generate_episode(1)

        self.assertEqual(sample_pairs.call_count, 2)
        self.assertEqual(
            tuple((item.q1, item.q2) for item in episode.items),
            valid_candidate,
        )
        self.assertGreaterEqual(episode.contradiction_count_post, 1)
        self.assertGreaterEqual(len(set(episode.probe_targets)), 2)

    def test_every_valid_episode_gets_exactly_one_difficulty_tier(self):
        for seed in range(64):
            with self.subTest(seed=seed):
                episode = generate_episode(seed)
                is_mixed_probe_block = len(set(episode.probe_targets)) >= 2
                matches_easy = episode.template_id is TemplateId.T1 and is_mixed_probe_block
                matches_medium = not matches_easy
                matches_hard = False

                self.assertEqual(sum((matches_easy, matches_medium, matches_hard)), 1)
                if matches_easy:
                    self.assertIs(episode.difficulty, Difficulty.EASY)
                if matches_medium:
                    self.assertIs(episode.difficulty, Difficulty.MEDIUM)

    def test_emitted_metadata_fields_are_consistent_with_episode_contents(self):
        for seed in range(32):
            with self.subTest(seed=seed):
                episode = generate_episode(seed)
                post_labeled_items = episode.items[episode.pre_count:LABELED_ITEM_COUNT]
                probe_items = episode.items[LABELED_ITEM_COUNT:]
                expected_contradictions = sum(
                    label(episode.rule_A, item.q1, item.q2)
                    != label(episode.rule_B, item.q1, item.q2)
                    for item in post_labeled_items
                )
                expected_sign_pattern_counts = (
                    ("++", sum(item.q1 > 0 and item.q2 > 0 for item in probe_items)),
                    ("--", sum(item.q1 < 0 and item.q2 < 0 for item in probe_items)),
                    ("+-", sum(item.q1 > 0 and item.q2 < 0 for item in probe_items)),
                    ("-+", sum(item.q1 < 0 and item.q2 > 0 for item in probe_items)),
                )

                self.assertEqual(episode.contradiction_count_post, expected_contradictions)
                self.assertEqual(
                    episode.probe_label_counts,
                    (
                        (
                            InteractionLabel.ATTRACT,
                            episode.probe_targets.count(InteractionLabel.ATTRACT),
                        ),
                        (
                            InteractionLabel.REPEL,
                            episode.probe_targets.count(InteractionLabel.REPEL),
                        ),
                    ),
                )
                self.assertEqual(
                    episode.probe_sign_pattern_counts,
                    expected_sign_pattern_counts,
                )
                self.assertEqual(episode.difficulty_version, "R3")


if __name__ == "__main__":
    unittest.main()
