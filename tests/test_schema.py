from dataclasses import fields, replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from protocol import (
    Difficulty,
    InteractionLabel,
    ItemKind,
    Phase,
    RuleName,
    Split,
    TemplateId,
    Transition,
)
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
    items = (
        EpisodeItem(
            position=1,
            phase=Phase.PRE,
            kind=ItemKind.LABELED,
            q1=1,
            q2=2,
            label=InteractionLabel.REPEL,
        ),
        EpisodeItem(
            position=2,
            phase=Phase.PRE,
            kind=ItemKind.LABELED,
            q1=-1,
            q2=-2,
            label=InteractionLabel.REPEL,
        ),
        EpisodeItem(
            position=3,
            phase=Phase.POST,
            kind=ItemKind.LABELED,
            q1=1,
            q2=-1,
            label=InteractionLabel.REPEL,
        ),
        EpisodeItem(
            position=4,
            phase=Phase.POST,
            kind=ItemKind.LABELED,
            q1=2,
            q2=-2,
            label=InteractionLabel.REPEL,
        ),
        EpisodeItem(
            position=5,
            phase=Phase.POST,
            kind=ItemKind.LABELED,
            q1=-3,
            q2=3,
            label=InteractionLabel.REPEL,
        ),
        EpisodeItem(
            position=6,
            phase=Phase.POST,
            kind=ItemKind.PROBE,
            q1=1,
            q2=3,
        ),
        EpisodeItem(
            position=7,
            phase=Phase.POST,
            kind=ItemKind.PROBE,
            q1=-1,
            q2=3,
        ),
        EpisodeItem(
            position=8,
            phase=Phase.POST,
            kind=ItemKind.PROBE,
            q1=2,
            q2=3,
        ),
        EpisodeItem(
            position=9,
            phase=Phase.POST,
            kind=ItemKind.PROBE,
            q1=-2,
            q2=-3,
        ),
    )
    return Episode(
        episode_id="ife-r3-schema-t1",
        split=Split.DEV,
        difficulty=Difficulty.EASY,
        template_id=TemplateId.T1,
        rule_A=RuleName.R_STD,
        rule_B=RuleName.R_INV,
        transition=Transition.R_STD_TO_R_INV,
        pre_count=2,
        post_labeled_count=3,
        shift_after_position=2,
        contradiction_count_post=3,
        items=items,
        probe_targets=(
            InteractionLabel.ATTRACT,
            InteractionLabel.REPEL,
            InteractionLabel.ATTRACT,
            InteractionLabel.ATTRACT,
        ),
        probe_label_counts=(
            (InteractionLabel.ATTRACT, 3),
            (InteractionLabel.REPEL, 1),
        ),
        probe_sign_pattern_counts=(
            ("++", 2),
            ("--", 1),
            ("+-", 0),
            ("-+", 1),
        ),
        probe_metadata=(
            ProbeMetadata(
                position=6,
                is_disagreement_probe=True,
                old_rule_label=InteractionLabel.REPEL,
                new_rule_label=InteractionLabel.ATTRACT,
            ),
            ProbeMetadata(
                position=7,
                is_disagreement_probe=True,
                old_rule_label=InteractionLabel.ATTRACT,
                new_rule_label=InteractionLabel.REPEL,
            ),
            ProbeMetadata(
                position=8,
                is_disagreement_probe=True,
                old_rule_label=InteractionLabel.REPEL,
                new_rule_label=InteractionLabel.ATTRACT,
            ),
            ProbeMetadata(
                position=9,
                is_disagreement_probe=True,
                old_rule_label=InteractionLabel.REPEL,
                new_rule_label=InteractionLabel.ATTRACT,
            ),
        ),
        difficulty_version=DIFFICULTY_VERSION,
    )


def make_valid_t2_episode() -> Episode:
    items = (
        EpisodeItem(
            position=1,
            phase=Phase.PRE,
            kind=ItemKind.LABELED,
            q1=1,
            q2=2,
            label=InteractionLabel.REPEL,
        ),
        EpisodeItem(
            position=2,
            phase=Phase.PRE,
            kind=ItemKind.LABELED,
            q1=-1,
            q2=2,
            label=InteractionLabel.ATTRACT,
        ),
        EpisodeItem(
            position=3,
            phase=Phase.PRE,
            kind=ItemKind.LABELED,
            q1=-2,
            q2=-3,
            label=InteractionLabel.REPEL,
        ),
        EpisodeItem(
            position=4,
            phase=Phase.POST,
            kind=ItemKind.LABELED,
            q1=1,
            q2=-1,
            label=InteractionLabel.REPEL,
        ),
        EpisodeItem(
            position=5,
            phase=Phase.POST,
            kind=ItemKind.LABELED,
            q1=2,
            q2=-2,
            label=InteractionLabel.REPEL,
        ),
        EpisodeItem(
            position=6,
            phase=Phase.POST,
            kind=ItemKind.PROBE,
            q1=1,
            q2=3,
        ),
        EpisodeItem(
            position=7,
            phase=Phase.POST,
            kind=ItemKind.PROBE,
            q1=-1,
            q2=3,
        ),
        EpisodeItem(
            position=8,
            phase=Phase.POST,
            kind=ItemKind.PROBE,
            q1=-2,
            q2=-1,
        ),
        EpisodeItem(
            position=9,
            phase=Phase.POST,
            kind=ItemKind.PROBE,
            q1=2,
            q2=1,
        ),
    )
    return Episode(
        episode_id="ife-r3-schema-t2",
        split=Split.DEV,
        difficulty=Difficulty.MEDIUM,
        template_id=TemplateId.T2,
        rule_A=RuleName.R_STD,
        rule_B=RuleName.R_INV,
        transition=Transition.R_STD_TO_R_INV,
        pre_count=3,
        post_labeled_count=2,
        shift_after_position=3,
        contradiction_count_post=2,
        items=items,
        probe_targets=(
            InteractionLabel.ATTRACT,
            InteractionLabel.REPEL,
            InteractionLabel.ATTRACT,
            InteractionLabel.ATTRACT,
        ),
        probe_label_counts=(
            (InteractionLabel.ATTRACT, 3),
            (InteractionLabel.REPEL, 1),
        ),
        probe_sign_pattern_counts=(
            ("++", 2),
            ("--", 1),
            ("+-", 0),
            ("-+", 1),
        ),
        probe_metadata=(
            ProbeMetadata(
                position=6,
                is_disagreement_probe=True,
                old_rule_label=InteractionLabel.REPEL,
                new_rule_label=InteractionLabel.ATTRACT,
            ),
            ProbeMetadata(
                position=7,
                is_disagreement_probe=True,
                old_rule_label=InteractionLabel.ATTRACT,
                new_rule_label=InteractionLabel.REPEL,
            ),
            ProbeMetadata(
                position=8,
                is_disagreement_probe=True,
                old_rule_label=InteractionLabel.REPEL,
                new_rule_label=InteractionLabel.ATTRACT,
            ),
            ProbeMetadata(
                position=9,
                is_disagreement_probe=True,
                old_rule_label=InteractionLabel.REPEL,
                new_rule_label=InteractionLabel.ATTRACT,
            ),
        ),
        difficulty_version=DIFFICULTY_VERSION,
    )


class SchemaTestCase(unittest.TestCase):
    def test_schema_accepts_valid_t1_episode(self):
        episode = make_valid_t1_episode()

        self.assertEqual(episode.episode_id, "ife-r3-schema-t1")
        self.assertEqual(len(episode.items), 9)
        self.assertIs(episode.difficulty, Difficulty.EASY)

    def test_schema_accepts_valid_t2_episode(self):
        episode = make_valid_t2_episode()

        self.assertEqual(episode.episode_id, "ife-r3-schema-t2")
        self.assertEqual(len(episode.items), 9)
        self.assertIs(episode.difficulty, Difficulty.MEDIUM)

    def test_schema_fields_are_present(self):
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

        self.assertEqual(tuple(field.name for field in fields(Episode)), expected_fields)

        episode = make_valid_t1_episode()
        self.assertEqual(episode.spec_version, SPEC_VERSION)
        self.assertEqual(episode.generator_version, GENERATOR_VERSION)
        self.assertEqual(episode.template_set_version, TEMPLATE_SET_VERSION)
        self.assertEqual(episode.difficulty_version, DIFFICULTY_VERSION)
        self.assertIs(episode.split, Split.DEV)
        self.assertIs(episode.transition, Transition.R_STD_TO_R_INV)

    def test_schema_requires_first_five_items_to_be_labeled(self):
        episode = make_valid_t1_episode()
        invalid_items = list(episode.items)
        invalid_items[4] = EpisodeItem(
            position=5,
            phase=Phase.POST,
            kind=ItemKind.PROBE,
            q1=-3,
            q2=3,
        )

        with self.assertRaisesRegex(ValueError, "first 5 items"):
            replace(episode, items=tuple(invalid_items))

    def test_schema_requires_last_four_items_to_be_probes(self):
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

        with self.assertRaisesRegex(ValueError, "last 4 items"):
            replace(episode, items=tuple(invalid_items))

    def test_schema_requires_shift_after_position_to_match_pre_count(self):
        episode = make_valid_t1_episode()

        with self.assertRaisesRegex(ValueError, "shift_after_position must equal pre_count"):
            replace(episode, shift_after_position=3)

    def test_schema_rejects_duplicate_charge_pairs(self):
        episode = make_valid_t1_episode()
        invalid_items = list(episode.items)
        invalid_items[8] = EpisodeItem(
            position=9,
            phase=Phase.POST,
            kind=ItemKind.PROBE,
            q1=1,
            q2=2,
        )

        with self.assertRaisesRegex(ValueError, "must not repeat"):
            replace(episode, items=tuple(invalid_items))

    def test_schema_rejects_invalid_total_item_count(self):
        episode = make_valid_t1_episode()

        with self.assertRaisesRegex(ValueError, "exactly 9"):
            replace(episode, items=episode.items[:-1])

    def test_schema_requires_probe_targets_to_match_probe_items(self):
        episode = make_valid_t1_episode()

        with self.assertRaisesRegex(ValueError, "probe_targets must match rule_B labels"):
            replace(
                episode,
                probe_targets=(
                    InteractionLabel.REPEL,
                    InteractionLabel.REPEL,
                    InteractionLabel.ATTRACT,
                    InteractionLabel.ATTRACT,
                ),
            )

    def test_schema_requires_probe_metadata_to_match_probe_items(self):
        episode = make_valid_t1_episode()
        invalid_probe_metadata = list(episode.probe_metadata)
        invalid_probe_metadata[0] = ProbeMetadata(
            position=6,
            is_disagreement_probe=True,
            old_rule_label=InteractionLabel.ATTRACT,
            new_rule_label=InteractionLabel.ATTRACT,
        )

        with self.assertRaisesRegex(ValueError, "probe_metadata must match"):
            replace(episode, probe_metadata=tuple(invalid_probe_metadata))

    def test_schema_rejects_homogeneous_probe_targets(self):
        episode = make_valid_t1_episode()
        invalid_items = list(episode.items)
        invalid_items[5] = EpisodeItem(
            position=6,
            phase=Phase.POST,
            kind=ItemKind.PROBE,
            q1=1,
            q2=3,
        )
        invalid_items[6] = EpisodeItem(
            position=7,
            phase=Phase.POST,
            kind=ItemKind.PROBE,
            q1=2,
            q2=3,
        )
        invalid_items[7] = EpisodeItem(
            position=8,
            phase=Phase.POST,
            kind=ItemKind.PROBE,
            q1=-1,
            q2=-3,
        )
        invalid_items[8] = EpisodeItem(
            position=9,
            phase=Phase.POST,
            kind=ItemKind.PROBE,
            q1=-2,
            q2=-3,
        )
        invalid_probe_metadata = (
            ProbeMetadata(
                position=6,
                is_disagreement_probe=True,
                old_rule_label=InteractionLabel.REPEL,
                new_rule_label=InteractionLabel.ATTRACT,
            ),
            ProbeMetadata(
                position=7,
                is_disagreement_probe=True,
                old_rule_label=InteractionLabel.REPEL,
                new_rule_label=InteractionLabel.ATTRACT,
            ),
            ProbeMetadata(
                position=8,
                is_disagreement_probe=True,
                old_rule_label=InteractionLabel.REPEL,
                new_rule_label=InteractionLabel.ATTRACT,
            ),
            ProbeMetadata(
                position=9,
                is_disagreement_probe=True,
                old_rule_label=InteractionLabel.REPEL,
                new_rule_label=InteractionLabel.ATTRACT,
            ),
        )

        with self.assertRaisesRegex(ValueError, "at least two distinct labels"):
            replace(
                episode,
                items=tuple(invalid_items),
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
                probe_metadata=invalid_probe_metadata,
            )

    def test_schema_requires_correct_contradiction_count_post(self):
        episode = make_valid_t1_episode()

        with self.assertRaisesRegex(ValueError, "contradiction_count_post"):
            replace(episode, contradiction_count_post=2)

    def test_schema_requires_canonical_probe_label_counts(self):
        episode = make_valid_t1_episode()

        with self.assertRaisesRegex(ValueError, "probe_label_counts"):
            replace(
                episode,
                probe_label_counts=(
                    (InteractionLabel.ATTRACT, 2),
                    (InteractionLabel.REPEL, 2),
                ),
            )

    def test_schema_requires_canonical_probe_sign_pattern_counts(self):
        episode = make_valid_t1_episode()

        with self.assertRaisesRegex(ValueError, "probe_sign_pattern_counts"):
            replace(
                episode,
                probe_sign_pattern_counts=(
                    ("++", 1),
                    ("--", 1),
                    ("+-", 1),
                    ("-+", 1),
                ),
            )

    def test_schema_requires_difficulty_to_match_r3_rules(self):
        episode = make_valid_t1_episode()

        with self.assertRaisesRegex(ValueError, "difficulty must match"):
            replace(episode, difficulty=Difficulty.MEDIUM)

    def test_schema_rejects_hard_difficulty_for_valid_r3_fixtures(self):
        episode = make_valid_t2_episode()

        with self.assertRaisesRegex(ValueError, "difficulty must match"):
            replace(episode, difficulty=Difficulty.HARD)

    def test_schema_requires_difficulty_version_to_match_constant(self):
        episode = make_valid_t1_episode()

        with self.assertRaisesRegex(ValueError, "difficulty_version"):
            replace(episode, difficulty_version="R4")

    def test_fixture_based_difficulty_rules_match_clarified_r3_behavior(self):
        self.assertIs(make_valid_t1_episode().difficulty, Difficulty.EASY)
        self.assertIs(make_valid_t2_episode().difficulty, Difficulty.MEDIUM)
        self.assertNotEqual(make_valid_t1_episode().difficulty, Difficulty.HARD)
        self.assertNotEqual(make_valid_t2_episode().difficulty, Difficulty.HARD)


if __name__ == "__main__":
    unittest.main()
