from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from protocol import (
    CASE_SPACE,
    CHARGES,
    DIFFICULTIES,
    EPISODE_LENGTH,
    ITEM_KINDS,
    PHASES,
    PROBE_COUNT,
    RULES,
    SPLITS,
    TEMPLATE_IDS,
    TEMPLATES,
    Difficulty,
    InteractionLabel,
    ItemKind,
    Phase,
    RuleName,
    Split,
    TemplateSpec,
    TemplateId,
    Transition,
    parse_difficulty,
    parse_item_kind,
    parse_label,
    parse_phase,
    parse_rule,
    parse_split,
    parse_template_id,
    parse_transition,
)


class ProtocolTestCase(unittest.TestCase):
    def test_rule_parser_accepts_enums_and_canonical_strings(self):
        self.assertIs(parse_rule(RuleName.R_STD), RuleName.R_STD)
        self.assertIs(parse_rule("R_inv"), RuleName.R_INV)

    def test_all_enum_parsers_accept_enum_instances_and_canonical_strings(self):
        cases = [
            (parse_label, InteractionLabel.ATTRACT, "attract", InteractionLabel.ATTRACT),
            (parse_template_id, TemplateId.T1, "T2", TemplateId.T2),
            (
                parse_transition,
                Transition.R_STD_TO_R_INV,
                "R_inv_to_R_std",
                Transition.R_INV_TO_R_STD,
            ),
            (parse_split, Split.DEV, "public", Split.PUBLIC),
            (parse_difficulty, Difficulty.EASY, "hard", Difficulty.HARD),
            (parse_phase, Phase.PRE, "post", Phase.POST),
            (parse_item_kind, ItemKind.LABELED, "probe", ItemKind.PROBE),
        ]

        for parser, enum_value, string_value, expected in cases:
            with self.subTest(parser=parser.__name__, input=enum_value):
                self.assertIs(parser(enum_value), enum_value)

            with self.subTest(parser=parser.__name__, input=string_value):
                self.assertIs(parser(string_value), expected)

    def test_rule_parser_rejects_typos_with_canonical_values(self):
        with self.assertRaisesRegex(ValueError, r"unknown rule: R-std"):
            parse_rule("R-std")

    def test_label_parser_rejects_noncanonical_values(self):
        with self.assertRaisesRegex(ValueError, r"unknown label: repells"):
            parse_label("repells")

    def test_template_parser_rejects_noncanonical_values(self):
        with self.assertRaisesRegex(ValueError, r"unknown template_id: t1"):
            parse_template_id("t1")

    def test_other_enum_parsers_reject_noncanonical_values(self):
        invalid_cases = [
            (parse_transition, "R_std_to_R_std", "transition"),
            (parse_split, "prod", "split"),
            (parse_difficulty, "expert", "difficulty"),
            (parse_phase, "during", "phase"),
            (parse_item_kind, "target", "item_kind"),
        ]

        for parser, value, field_name in invalid_cases:
            with self.subTest(parser=parser.__name__, value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    rf"unknown {field_name}: {value}",
                ):
                    parser(value)

    def test_rules_expose_canonical_opposites_and_transitions(self):
        self.assertIs(RuleName.R_STD.opposite, RuleName.R_INV)
        self.assertIs(RuleName.R_INV.opposite, RuleName.R_STD)
        self.assertEqual(
            Transition.from_rules(RuleName.R_STD, RuleName.R_INV),
            Transition.R_STD_TO_R_INV,
        )
        self.assertEqual(
            Transition.from_rules(RuleName.R_INV, RuleName.R_STD),
            Transition.R_INV_TO_R_STD,
        )

    def test_protocol_enums_preserve_raw_string_behavior(self):
        cases = [
            (RuleName.R_STD, "R_std"),
            (InteractionLabel.REPEL, "repel"),
            (TemplateId.T2, "T2"),
            (Transition.R_INV_TO_R_STD, "R_inv_to_R_std"),
            (Split.PRIVATE, "private"),
            (Difficulty.MEDIUM, "medium"),
            (Phase.POST, "post"),
            (ItemKind.PROBE, "probe"),
        ]

        for member, expected in cases:
            with self.subTest(member=member):
                self.assertEqual(str(member), expected)
                self.assertIsInstance(member, str)
                self.assertEqual(member, expected)

    def test_template_specs_match_frozen_counts(self):
        self.assertEqual(RULES, frozenset(RuleName))
        self.assertEqual(TEMPLATE_IDS, frozenset(TemplateId))
        self.assertEqual(SPLITS, frozenset(Split))
        self.assertEqual(DIFFICULTIES, frozenset(Difficulty))
        self.assertEqual(PHASES, frozenset(Phase))
        self.assertEqual(ITEM_KINDS, frozenset(ItemKind))
        self.assertEqual(CHARGES, (-3, -2, -1, 1, 2, 3))
        self.assertEqual(frozenset(TEMPLATES), TEMPLATE_IDS)

        for template_id, spec in TEMPLATES.items():
            with self.subTest(template_id=template_id):
                self.assertIs(spec.template_id, template_id)
                self.assertEqual(spec.probe_count, PROBE_COUNT)
                self.assertEqual(spec.total_items, EPISODE_LENGTH)
                self.assertEqual(spec.pre_count + spec.post_labeled_count, 5)

        self.assertEqual(TEMPLATES[TemplateId.T1].pre_count, 2)
        self.assertEqual(TEMPLATES[TemplateId.T1].post_labeled_count, 3)
        self.assertEqual(TEMPLATES[TemplateId.T2].pre_count, 3)
        self.assertEqual(TEMPLATES[TemplateId.T2].post_labeled_count, 2)

    def test_case_space_matches_the_frozen_cartesian_product(self):
        expected = tuple((q1, q2) for q1 in CHARGES for q2 in CHARGES)
        self.assertEqual(CASE_SPACE, expected)

    def test_templates_catalog_is_read_only(self):
        with self.assertRaises(TypeError):
            TEMPLATES[TemplateId.T1] = TemplateSpec(
                template_id=TemplateId.T1,
                pre_count=2,
                post_labeled_count=3,
            )

    def test_template_spec_rejects_invalid_states_early(self):
        invalid_specs = [
            {
                "template_id": "T1",
                "pre_count": 2,
                "post_labeled_count": 3,
                "probe_count": PROBE_COUNT,
            },
            {
                "template_id": TemplateId.T1,
                "pre_count": 0,
                "post_labeled_count": 5,
                "probe_count": PROBE_COUNT,
            },
            {
                "template_id": TemplateId.T1,
                "pre_count": 2,
                "post_labeled_count": 2,
                "probe_count": PROBE_COUNT,
            },
            {
                "template_id": TemplateId.T1,
                "pre_count": 2,
                "post_labeled_count": 3,
                "probe_count": PROBE_COUNT + 1,
            },
        ]

        for kwargs in invalid_specs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises((TypeError, ValueError)):
                    TemplateSpec(**kwargs)

    def test_transition_rejects_same_rule_endpoints(self):
        with self.assertRaisesRegex(ValueError, "requires two distinct rules"):
            Transition.from_rules(RuleName.R_STD, RuleName.R_STD)


if __name__ == "__main__":
    unittest.main()
