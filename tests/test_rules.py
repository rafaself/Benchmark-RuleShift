from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from protocol import CHARGES, InteractionLabel, RuleName
from rules import label


class RulesTestCase(unittest.TestCase):
    def test_same_sign_behavior(self):
        cases = [
            (RuleName.R_STD, 1, 3, InteractionLabel.REPEL),
            (RuleName.R_STD, -1, -3, InteractionLabel.REPEL),
            (RuleName.R_INV, 1, 3, InteractionLabel.ATTRACT),
            (RuleName.R_INV, -1, -3, InteractionLabel.ATTRACT),
        ]

        for rule, q1, q2, expected in cases:
            with self.subTest(rule=rule, q1=q1, q2=q2):
                self.assertEqual(label(rule, q1, q2), expected)

    def test_opposite_sign_behavior(self):
        cases = [
            (RuleName.R_STD, 1, -3, InteractionLabel.ATTRACT),
            (RuleName.R_STD, -1, 3, InteractionLabel.ATTRACT),
            (RuleName.R_INV, 1, -3, InteractionLabel.REPEL),
            (RuleName.R_INV, -1, 3, InteractionLabel.REPEL),
        ]

        for rule, q1, q2, expected in cases:
            with self.subTest(rule=rule, q1=q1, q2=q2):
                self.assertEqual(label(rule, q1, q2), expected)

    def test_swap_invariance(self):
        for rule in RuleName:
            for i, q1 in enumerate(CHARGES):
                for q2 in CHARGES[i:]:
                    with self.subTest(rule=rule, q1=q1, q2=q2):
                        self.assertEqual(label(rule, q1, q2), label(rule, q2, q1))

    def test_magnitude_irrelevance_with_fixed_sign_pattern(self):
        cases = [
            (RuleName.R_STD, [(1, 2), (1, 3), (2, 3), (-1, -2), (-1, -3), (-2, -3)]),
            (RuleName.R_STD, [(1, -1), (1, -2), (2, -3), (-1, 1), (-2, 1), (-3, 2)]),
            (RuleName.R_INV, [(1, 2), (1, 3), (2, 3), (-1, -2), (-1, -3), (-2, -3)]),
            (RuleName.R_INV, [(1, -1), (1, -2), (2, -3), (-1, 1), (-2, 1), (-3, 2)]),
        ]

        for rule, examples in cases:
            with self.subTest(rule=rule, examples=examples):
                labels = {label(rule, q1, q2) for q1, q2 in examples}
                self.assertEqual(len(labels), 1)

    def test_rules_always_disagree_for_allowed_charge_pairs(self):
        for q1 in CHARGES:
            for q2 in CHARGES:
                with self.subTest(q1=q1, q2=q2):
                    self.assertNotEqual(
                        label(RuleName.R_STD, q1, q2),
                        label(RuleName.R_INV, q1, q2),
                    )

    def test_invalid_rule_name_fails_cleanly(self):
        with self.assertRaisesRegex(ValueError, "unknown rule"):
            label("R_bad", 1, -1)

    def test_label_accepts_canonical_rule_string_input(self):
        self.assertEqual(label("R_std", 1, -1), InteractionLabel.ATTRACT)


if __name__ == "__main__":
    unittest.main()
