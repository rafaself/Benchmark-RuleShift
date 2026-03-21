from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rules import CHARGES, RULES, label


class RulesTestCase(unittest.TestCase):
    def test_same_sign_behavior(self):
        cases = [
            ("R_std", 1, 3, "repel"),
            ("R_std", -1, -3, "repel"),
            ("R_inv", 1, 3, "attract"),
            ("R_inv", -1, -3, "attract"),
        ]

        for rule, q1, q2, expected in cases:
            with self.subTest(rule=rule, q1=q1, q2=q2):
                self.assertEqual(label(rule, q1, q2), expected)

    def test_opposite_sign_behavior(self):
        cases = [
            ("R_std", 1, -3, "attract"),
            ("R_std", -1, 3, "attract"),
            ("R_inv", 1, -3, "repel"),
            ("R_inv", -1, 3, "repel"),
        ]

        for rule, q1, q2, expected in cases:
            with self.subTest(rule=rule, q1=q1, q2=q2):
                self.assertEqual(label(rule, q1, q2), expected)

    def test_swap_invariance(self):
        for rule in sorted(RULES):
            for q1 in CHARGES:
                for q2 in CHARGES:
                    with self.subTest(rule=rule, q1=q1, q2=q2):
                        self.assertEqual(label(rule, q1, q2), label(rule, q2, q1))

    def test_magnitude_irrelevance_with_fixed_sign_pattern(self):
        cases = [
            ("R_std", [(1, 2), (1, 3), (2, 3), (-1, -2), (-1, -3), (-2, -3)]),
            ("R_std", [(1, -1), (1, -2), (2, -3), (-1, 1), (-2, 1), (-3, 2)]),
            ("R_inv", [(1, 2), (1, 3), (2, 3), (-1, -2), (-1, -3), (-2, -3)]),
            ("R_inv", [(1, -1), (1, -2), (2, -3), (-1, 1), (-2, 1), (-3, 2)]),
        ]

        for rule, examples in cases:
            with self.subTest(rule=rule, examples=examples):
                labels = {label(rule, q1, q2) for q1, q2 in examples}
                self.assertEqual(len(labels), 1)

    def test_rules_always_disagree_for_allowed_charge_pairs(self):
        for q1 in CHARGES:
            for q2 in CHARGES:
                with self.subTest(q1=q1, q2=q2):
                    self.assertNotEqual(label("R_std", q1, q2), label("R_inv", q1, q2))

    def test_invalid_rule_name_fails_cleanly(self):
        with self.assertRaisesRegex(ValueError, "unknown rule"):
            label("R_bad", 1, -1)


if __name__ == "__main__":
    unittest.main()
