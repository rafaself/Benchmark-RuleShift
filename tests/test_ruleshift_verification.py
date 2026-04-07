import unittest

from scripts.verify_ruleshift import verify_split_isolation


def make_row(
    episode_id: str,
    example_lines: list[str],
    probe_lines: list[str],
    targets: list[str],
) -> dict[str, object]:
    prompt = "\n\n".join(
        [
            f"RuleShift classification task. Episode {episode_id}.",
            "Examples:\n" + "\n".join(example_lines),
            "Probes:\n" + "\n".join(probe_lines),
            (
                "Return exactly 4 outputs in order, one per probe. "
                "Use only type_a or type_b. Map zark to type_a and blim to type_b."
            ),
        ]
    )
    return {
        "episode_id": episode_id,
        "inference": {"prompt": prompt},
        "scoring": {"probe_targets": targets},
        "analysis": {
            "group_id": "simple",
            "rule_id": "r1_gt_r2",
            "shortcut_type": "rule_shortcut",
            "shortcut_rule_id": "r1_lt_r2",
        },
    }


class RuleshiftVerificationTests(unittest.TestCase):
    def test_verify_split_isolation_rejects_semantic_overlap(self) -> None:
        public_rows = [
            make_row(
                "0001",
                [
                    "1. r1=+1, r2=-1 -> zark",
                    "2. r1=+2, r2=-2 -> zark",
                    "3. r1=-1, r2=+1 -> blim",
                    "4. r1=+3, r2=+2 -> zark",
                    "5. r1=-2, r2=+2 -> blim",
                ],
                [
                    "6. r1=+1, r2=+1 -> ?",
                    "7. r1=+3, r2=-3 -> ?",
                    "8. r1=-3, r2=+3 -> ?",
                    "9. r1=+2, r2=+1 -> ?",
                ],
                ["type_b", "type_a", "type_b", "type_a"],
            )
        ]
        private_rows = [
            make_row(
                "1001",
                [
                    "1. r1=+1, r2=-1 -> zark",
                    "2. r1=+2, r2=-2 -> zark",
                    "3. r1=-1, r2=+1 -> blim",
                    "4. r1=+3, r2=+2 -> zark",
                    "5. r1=-2, r2=+2 -> blim",
                ],
                [
                    "6. r1=+1, r2=+1 -> ?",
                    "7. r1=+3, r2=-3 -> ?",
                    "8. r1=-3, r2=+3 -> ?",
                    "9. r1=+2, r2=+1 -> ?",
                ],
                ["type_b", "type_a", "type_b", "type_a"],
            )
        ]

        with self.assertRaisesRegex(RuntimeError, "split isolation violated"):
            verify_split_isolation(public_rows, private_rows)

    def test_verify_split_isolation_accepts_distinct_rows(self) -> None:
        public_rows = [
            make_row(
                "0001",
                [
                    "1. r1=+1, r2=-1 -> zark",
                    "2. r1=+2, r2=-2 -> zark",
                    "3. r1=-1, r2=+1 -> blim",
                    "4. r1=+3, r2=+2 -> zark",
                    "5. r1=-2, r2=+2 -> blim",
                ],
                [
                    "6. r1=+1, r2=+1 -> ?",
                    "7. r1=+3, r2=-3 -> ?",
                    "8. r1=-3, r2=+3 -> ?",
                    "9. r1=+2, r2=+1 -> ?",
                ],
                ["type_b", "type_a", "type_b", "type_a"],
            )
        ]
        private_rows = [
            make_row(
                "1001",
                [
                    "1. r1=-1, r2=+1 -> blim",
                    "2. r1=-2, r2=+2 -> blim",
                    "3. r1=+1, r2=-1 -> zark",
                    "4. r1=-3, r2=-2 -> blim",
                    "5. r1=+2, r2=-2 -> zark",
                ],
                [
                    "6. r1=-1, r2=-1 -> ?",
                    "7. r1=+2, r2=-3 -> ?",
                    "8. r1=-3, r2=-2 -> ?",
                    "9. r1=+2, r2=+3 -> ?",
                ],
                ["type_b", "type_a", "type_b", "type_b"],
            )
        ]

        verify_split_isolation(public_rows, private_rows)
