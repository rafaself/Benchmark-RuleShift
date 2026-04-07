import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.build_ruleshift_dataset import build_split, private_answer_key_payload, sanitize_private_rows
from scripts.verify_ruleshift import attach_private_scoring, verify_split, verify_split_isolation


def make_row(
    episode_id: str,
    example_lines: list[str],
    probe_lines: list[str],
    targets: list[str],
    *,
    include_scoring: bool = True,
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
    row = {
        "episode_id": episode_id,
        "inference": {"prompt": prompt},
        "analysis": {
            "group_id": "simple",
            "rule_id": "r1_gt_r2",
            "shortcut_type": "rule_shortcut",
            "shortcut_rule_id": "r1_lt_r2",
        },
    }
    if include_scoring:
        row["scoring"] = {"probe_targets": targets}
    return row


def make_private_answer_key(rows: list[dict[str, object]]) -> dict[str, object]:
    return private_answer_key_payload(
        [
            {
                "episode_id": row["episode_id"],
                "group_id": row["analysis"]["group_id"],
                "rule_id": row["analysis"]["rule_id"],
                "rule_description": "type_a iff r1 is greater than r2",
                "shortcut_type": row["analysis"]["shortcut_type"],
                "shortcut_rule_id": row["analysis"]["shortcut_rule_id"],
                "shortcut_text": "guess from the sign of r1 instead of comparing both markers",
                "probe_targets": row["scoring"]["probe_targets"],
                "examples": [
                    {"index": 1, "r1": 1, "r2": -1, "label": "type_a"},
                ],
                "probes": [
                    {"index": 6, "r1": 1, "r2": 1, "label": "type_b", "justification": "mock"},
                ],
            }
            for row in rows
        ]
    )


class RuleshiftVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        private_rows, private_answers = build_split(
            "private",
            variants_per_rule=5,
            variant_start=1,
            private_seed="verification-test-private-seed",
        )
        cls.valid_private_rows = sanitize_private_rows(private_rows)
        cls.valid_private_answer_key = private_answer_key_payload(private_answers)

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

    def test_attach_private_scoring_accepts_inference_only_rows(self) -> None:
        public_like_rows = [
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
                include_scoring=False,
            )
        ]
        answer_key = make_private_answer_key(public_like_rows)

        attached = attach_private_scoring(private_rows, answer_key)

        self.assertEqual(attached[0]["scoring"]["probe_targets"], ("type_b", "type_a", "type_b", "type_b"))

    def test_verify_private_requires_answer_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            private_rows_path = Path(tmpdir) / "private_rows.json"
            private_rows_path.write_text(json.dumps(self.valid_private_rows), encoding="utf-8")

            with patch("scripts.verify_ruleshift.PRIVATE_ROWS_PATH", private_rows_path):
                with self.assertRaisesRegex(RuntimeError, "private split requires an answer key"):
                    verify_split("private")

    def test_verify_private_rejects_answer_key_mismatch(self) -> None:
        answer_key = json.loads(json.dumps(self.valid_private_answer_key))
        answer_key["episodes"][0]["episode_id"] = "9999"

        with tempfile.TemporaryDirectory() as tmpdir:
            private_rows_path = Path(tmpdir) / "private_rows.json"
            answer_key_path = Path(tmpdir) / "private_answer_key.json"
            private_rows_path.write_text(json.dumps(self.valid_private_rows), encoding="utf-8")
            answer_key_path.write_text(json.dumps(answer_key), encoding="utf-8")

            with patch("scripts.verify_ruleshift.PRIVATE_ROWS_PATH", private_rows_path), patch(
                "scripts.verify_ruleshift.PRIVATE_ANSWER_KEY_PATH", answer_key_path
            ):
                with self.assertRaisesRegex(RuntimeError, "unknown episode_id|missing episode_ids"):
                    verify_split("private")

    def test_verify_private_accepts_inference_only_rows_with_external_answer_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            private_rows_path = Path(tmpdir) / "private_rows.json"
            answer_key_path = Path(tmpdir) / "private_answer_key.json"
            manifest_path = Path(tmpdir) / "private_split_manifest.json"
            private_rows_path.write_text(json.dumps(self.valid_private_rows), encoding="utf-8")
            answer_key_path.write_text(json.dumps(self.valid_private_answer_key), encoding="utf-8")
            manifest_path.write_text(json.dumps({"private_seed": "verification-test-private-seed"}), encoding="utf-8")

            with patch("scripts.verify_ruleshift.PRIVATE_ROWS_PATH", private_rows_path), patch(
                "scripts.verify_ruleshift.PRIVATE_ANSWER_KEY_PATH", answer_key_path
            ), patch("scripts.verify_ruleshift.PRIVATE_MANIFEST_PATH", manifest_path):
                verify_split("private")
