import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.build_ruleshift_dataset import build_split, private_answer_key_payload, sanitize_private_rows
from scripts.verify_ruleshift import (
    attach_private_scoring,
    run_context_agnostic_baseline,
    run_previous_rule_baseline,
    verify_split,
    verify_split_isolation,
)


class RuleshiftVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.public_rows, _ = build_split("public", variants_per_rule=1)
        for row in cls.public_rows:
            row.pop("split", None)
        private_rows, private_answers = build_split(
            "private",
            variants_per_rule=5,
            variant_start=1,
            private_seed="verification-test-private-seed",
        )
        cls.valid_private_rows = sanitize_private_rows(private_rows)
        cls.valid_private_answer_key = private_answer_key_payload(private_answers)

    def test_verify_split_isolation_rejects_semantic_overlap(self) -> None:
        public_rows = [json.loads(json.dumps(self.public_rows[0]))]
        private_rows = [json.loads(json.dumps(self.public_rows[0]))]
        private_rows[0]["episode_id"] = "9001"
        private_rows[0]["scoring"]["final_probe_targets"] = tuple(private_rows[0]["scoring"]["final_probe_targets"])
        public_rows[0]["scoring"]["final_probe_targets"] = tuple(public_rows[0]["scoring"]["final_probe_targets"])

        with self.assertRaisesRegex(RuntimeError, "split isolation violated"):
            verify_split_isolation(public_rows, private_rows)

    def test_verify_split_isolation_rejects_family_overlap(self) -> None:
        public_rows = [json.loads(json.dumps(self.public_rows[0]))]
        private_rows = [json.loads(json.dumps(self.public_rows[1]))]
        public_rows[0]["scoring"]["final_probe_targets"] = tuple(public_rows[0]["scoring"]["final_probe_targets"])
        private_rows[0]["scoring"]["final_probe_targets"] = tuple(private_rows[0]["scoring"]["final_probe_targets"])
        private_rows[0]["analysis"]["transition_family_id"] = public_rows[0]["analysis"]["transition_family_id"]

        with self.assertRaisesRegex(RuntimeError, "transition families overlap"):
            verify_split_isolation(public_rows, private_rows)

    def test_attach_private_scoring_accepts_inference_only_rows(self) -> None:
        subset_rows = self.valid_private_rows[:3]
        subset_answer_key = {
            "version": self.valid_private_answer_key["version"],
            "split": self.valid_private_answer_key["split"],
            "episodes": self.valid_private_answer_key["episodes"][:3],
        }
        attached = attach_private_scoring(subset_rows, subset_answer_key)
        self.assertEqual(len(attached), 3)
        self.assertIn("scoring", attached[0])
        self.assertEqual(len(attached[0]["scoring"]["final_probe_targets"]), 4)

    def test_verify_private_requires_answer_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            private_rows_path = Path(tmpdir) / "private_rows.json"
            private_rows_path.write_text(json.dumps(self.valid_private_rows), encoding="utf-8")

            with patch("scripts.verify_ruleshift.PRIVATE_ROWS_PATH", private_rows_path):
                with self.assertRaisesRegex(RuntimeError, "private split requires an answer key"):
                    verify_split("private")

    def test_verify_private_rejects_answer_key_mismatch(self) -> None:
        answer_key = json.loads(json.dumps(self.valid_private_answer_key))
        answer_key["episodes"][0]["transition_family_id"] = "private::transition::mismatch"

        with tempfile.TemporaryDirectory() as tmpdir:
            private_rows_path = Path(tmpdir) / "private_rows.json"
            answer_key_path = Path(tmpdir) / "private_answer_key.json"
            private_rows_path.write_text(json.dumps(self.valid_private_rows), encoding="utf-8")
            answer_key_path.write_text(json.dumps(answer_key), encoding="utf-8")

            with patch("scripts.verify_ruleshift.PRIVATE_ROWS_PATH", private_rows_path), patch(
                "scripts.verify_ruleshift.PRIVATE_ANSWER_KEY_PATH", answer_key_path
            ):
                with self.assertRaisesRegex(RuntimeError, "transition_family_id mismatch"):
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

    def test_previous_rule_baseline_is_bounded_by_design(self) -> None:
        score = run_previous_rule_baseline(self.public_rows)
        self.assertEqual(score[1], 320)
        self.assertLess(score[0], 320)

    def test_context_agnostic_baseline_only_scores_context_switch_rows(self) -> None:
        score = run_context_agnostic_baseline(self.public_rows)
        self.assertEqual(score[1], 80)
        self.assertLessEqual(score[0], 40)
