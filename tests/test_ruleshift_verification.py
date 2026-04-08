import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.build_ruleshift_dataset import build_split, private_answer_key_payload, sanitize_private_rows
from scripts.verify_ruleshift import (
    attach_private_scoring,
    verify_private_answer_key,
    verify_split,
    verify_split_isolation,
)


class CogflexVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.public_rows, cls.public_answers = build_split("public", variants_per_family=2)
        for row in cls.public_rows:
            row.pop("split", None)
        private_rows, private_answers = build_split(
            "private",
            variants_per_family=10,
            private_seed="verification-test-private-seed",
        )
        cls.valid_private_rows = sanitize_private_rows(private_rows)
        cls.valid_private_answer_key = private_answer_key_payload(private_answers)

    def test_verify_split_isolation_rejects_semantic_overlap(self) -> None:
        public_rows = [json.loads(json.dumps(self.public_rows[0]))]
        private_rows = [json.loads(json.dumps(self.public_rows[0]))]
        private_rows[0]["episode_id"] = "9001"
        with self.assertRaisesRegex(RuntimeError, "split isolation violated"):
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
            missing_answer_key_path = Path(tmpdir) / "missing_private_answer_key.json"
            private_rows_path.write_text(json.dumps(self.valid_private_rows), encoding="utf-8")

            with patch("scripts.verify_ruleshift.PRIVATE_ROWS_PATH", private_rows_path), patch(
                "scripts.verify_ruleshift.PRIVATE_ANSWER_KEY_PATH", missing_answer_key_path
            ):
                with self.assertRaisesRegex(RuntimeError, "private split requires an answer key"):
                    verify_split("private")

    def test_verify_private_rejects_answer_key_mismatch(self) -> None:
        answer_key = json.loads(json.dumps(self.valid_private_answer_key))
        answer_key["episodes"][0]["difficulty_bin"] = "mismatch"

        with tempfile.TemporaryDirectory() as tmpdir:
            private_rows_path = Path(tmpdir) / "private_rows.json"
            answer_key_path = Path(tmpdir) / "private_answer_key.json"
            private_rows_path.write_text(json.dumps(self.valid_private_rows), encoding="utf-8")
            answer_key_path.write_text(json.dumps(answer_key), encoding="utf-8")

            with patch("scripts.verify_ruleshift.PRIVATE_ROWS_PATH", private_rows_path), patch(
                "scripts.verify_ruleshift.PRIVATE_ANSWER_KEY_PATH", answer_key_path
            ):
                with self.assertRaisesRegex(RuntimeError, "difficulty_bin mismatch"):
                    verify_private_answer_key(answer_key, self.valid_private_rows)

    def test_verify_private_accepts_inference_only_rows_with_external_answer_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, contextlib.redirect_stdout(io.StringIO()):
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

    def test_verify_private_surfaces_legacy_private_artifacts(self) -> None:
        legacy_rows = [
            {
                "episode_id": "0001",
                "inference": {"prompt": "legacy prompt"},
                "analysis": {
                    "faculty_id": "executive_functions/cognitive_flexibility",
                    "group_id": "old_shape",
                },
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            private_rows_path = Path(tmpdir) / "private_rows.json"
            private_rows_path.write_text(json.dumps(legacy_rows), encoding="utf-8")

            with patch("scripts.verify_ruleshift.PRIVATE_ROWS_PATH", private_rows_path):
                with self.assertRaisesRegex(RuntimeError, "legacy RuleShift artifacts"):
                    verify_split("private")

    def test_verify_public_reports_symbolic_baseline_summary(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            verify_split("public")
        payload = json.loads(stdout.getvalue())
        self.assertIn("symbolic_baseline", payload)
        self.assertIn("adversarial_baseline", payload)
        self.assertIn("nearest_neighbor_baseline", payload)
        self.assertIn("suite_task_counts", payload)
