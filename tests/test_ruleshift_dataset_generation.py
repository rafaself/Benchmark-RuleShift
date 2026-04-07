import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from scripts.build_ruleshift_dataset import (
    PRIVATE_DATASET_ID,
    build_private_artifacts,
    build_split,
    dataset_metadata,
    episode_signature,
    private_answer_key_payload,
    sanitize_private_rows,
)


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROWS_PATH = ROOT / "kaggle/dataset/public/public_leaderboard_rows.json"


class RuleshiftDatasetGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tracked_public_rows = json.loads(PUBLIC_ROWS_PATH.read_text(encoding="utf-8"))
        cls.generated_public_rows, cls.public_answers = build_split("public", variants_per_rule=1)
        cls.generated_private_rows, cls.private_answers = build_split(
            "private",
            variants_per_rule=5,
            variant_start=1,
            private_seed="unit-test-private-seed",
        )
        for row in cls.generated_public_rows:
            row.pop("split", None)
        cls.sanitized_private_rows = sanitize_private_rows(cls.generated_private_rows)
        cls.private_answer_key = private_answer_key_payload(cls.private_answers)

    def test_generated_public_split_matches_tracked_public_rows(self) -> None:
        self.assertEqual(self.generated_public_rows, self.tracked_public_rows)

    def test_generated_private_split_has_expected_counts(self) -> None:
        self.assertEqual(len(self.sanitized_private_rows), 400)
        counts = Counter(row["analysis"]["group_id"] for row in self.sanitized_private_rows)
        self.assertEqual(
            counts,
            Counter(
                {
                    "simple": 100,
                    "exception": 100,
                    "distractor": 100,
                    "hard": 100,
                }
            ),
        )

    def test_generated_private_split_has_no_semantic_duplicates(self) -> None:
        signatures = {episode_signature(answer) for answer in self.private_answers}
        self.assertEqual(len(signatures), len(self.private_answers))

    def test_generated_private_rows_do_not_expose_scoring(self) -> None:
        self.assertNotIn("scoring", self.sanitized_private_rows[0])

    def test_private_answer_key_contains_probe_targets(self) -> None:
        episodes = self.private_answer_key["episodes"]
        self.assertEqual(len(episodes), 400)
        self.assertEqual(len(episodes[0]["probe_targets"]), 4)

    def test_generated_public_and_private_splits_are_disjoint(self) -> None:
        public_signatures = {episode_signature(answer) for answer in self.public_answers}
        private_signatures = {episode_signature(answer) for answer in self.private_answers}
        self.assertFalse(public_signatures & private_signatures)

    def test_private_dataset_metadata_payload_matches_expected_id(self) -> None:
        self.assertEqual(
            dataset_metadata(PRIVATE_DATASET_ID, "RuleShift Runtime Private"),
            {
                "id": "raptorengineer/ruleshift-runtime-private",
                "title": "RuleShift Runtime Private",
                "licenses": [{"name": "CC0-1.0"}],
            },
        )

    def test_build_private_artifacts_requires_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_manifest = Path(tmpdir) / "missing_private_split_manifest.json"
            with self.assertRaisesRegex(FileNotFoundError, "Missing private split manifest"):
                build_private_artifacts(missing_manifest)
