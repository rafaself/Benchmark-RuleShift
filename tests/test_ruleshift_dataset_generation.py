import json
import unittest
from collections import Counter
from pathlib import Path

from scripts.build_ruleshift_dataset import build_split, episode_signature


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
        )
        for row in cls.generated_public_rows:
            row.pop("split", None)
        for row in cls.generated_private_rows:
            row.pop("split", None)

    def test_generated_public_split_matches_tracked_public_rows(self) -> None:
        self.assertEqual(self.generated_public_rows, self.tracked_public_rows)

    def test_generated_private_split_has_expected_counts(self) -> None:
        self.assertEqual(len(self.generated_private_rows), 400)
        counts = Counter(row["analysis"]["group_id"] for row in self.generated_private_rows)
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

    def test_generated_public_and_private_splits_are_disjoint(self) -> None:
        public_signatures = {episode_signature(answer) for answer in self.public_answers}
        private_signatures = {episode_signature(answer) for answer in self.private_answers}
        self.assertFalse(public_signatures & private_signatures)
