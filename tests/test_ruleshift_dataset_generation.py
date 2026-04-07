import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from scripts.build_ruleshift_dataset import (
    PRIVATE_DATASET_ID,
    PUBLIC_DATASET_ID,
    build_private_artifacts,
    build_split,
    dataset_metadata,
    episode_signature,
    private_answer_key_payload,
    sanitize_private_rows,
)


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROWS_PATH = ROOT / "kaggle/dataset/public/public_leaderboard_rows.json"
KERNEL_METADATA_PATH = ROOT / "kaggle/notebook/kernel-metadata.json"
MAKEFILE_PATH = ROOT / "Makefile"
README_PATH = ROOT / "README.md"


class RuleshiftDatasetGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.generated_public_rows, cls.public_answers = build_split("public", variants_per_rule=1)
        for row in cls.generated_public_rows:
            row.pop("split", None)
        cls.tracked_public_rows = json.loads(PUBLIC_ROWS_PATH.read_text(encoding="utf-8"))
        cls.generated_private_rows, cls.private_answers = build_split(
            "private",
            variants_per_rule=5,
            variant_start=1,
            private_seed="unit-test-private-seed",
        )
        cls.sanitized_private_rows = sanitize_private_rows(cls.generated_private_rows)
        cls.private_answer_key = private_answer_key_payload(cls.private_answers)

    def test_generated_public_split_matches_tracked_public_rows(self) -> None:
        self.assertEqual(self.generated_public_rows, self.tracked_public_rows)

    def test_generated_public_split_has_expected_multi_turn_shape(self) -> None:
        row = self.generated_public_rows[0]
        self.assertEqual(len(row["inference"]["turns"]), 3)
        self.assertEqual(len(row["scoring"]["final_probe_targets"]), 4)
        self.assertEqual(
            sorted(row["analysis"]),
            [
                "faculty_id",
                "group_id",
                "initial_rule_id",
                "shift_mode",
                "shift_rule_id",
                "transition_family_id",
            ],
        )

    def test_generated_private_split_has_expected_counts(self) -> None:
        self.assertEqual(len(self.sanitized_private_rows), 400)
        counts = Counter(row["analysis"]["group_id"] for row in self.sanitized_private_rows)
        self.assertEqual(
            counts,
            Counter(
                {
                    "explicit_switch": 100,
                    "reversal": 100,
                    "latent_switch": 100,
                    "context_switch": 100,
                }
            ),
        )

    def test_generated_private_split_has_no_semantic_duplicates(self) -> None:
        signatures = {episode_signature(answer) for answer in self.private_answers}
        self.assertEqual(len(signatures), len(self.private_answers))

    def test_generated_private_rows_do_not_expose_scoring(self) -> None:
        self.assertNotIn("scoring", self.sanitized_private_rows[0])
        self.assertEqual(len(self.sanitized_private_rows[0]["inference"]["turns"]), 3)

    def test_private_answer_key_contains_turns_and_final_probe_targets(self) -> None:
        episodes = self.private_answer_key["episodes"]
        self.assertEqual(len(episodes), 400)
        self.assertEqual(len(episodes[0]["turns"]), 3)
        self.assertEqual(len(episodes[0]["final_probe_targets"]), 4)

    def test_generated_public_and_private_splits_are_disjoint(self) -> None:
        public_signatures = {episode_signature(answer) for answer in self.public_answers}
        private_signatures = {episode_signature(answer) for answer in self.private_answers}
        self.assertFalse(public_signatures & private_signatures)

    def test_generated_public_and_private_transition_families_are_disjoint(self) -> None:
        public_families = {answer["transition_family_id"] for answer in self.public_answers}
        private_families = {answer["transition_family_id"] for answer in self.private_answers}
        self.assertFalse(public_families & private_families)

    def test_public_dataset_metadata_payload_matches_expected_id(self) -> None:
        self.assertEqual(
            dataset_metadata(PUBLIC_DATASET_ID, "RuleShift CogFlex Runtime v2"),
            {
                "id": "raptorengineer/ruleshift-cogflex-runtime-v2",
                "title": "RuleShift CogFlex Runtime v2",
                "licenses": [{"name": "CC0-1.0"}],
            },
        )

    def test_private_dataset_metadata_payload_matches_expected_id(self) -> None:
        self.assertEqual(
            dataset_metadata(PRIVATE_DATASET_ID, "RuleShift CogFlex Runtime Private v2"),
            {
                "id": "raptorengineer/ruleshift-cogflex-runtime-private-v2",
                "title": "RuleShift CogFlex Runtime Private v2",
                "licenses": [{"name": "CC0-1.0"}],
            },
        )

    def test_build_private_artifacts_requires_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_manifest = Path(tmpdir) / "missing_private_split_manifest.json"
            with self.assertRaisesRegex(FileNotFoundError, "Missing private split manifest"):
                build_private_artifacts(missing_manifest)

    def test_makefile_uses_module_verifier_entrypoint(self) -> None:
        makefile = MAKEFILE_PATH.read_text(encoding="utf-8")
        self.assertIn(".venv/bin/python -m scripts.verify_ruleshift --split public", makefile)
        self.assertIn(".venv/bin/python -m scripts.verify_ruleshift --split private", makefile)

    def test_kernel_metadata_matches_v2_assets(self) -> None:
        metadata = json.loads(KERNEL_METADATA_PATH.read_text(encoding="utf-8"))
        self.assertEqual(metadata["id"], "raptorengineer/ruleshift-cogflex-notebook-v2")
        self.assertEqual(
            metadata["dataset_sources"],
            [
                "raptorengineer/ruleshift-cogflex-runtime-v2",
                "raptorengineer/ruleshift-cogflex-runtime-private-v2",
            ],
        )

    def test_readme_references_new_task_and_assets(self) -> None:
        readme = README_PATH.read_text(encoding="utf-8")
        self.assertIn("ruleshift_cogflex_v2_binary", readme)
        self.assertIn("raptorengineer/ruleshift-cogflex-runtime-v2", readme)
        self.assertIn("raptorengineer/ruleshift-cogflex-runtime-private-v2", readme)
        self.assertIn("raptorengineer/ruleshift-cogflex-notebook-v2", readme)
