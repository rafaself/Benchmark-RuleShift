import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from scripts.build_ruleshift_dataset import (
    NOTEBOOK_ID,
    PRIVATE_DATASET_ID,
    PUBLIC_DATASET_ID,
    SUITE_TASKS,
    TASK_NAME,
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


class CogflexDatasetGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.generated_public_rows, cls.public_answers = build_split("public", variants_per_family=2)
        for row in cls.generated_public_rows:
            row.pop("split", None)
        cls.tracked_public_rows = json.loads(PUBLIC_ROWS_PATH.read_text(encoding="utf-8"))
        cls.generated_private_rows, cls.private_answers = build_split(
            "private",
            variants_per_family=10,
            private_seed="unit-test-private-seed",
        )
        cls.sanitized_private_rows = sanitize_private_rows(cls.generated_private_rows)
        cls.private_answer_key = private_answer_key_payload(cls.private_answers)

    def test_generated_public_split_matches_tracked_public_rows(self) -> None:
        self.assertEqual(self.generated_public_rows, self.tracked_public_rows)

    def test_generated_public_split_has_expected_suite_shape(self) -> None:
        row = self.generated_public_rows[0]
        self.assertEqual(len(row["inference"]["turns"]), 3)
        self.assertEqual(len(row["scoring"]["final_probe_targets"]), 4)
        self.assertEqual(
            sorted(row["analysis"]),
            ["difficulty_bin", "faculty_id", "shift_mode", "suite_task_id"],
        )
        self.assertNotIn("initial_rule_id", row["analysis"])
        self.assertNotIn("shift_rule_id", row["analysis"])
        self.assertIn("shape=", row["inference"]["turns"][0])
        self.assertIn("tone=", row["inference"]["turns"][0])

    def test_generated_private_split_has_expected_counts(self) -> None:
        self.assertEqual(len(self.sanitized_private_rows), 400)
        task_counts = Counter(row["analysis"]["suite_task_id"] for row in self.sanitized_private_rows)
        self.assertEqual(task_counts, Counter({suite_task_id: 100 for suite_task_id in SUITE_TASKS}))
        difficulty_counts = Counter(row["analysis"]["difficulty_bin"] for row in self.sanitized_private_rows)
        self.assertEqual(difficulty_counts, Counter({"hard": 200, "medium": 200}))

    def test_generated_private_split_has_no_semantic_duplicates(self) -> None:
        signatures = {episode_signature(answer) for answer in self.private_answers}
        self.assertEqual(len(signatures), len(self.private_answers))

    def test_generated_private_rows_do_not_expose_scoring_or_latent_metadata(self) -> None:
        row = self.sanitized_private_rows[0]
        self.assertNotIn("scoring", row)
        self.assertEqual(
            sorted(row["analysis"]),
            ["difficulty_bin", "faculty_id", "shift_mode", "suite_task_id"],
        )

    def test_private_answer_key_contains_latent_metadata(self) -> None:
        episode = self.private_answer_key["episodes"][0]
        self.assertIn("initial_rule_id", episode)
        self.assertIn("shift_rule_id", episode)
        self.assertIn("cue_template_id", episode)
        self.assertIn("context_template_id", episode)
        self.assertIn("surface_template_id", episode)
        self.assertIn("generator_diagnostics", episode)
        self.assertEqual(len(episode["final_probe_targets"]), 4)

    def test_shift_tasks_use_distinct_structural_evidence_profiles(self) -> None:
        explicit_conflicts = {
            answer["generator_diagnostics"]["shift_evidence_conflict_count"]
            for answer in self.public_answers
            if answer["suite_task_id"] == "explicit_rule_update"
        }
        latent_conflicts = {
            answer["generator_diagnostics"]["shift_evidence_conflict_count"]
            for answer in self.public_answers
            if answer["suite_task_id"] == "latent_rule_update"
        }
        self.assertTrue(all(conflicts >= 3 for conflicts in explicit_conflicts))
        self.assertEqual(latent_conflicts, {2})

    def test_generated_public_and_private_splits_are_disjoint(self) -> None:
        public_signatures = {episode_signature(answer) for answer in self.public_answers}
        private_signatures = {episode_signature(answer) for answer in self.private_answers}
        self.assertFalse(public_signatures & private_signatures)

    def test_public_dataset_metadata_payload_matches_expected_id(self) -> None:
        self.assertEqual(
            dataset_metadata(PUBLIC_DATASET_ID, "CogFlex Suite Runtime"),
            {
                "id": "raptorengineer/cogflex-suite-runtime",
                "title": "CogFlex Suite Runtime",
                "licenses": [{"name": "CC0-1.0"}],
            },
        )

    def test_private_dataset_metadata_payload_matches_expected_id(self) -> None:
        self.assertEqual(
            dataset_metadata(PRIVATE_DATASET_ID, "CogFlex Suite Runtime Private"),
            {
                "id": "raptorengineer/cogflex-suite-runtime-private",
                "title": "CogFlex Suite Runtime Private",
                "licenses": [{"name": "CC0-1.0"}],
            },
        )

    def test_build_private_artifacts_requires_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_manifest = Path(tmpdir) / "missing_private_split_manifest.json"
            with self.assertRaisesRegex(FileNotFoundError, "Missing private split manifest"):
                build_private_artifacts(missing_manifest)

    def test_makefile_exposes_test_and_verifier_targets(self) -> None:
        makefile = MAKEFILE_PATH.read_text(encoding="utf-8")
        self.assertIn(".venv/bin/python -m unittest discover -s tests -q", makefile)
        self.assertIn(".venv/bin/python -m scripts.verify_ruleshift --split public", makefile)
        self.assertIn(".venv/bin/python -m scripts.verify_ruleshift --split private", makefile)

    def test_kernel_metadata_matches_canonical_assets(self) -> None:
        metadata = json.loads(KERNEL_METADATA_PATH.read_text(encoding="utf-8"))
        self.assertEqual(metadata["id"], NOTEBOOK_ID)
        self.assertEqual(
            metadata["dataset_sources"],
            [
                "raptorengineer/cogflex-suite-runtime",
                "raptorengineer/cogflex-suite-runtime-private",
            ],
        )

    def test_readme_references_new_task_and_assets(self) -> None:
        readme = README_PATH.read_text(encoding="utf-8")
        self.assertIn(TASK_NAME, readme)
        self.assertIn("raptorengineer/cogflex-suite-runtime", readme)
        self.assertIn("raptorengineer/cogflex-suite-runtime-private", readme)
        self.assertIn("raptorengineer/cogflex-suite-notebook", readme)
