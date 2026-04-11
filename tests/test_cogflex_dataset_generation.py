import json
import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.build_cogflex_dataset import (
    NOTEBOOK_ID,
    PRIVATE_DATASET_ID,
    PUBLIC_DATASET_ID,
    PUBLIC_DIFFICULTY_CALIBRATION_PATH,
    PUBLIC_QUALITY_REPORT_PATH,
    PUBLIC_ROWS_PATH,
    REQUIRED_PRIVATE_STRUCTURE_FAMILY_IDS,
    SUITE_TASKS,
    TASK_NAME,
    build_public_artifacts,
    dataset_metadata,
    empirical_difficulty_entries_from_scores,
    load_public_difficulty_calibration,
)
from scripts.build_private_cogflex_dataset import build_private_bundle


ROOT = Path(__file__).resolve().parents[1]
KERNEL_METADATA_PATH = ROOT / "kaggle/notebook/kernel-metadata.json"
MAKEFILE_PATH = ROOT / "Makefile"
README_PATH = ROOT / "README.md"


class CogflexDatasetGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.generated_rows, cls.generated_answers, cls.generated_report = build_public_artifacts()
        cls.tracked_rows = json.loads(PUBLIC_ROWS_PATH.read_text(encoding="utf-8"))
        cls.tracked_report = json.loads(PUBLIC_QUALITY_REPORT_PATH.read_text(encoding="utf-8"))

    def test_generated_public_split_matches_tracked_rows(self) -> None:
        self.assertEqual(self.generated_rows, self.tracked_rows)

    def test_generated_public_report_matches_tracked_report(self) -> None:
        self.assertEqual(self.generated_report, self.tracked_report)

    def test_public_row_uses_flexible_contract(self) -> None:
        row = self.generated_rows[0]
        self.assertEqual(
            sorted(row["analysis"]),
            ["difficulty_bin", "faculty_id", "shift_mode", "structure_family_id", "suite_task_id"],
        )
        self.assertEqual(sorted(row["inference"]), ["response_spec", "turn_specs", "turns"])
        self.assertEqual(len(row["inference"]["turns"]), len(row["inference"]["turn_specs"]))
        self.assertEqual(row["inference"]["turn_specs"][-1]["kind"], "decision")
        self.assertEqual(
            row["inference"]["response_spec"]["probe_count"],
            row["inference"]["turn_specs"][-1]["item_count"],
        )
        self.assertEqual(
            len(row["scoring"]["final_probe_targets"]),
            row["inference"]["response_spec"]["probe_count"],
        )
        self.assertIn("probe_annotations", row["scoring"])
        self.assertEqual(
            len(row["scoring"]["probe_annotations"]),
            row["inference"]["response_spec"]["probe_count"],
        )
        self.assertTrue(all(a in ("congruent", "incongruent") for a in row["scoring"]["probe_annotations"]))

    def test_public_split_has_expected_counts(self) -> None:
        self.assertEqual(len(self.generated_rows), 120)
        task_counts = Counter(row["analysis"]["suite_task_id"] for row in self.generated_rows)
        self.assertEqual(task_counts, Counter({suite_task_id: 30 for suite_task_id in SUITE_TASKS}))
        difficulty_counts = Counter(row["analysis"]["difficulty_bin"] for row in self.generated_rows)
        self.assertEqual(difficulty_counts, Counter({"hard": 60, "medium": 60}))

    def test_public_split_uses_tracked_difficulty_calibration_snapshot(self) -> None:
        _payload, entries_by_episode = load_public_difficulty_calibration()
        self.assertEqual(
            {row["episode_id"]: row["analysis"]["difficulty_bin"] for row in self.generated_rows},
            {episode_id: entry["difficulty_bin"] for episode_id, entry in entries_by_episode.items()},
        )

    def test_empirical_difficulty_tie_breaks_by_episode_id(self) -> None:
        entries = empirical_difficulty_entries_from_scores({"0002": 0.25, "0001": 0.25, "0003": 0.9, "0004": 0.9})
        self.assertEqual(entries["0001"]["rank"], 1)
        self.assertEqual(entries["0001"]["difficulty_bin"], "hard")
        self.assertEqual(entries["0002"]["rank"], 2)
        self.assertEqual(entries["0002"]["difficulty_bin"], "hard")
        self.assertEqual(entries["0003"]["rank"], 3)
        self.assertEqual(entries["0003"]["difficulty_bin"], "medium")
        self.assertEqual(entries["0004"]["rank"], 4)
        self.assertEqual(entries["0004"]["difficulty_bin"], "medium")

    def test_build_public_artifacts_rejects_calibration_coverage_mismatch(self) -> None:
        _payload, entries_by_episode = load_public_difficulty_calibration()
        missing_episode_id = next(iter(entries_by_episode))
        incomplete_entries = {episode_id: value for episode_id, value in entries_by_episode.items() if episode_id != missing_episode_id}
        with patch(
            "scripts.build_cogflex_dataset.load_public_difficulty_calibration",
            return_value=({"version": "test"}, incomplete_entries),
        ):
            with self.assertRaisesRegex(RuntimeError, "coverage mismatch"):
                build_public_artifacts()

    def test_public_split_varies_turn_probe_and_label_shapes(self) -> None:
        self.assertEqual(
            sorted({len(row["inference"]["turns"]) for row in self.generated_rows}),
            [3, 4],
        )
        self.assertEqual(
            sorted({row["inference"]["response_spec"]["probe_count"] for row in self.generated_rows}),
            [5, 6],
        )
        self.assertEqual(
            sorted({len(row["inference"]["response_spec"]["label_vocab"]) for row in self.generated_rows}),
            [2, 3],
        )

    def test_each_suite_task_uses_at_least_two_structure_families(self) -> None:
        per_task = self.generated_report["suite_task_structure_counts"]
        for suite_task_id in SUITE_TASKS:
            self.assertGreaterEqual(len(per_task[suite_task_id]), 2)

    def test_public_report_tracks_flexible_distributions(self) -> None:
        self.assertEqual(self.generated_report["task_name"], TASK_NAME)
        self.assertEqual(self.generated_report["row_count"], 120)
        self.assertEqual(
            self.generated_report["turn_count_distribution"],
            {"3": 60, "4": 60},
        )
        self.assertEqual(
            self.generated_report["probe_count_distribution"],
            {"5": 60, "6": 60},
        )
        self.assertEqual(
            self.generated_report["structure_family_counts"],
            {"three_step_bridge": 60, "two_step_focus": 60},
        )
        self.assertIn("optional_field_keys", self.generated_report["stimulus_space_summary"])

    def test_public_dataset_metadata_payload_matches_expected_id(self) -> None:
        self.assertEqual(
            dataset_metadata(PUBLIC_DATASET_ID, "CogFlex Cognitive Flexibility Runtime"),
            {
                "id": "raptorengineer/cogflex-suite-runtime",
                "title": "CogFlex Cognitive Flexibility Runtime",
                "licenses": [{"name": "CC0-1.0"}],
            },
        )

    def test_build_private_bundle_materializes_required_files(self) -> None:
        with TemporaryDirectory() as tmpdir:
            bundle_paths = build_private_bundle(Path(tmpdir))
            self.assertEqual(
                json.loads(bundle_paths["metadata"].read_text(encoding="utf-8")),
                dataset_metadata(PRIVATE_DATASET_ID, "CogFlex Suite Runtime Private"),
            )
            for key in ("rows", "answer_key", "predictions", "quality", "manifest", "metadata"):
                self.assertTrue(bundle_paths[key].exists(), key)
            rows = json.loads(bundle_paths["rows"].read_text(encoding="utf-8"))
            self.assertEqual(len(rows), len(REQUIRED_PRIVATE_STRUCTURE_FAMILY_IDS) * len(SUITE_TASKS))
            structure_counts = Counter(row["analysis"]["structure_family_id"] for row in rows)
            self.assertEqual(
                structure_counts,
                Counter({structure_family_id: len(SUITE_TASKS) for structure_family_id in REQUIRED_PRIVATE_STRUCTURE_FAMILY_IDS}),
            )

    def test_makefile_and_kernel_metadata_point_to_cogflex_assets(self) -> None:
        makefile = MAKEFILE_PATH.read_text(encoding="utf-8")
        self.assertIn("PYTHON ?= python3", makefile)
        self.assertIn("JUPYTER ?= jupyter", makefile)
        self.assertIn("$(JUPYTER) lab --no-browser kaggle/notebook/cogflex_notebook_task.ipynb", makefile)
        self.assertIn("$(PYTHON) -m scripts.verify_cogflex --split public", makefile)
        self.assertIn("$(PYTHON) -m scripts.verify_cogflex --split private", makefile)
        self.assertIn("COGFLEX_PRIVATE_BUNDLE_DIR is required for verify-private", makefile)
        self.assertIn("cogflex_notebook_task.ipynb", makefile)
        metadata = json.loads(KERNEL_METADATA_PATH.read_text(encoding="utf-8"))
        self.assertEqual(metadata["id"], NOTEBOOK_ID)
        self.assertEqual(metadata["code_file"], "cogflex_notebook_task.ipynb")

    def test_readme_references_flexible_contract(self) -> None:
        readme = README_PATH.read_text(encoding="utf-8")
        self.assertIn("cognitive flexibility within executive functions", readme)
        self.assertIn(TASK_NAME, readme)
        self.assertIn("turn_specs", readme)
        self.assertIn("response_spec", readme)
        self.assertIn("structure_family_id", readme)
        self.assertIn(PUBLIC_DIFFICULTY_CALIBRATION_PATH.name, readme)
        self.assertIn("generator_isolation_summary", readme)
        self.assertIn("operator_class", readme)
