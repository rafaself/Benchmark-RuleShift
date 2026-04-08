import json
import unittest
from collections import Counter
from pathlib import Path

from scripts.build_cogflex_dataset import (
    NOTEBOOK_ID,
    PUBLIC_DATASET_ID,
    PUBLIC_FAMILY_IDS,
    PUBLIC_QUALITY_REPORT_PATH,
    PUBLIC_RULES,
    PUBLIC_ROWS_PATH,
    SUITE_TASKS,
    TASK_NAME,
    attack_limits_for_task,
    build_public_artifacts,
    dataset_metadata,
    episode_signature,
    has_exact_label_balance,
)


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

    def test_generated_public_split_matches_tracked_public_rows(self) -> None:
        self.assertEqual(self.generated_rows, self.tracked_rows)

    def test_generated_public_quality_report_matches_tracked_report(self) -> None:
        self.assertEqual(self.generated_report, self.tracked_report)

    def test_generated_public_split_has_expected_suite_shape(self) -> None:
        row = self.generated_rows[0]
        self.assertEqual(len(row["inference"]["turns"]), 3)
        self.assertEqual(len(row["scoring"]["final_probe_targets"]), 8)
        self.assertEqual(
            sorted(row["analysis"]),
            ["difficulty_bin", "faculty_id", "shift_mode", "suite_task_id"],
        )
        self.assertIn("shape=", row["inference"]["turns"][0])
        self.assertIn("tone=", row["inference"]["turns"][0])

    def test_generated_public_split_has_expected_counts(self) -> None:
        self.assertEqual(len(self.generated_rows), 120)
        task_counts = Counter(row["analysis"]["suite_task_id"] for row in self.generated_rows)
        self.assertEqual(task_counts, Counter({suite_task_id: 30 for suite_task_id in SUITE_TASKS}))
        difficulty_counts = Counter(row["analysis"]["difficulty_bin"] for row in self.generated_rows)
        self.assertEqual(difficulty_counts, Counter({"hard": 60, "medium": 60}))

    def test_generated_public_split_has_no_semantic_duplicates(self) -> None:
        signatures = {episode_signature(answer) for answer in self.generated_answers}
        self.assertEqual(len(signatures), len(self.generated_answers))

    def test_generated_public_split_neutralizes_simple_label_frequency_cues(self) -> None:
        for answer in self.generated_answers:
            self.assertTrue(
                has_exact_label_balance(
                    (item["label"] for item in answer["learn_turn_examples"]),
                    true_count=3,
                    false_count=3,
                )
            )
            self.assertTrue(
                has_exact_label_balance(
                    (item["label"] for item in answer["shift_turn_examples"]),
                    true_count=3,
                    false_count=3,
                )
            )
            self.assertTrue(
                has_exact_label_balance(
                    answer["final_probe_targets"],
                    true_count=4,
                    false_count=4,
                )
            )

    def test_public_report_tracks_transition_family_and_disagreement_coverage(self) -> None:
        self.assertEqual(self.generated_report["rule_inventory_count"], len(PUBLIC_RULES))
        self.assertEqual(self.generated_report["rule_family_count"], len(PUBLIC_FAMILY_IDS))
        self.assertEqual(self.generated_report["transition_family_count"], 15)
        self.assertEqual(self.generated_report["transition_pair_pattern_count"], 15)
        self.assertGreaterEqual(self.generated_report["selected_initial_family_count"], 5)
        self.assertGreaterEqual(self.generated_report["selected_shift_family_count"], 6)
        self.assertGreaterEqual(len(self.generated_report["disagreement_bin_counts"]), 2)
        usage_counts = Counter(self.generated_report["transition_family_usage"].values())
        self.assertEqual(usage_counts, Counter({8: 15}))

    def test_generated_public_answers_respect_attack_limits(self) -> None:
        for answer in self.generated_answers:
            limits = attack_limits_for_task(answer["suite_task_id"])
            diagnostics = answer["generator_diagnostics"]
            for metric, ceiling in limits.items():
                value = diagnostics.get(metric)
                if value is None:
                    continue
                self.assertLessEqual(float(value), ceiling, (answer["episode_id"], metric, value, ceiling))

    def test_generated_public_answers_require_turn2_and_are_disambiguated(self) -> None:
        for answer in self.generated_answers:
            diagnostics = answer["generator_diagnostics"]
            learn_only_ceiling = 0.5 if answer["suite_task_id"] in {"explicit_rule_update", "latent_rule_update"} else 0.625
            prediction_set_ceiling = 4 if answer["suite_task_id"] in {"explicit_rule_update", "latent_rule_update"} else 8
            self.assertLessEqual(float(diagnostics["learn_only_max_probe_accuracy"]), learn_only_ceiling)
            self.assertLessEqual(int(diagnostics["post_shift_prediction_set_size"]), prediction_set_ceiling)
            expected_required = 8 if answer["suite_task_id"] in {"explicit_rule_update", "latent_rule_update"} else 4
            self.assertEqual(int(diagnostics["turn2_required_probe_count"]), expected_required)

    def test_public_report_caps_majority_label_shortcut(self) -> None:
        majority_report = self.generated_report["attack_suite"]["majority_label_accuracy"]
        self.assertLessEqual(float(majority_report["micro_accuracy"]), 0.5)
        for value in majority_report["per_task_accuracy"].values():
            self.assertLessEqual(float(value), 0.5)

    def test_public_report_tracks_switching_diagnostics(self) -> None:
        switching = self.generated_report["switching_diagnostics"]
        self.assertLessEqual(float(switching["learn_only_max_probe_accuracy"]), 0.625)
        self.assertEqual(sum(switching["post_shift_prediction_set_size_distribution"].values()), 120)
        self.assertLessEqual(max(int(key) for key in switching["post_shift_prediction_set_size_distribution"]), 8)

    def test_public_dataset_metadata_payload_matches_expected_id(self) -> None:
        self.assertEqual(
            dataset_metadata(PUBLIC_DATASET_ID, "CogFlex Suite Runtime"),
            {
                "id": "raptorengineer/cogflex-suite-runtime",
                "title": "CogFlex Suite Runtime",
                "licenses": [{"name": "CC0-1.0"}],
            },
        )

    def test_makefile_and_kernel_metadata_point_to_cogflex_assets(self) -> None:
        makefile = MAKEFILE_PATH.read_text(encoding="utf-8")
        self.assertIn(".venv/bin/python -m scripts.verify_cogflex --split public", makefile)
        self.assertIn(".venv/bin/python -m scripts.verify_cogflex --split private", makefile)
        self.assertIn("cogflex_notebook_task.ipynb", makefile)
        metadata = json.loads(KERNEL_METADATA_PATH.read_text(encoding="utf-8"))
        self.assertEqual(metadata["id"], NOTEBOOK_ID)
        self.assertEqual(metadata["code_file"], "cogflex_notebook_task.ipynb")

    def test_readme_references_new_generator_and_private_bundle_contract(self) -> None:
        readme = README_PATH.read_text(encoding="utf-8")
        self.assertIn(TASK_NAME, readme)
        self.assertIn("build_cogflex_dataset.py", readme)
        self.assertIn("verify_cogflex.py", readme)
        self.assertIn("COGFLEX_PRIVATE_BUNDLE_DIR", readme)
        self.assertNotIn("rule" + "shift", readme.lower())
