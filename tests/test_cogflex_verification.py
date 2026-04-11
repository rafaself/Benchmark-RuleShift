import contextlib
import io
import json
import re
import tempfile
import unittest
from pathlib import Path

from scripts.private_cogflex_bundle import (
    private_row_summary as _private_row_summary,
    public_fixture,
    write_private_bundle,
)
from scripts.build_cogflex_dataset import (
    PRIVATE_ANSWER_KEY_FILENAME,
    PRIVATE_BUNDLE_VERSION,
    PRIVATE_CALIBRATION_PREDICTIONS_FILENAME,
    PRIVATE_QUALITY_REPORT_FILENAME,
    PRIVATE_QUALITY_REPORT_VERSION,
    PRIVATE_RELEASE_MANIFEST_FILENAME,
    PRIVATE_ROWS_FILENAME,
    PUBLIC_DIFFICULTY_CALIBRATION_PATH,
    PUBLIC_QUALITY_REPORT_PATH,
    PUBLIC_ROWS_PATH,
    REQUIRED_PRIVATE_STRUCTURE_FAMILY_IDS,
    compute_sha256,
    empirical_difficulty_entries_from_predictions,
    public_generator_reference,
)
from scripts.verify_cogflex import (
    attach_private_scoring,
    build_private_quality_report,
    load_private_calibration_predictions,
    load_private_answer_key,
    structural_overlap_score,
    verify_manifest,
    verify_private_answer_key,
    verify_private_bundle,
    verify_private_calibration_predictions,
    verify_public_difficulty_calibration,
    verify_public_split,
    verify_quality_report,
    verify_schema,
    verify_split_isolation,
)


def _load_bundle_payloads(
    bundle_paths: dict[str, Path],
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    """Load the private bundle JSON artifacts used by verification tests.

    Args:
        bundle_paths: Mapping from logical bundle names to artifact paths.

    Returns:
        The parsed rows, answer key, predictions, manifest, and quality report.

    """
    return (
        json.loads(bundle_paths["rows"].read_text(encoding="utf-8")),
        json.loads(bundle_paths["answer_key"].read_text(encoding="utf-8")),
        json.loads(bundle_paths["predictions"].read_text(encoding="utf-8")),
        json.loads(bundle_paths["manifest"].read_text(encoding="utf-8")),
        json.loads(bundle_paths["quality"].read_text(encoding="utf-8")),
    )


def _write_bundle_payloads(
    bundle_paths: dict[str, Path],
    rows: list[dict[str, object]],
    answer_key: dict[str, object],
    predictions: dict[str, object],
    manifest: dict[str, object],
    quality: dict[str, object],
) -> None:
    """Rewrite bundle fixtures and refresh manifest hashes for tests.

    Args:
        bundle_paths: Mapping from logical bundle names to artifact paths.
        rows: Replacement private rows payload.
        answer_key: Replacement private answer key payload.
        predictions: Replacement calibration predictions payload.
        manifest: Replacement release manifest payload.
        quality: Replacement private quality report payload.

    Returns:
        None.

    """
    bundle_paths["rows"].write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    bundle_paths["answer_key"].write_text(json.dumps(answer_key, indent=2) + "\n", encoding="utf-8")
    bundle_paths["predictions"].write_text(json.dumps(predictions, indent=2) + "\n", encoding="utf-8")
    bundle_paths["quality"].write_text(json.dumps(quality, indent=2) + "\n", encoding="utf-8")
    manifest["sha256"]["private_leaderboard_rows.json"] = compute_sha256(bundle_paths["rows"])
    manifest["sha256"]["private_answer_key.json"] = compute_sha256(bundle_paths["answer_key"])
    manifest["sha256"][PRIVATE_CALIBRATION_PREDICTIONS_FILENAME] = compute_sha256(bundle_paths["predictions"])
    manifest["sha256"]["private_quality_report.json"] = compute_sha256(bundle_paths["quality"])
    bundle_paths["manifest"].write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


class CogflexVerificationTests(unittest.TestCase):
    def test_verify_public_split_reports_flexible_summary(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            verify_public_split()
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["split"], "public")
        self.assertEqual(payload["row_count"], 120)
        self.assertIn("structure_family_counts", payload)
        self.assertIn("turn_count_distribution", payload)
        self.assertIn("probe_count_distribution", payload)

    def test_verify_public_split_emits_audit_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, contextlib.redirect_stdout(io.StringIO()) as stdout:
            audit_path = Path(tmpdir) / "public-audit.json"
            verify_public_split(emit_audit_report=audit_path)
            audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["split"], "public")
        self.assertEqual(audit_payload["split"], "public")
        self.assertEqual(audit_payload["verification_result"], "passed")
        self.assertEqual(audit_payload["summary"], {"row_count": 120})
        self.assertEqual(
            audit_payload["artifact_digests"],
            {
                PUBLIC_ROWS_PATH.name: compute_sha256(PUBLIC_ROWS_PATH),
                PUBLIC_QUALITY_REPORT_PATH.name: compute_sha256(PUBLIC_QUALITY_REPORT_PATH),
                PUBLIC_DIFFICULTY_CALIBRATION_PATH.name: compute_sha256(PUBLIC_DIFFICULTY_CALIBRATION_PATH),
            },
        )
        self.assertEqual(
            audit_payload["checks_executed"],
            [
                "schema",
                "difficulty_calibration",
                "public_rows_reproducibility",
                "public_quality_report_reproducibility",
                "public_quality_report_consistency",
            ],
        )

    def test_attach_private_scoring_accepts_inference_only_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_paths = write_private_bundle(Path(tmpdir) / "bundle")
            private_rows = json.loads(bundle_paths["rows"].read_text(encoding="utf-8"))
            answer_key = load_private_answer_key(bundle_paths["answer_key"])
            attached = attach_private_scoring(private_rows[:3], {**answer_key, "episodes": answer_key["episodes"][:3]})
        self.assertEqual(len(attached), 3)
        self.assertIn("scoring", attached[0])
        self.assertEqual(
            len(attached[0]["scoring"]["final_probe_targets"]),
            attached[0]["inference"]["response_spec"]["probe_count"],
        )
        self.assertIn("probe_annotations", attached[0]["scoring"])
        self.assertEqual(
            len(attached[0]["scoring"]["probe_annotations"]),
            attached[0]["inference"]["response_spec"]["probe_count"],
        )

    def test_verify_private_bundle_accepts_valid_external_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, contextlib.redirect_stdout(io.StringIO()):
            bundle_dir = Path(tmpdir) / "bundle"
            write_private_bundle(bundle_dir)
            verify_private_bundle(bundle_dir)

    def test_verify_private_bundle_emits_safe_audit_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, contextlib.redirect_stdout(io.StringIO()):
            bundle_dir = Path(tmpdir) / "bundle"
            audit_path = Path(tmpdir) / "private-audit.json"
            bundle_paths = write_private_bundle(bundle_dir)
            verify_private_bundle(bundle_dir, emit_audit_report=audit_path)
            audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
            manifest = json.loads(bundle_paths["manifest"].read_text(encoding="utf-8"))
        self.assertEqual(audit_payload["split"], "private")
        self.assertEqual(audit_payload["verification_result"], "passed")
        self.assertEqual(audit_payload["summary"], {"row_count": 24})
        self.assertEqual(
            audit_payload["artifact_digests"],
            {
                PRIVATE_ROWS_FILENAME: manifest["sha256"][PRIVATE_ROWS_FILENAME],
                PRIVATE_ANSWER_KEY_FILENAME: manifest["sha256"][PRIVATE_ANSWER_KEY_FILENAME],
                PRIVATE_CALIBRATION_PREDICTIONS_FILENAME: manifest["sha256"][PRIVATE_CALIBRATION_PREDICTIONS_FILENAME],
                PRIVATE_QUALITY_REPORT_FILENAME: manifest["sha256"][PRIVATE_QUALITY_REPORT_FILENAME],
            },
        )
        self.assertEqual(
            audit_payload["checks_executed"],
            [
                "schema",
                "answer_key_consistency",
                "calibration_predictions",
                "empirical_difficulty",
                "release_manifest_digests",
                "private_quality_report_schema",
                "private_quality_report_reproducibility",
                "public_private_split_isolation",
                "generator_non_overlap",
                "required_structure_family_coverage",
            ],
        )
        audit_text = json.dumps(audit_payload, sort_keys=True)
        self.assertNotIn("answer_key_label_counts", audit_text)
        self.assertNotIn("generator_isolation_summary", audit_text)
        self.assertNotIn("template_ids", audit_text)
        self.assertNotIn("family_ids", audit_text)

    def test_private_bundle_quality_report_covers_all_required_structure_families(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_paths = write_private_bundle(Path(tmpdir) / "bundle")
            quality = json.loads(bundle_paths["quality"].read_text(encoding="utf-8"))
        self.assertEqual(
            quality["structure_family_counts"],
            {structure_family_id: 4 for structure_family_id in sorted(REQUIRED_PRIVATE_STRUCTURE_FAMILY_IDS)},
        )

    def test_verify_public_difficulty_calibration_rejects_mismatch(self) -> None:
        public_rows, _answers, _report = public_fixture()
        row = json.loads(json.dumps(public_rows[0]))
        row["analysis"]["difficulty_bin"] = "medium" if row["analysis"]["difficulty_bin"] == "hard" else "hard"
        with self.assertRaisesRegex(RuntimeError, "public difficulty calibration mismatch"):
            verify_public_difficulty_calibration([row, *public_rows[1:]])

    def test_verify_split_isolation_rejects_exact_overlap(self) -> None:
        public_rows, _answers, _report = public_fixture()
        private_row = json.loads(json.dumps(public_rows[0]))
        private_row["episode_id"] = "x9001"
        with self.assertRaisesRegex(RuntimeError, "semantic overlap"):
            verify_split_isolation(public_rows, [private_row])

    def test_verify_split_isolation_rejects_structural_overlap(self) -> None:
        public_rows, _answers, _report = public_fixture()
        private_row = json.loads(
            json.dumps(next(row for row in public_rows if row["analysis"]["suite_task_id"] == "context_binding"))
        )
        private_row["episode_id"] = "x9002"
        contexts = []
        for turn in private_row["inference"]["turns"]:
            for value in re.findall(r"context=([a-z_]+)", turn):
                if value not in contexts:
                    contexts.append(value)
        mapping = {contexts[0]: "mesa", contexts[1]: "fjord"}
        private_row["inference"]["turns"] = [
            re.sub(r"context=([a-z_]+)", lambda match: f"context={mapping.get(match.group(1), match.group(1))}", turn)
            for turn in private_row["inference"]["turns"]
        ]
        with self.assertRaisesRegex(RuntimeError, "structural overlap"):
            verify_split_isolation(public_rows, [private_row])

    def test_verify_split_isolation_rejects_near_duplicate_overlap(self) -> None:
        public_rows, _answers, _report = public_fixture()
        private_row = json.loads(json.dumps(public_rows[2]))
        private_row["episode_id"] = "x9003"
        private_row["analysis"]["structure_family_id"] = "private_clone"
        private_row["inference"]["turns"][0] = private_row["inference"]["turns"][0].replace(
            "Infer the current rule from these labeled examples.",
            "Infer the current rule from these labeled examples. Keep track of the cadence too.",
            1,
        )
        self.assertGreaterEqual(structural_overlap_score(public_rows[2], private_row), 0.9)
        with self.assertRaisesRegex(RuntimeError, "near-duplicate overlap"):
            verify_split_isolation(public_rows, [private_row])

    def test_verify_schema_rejects_turn_spec_count_mismatch(self) -> None:
        public_rows, _answers, _report = public_fixture()
        row = json.loads(json.dumps(public_rows[0]))
        row["inference"]["turn_specs"][0]["item_count"] += 1
        with self.assertRaisesRegex(RuntimeError, "count does not match turn_specs"):
            verify_schema([row], "public")

    def test_verify_schema_rejects_label_outside_vocab(self) -> None:
        public_rows, _answers, _report = public_fixture()
        row = json.loads(json.dumps(public_rows[0]))
        row["scoring"]["final_probe_targets"][0] = "not_in_vocab"
        with self.assertRaisesRegex(RuntimeError, "invalid final_probe_targets"):
            verify_schema([row], "public")

    def test_verify_private_bundle_rejects_missing_quality_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_dir = Path(tmpdir) / "bundle"
            bundle_paths = write_private_bundle(bundle_dir)
            bundle_paths["quality"].unlink()
            with self.assertRaisesRegex(RuntimeError, "missing required files"):
                verify_private_bundle(bundle_dir)

    def test_verify_private_bundle_rejects_missing_predictions_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_dir = Path(tmpdir) / "bundle"
            bundle_paths = write_private_bundle(bundle_dir)
            bundle_paths["predictions"].unlink()
            with self.assertRaisesRegex(RuntimeError, "missing required files"):
                verify_private_bundle(bundle_dir)

    def test_verify_private_answer_key_rejects_analysis_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_paths = write_private_bundle(Path(tmpdir) / "bundle")
            private_rows = json.loads(bundle_paths["rows"].read_text(encoding="utf-8"))
            answer_key = load_private_answer_key(bundle_paths["answer_key"])
            answer_key["episodes"][0]["difficulty_bin"] = "mismatch"
            with self.assertRaisesRegex(RuntimeError, "difficulty_bin mismatch"):
                verify_private_answer_key(answer_key, private_rows)

    def test_verify_private_answer_key_rejects_missing_generator_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_paths = write_private_bundle(Path(tmpdir) / "bundle")
            private_rows = json.loads(bundle_paths["rows"].read_text(encoding="utf-8"))
            answer_key = load_private_answer_key(bundle_paths["answer_key"])
            answer_key["episodes"][0].pop("generator")
            with self.assertRaisesRegex(RuntimeError, "must include generator metadata"):
                verify_private_answer_key(answer_key, private_rows)

    def test_verify_private_answer_key_rejects_unknown_operator_class(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_paths = write_private_bundle(Path(tmpdir) / "bundle")
            private_rows = json.loads(bundle_paths["rows"].read_text(encoding="utf-8"))
            answer_key = load_private_answer_key(bundle_paths["answer_key"])
            answer_key["episodes"][0]["generator"]["operator_class"] = "unknown_operator"
            with self.assertRaisesRegex(RuntimeError, "unsupported operator_class"):
                verify_private_answer_key(answer_key, private_rows)

    def test_verify_manifest_rejects_digest_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_paths = write_private_bundle(Path(tmpdir) / "bundle")
            payload = json.loads(bundle_paths["manifest"].read_text(encoding="utf-8"))
            payload["version"] = PRIVATE_BUNDLE_VERSION
            payload["sha256"]["private_leaderboard_rows.json"] = "0" * 64
            bundle_paths["manifest"].write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "digest mismatch"):
                verify_manifest(bundle_paths["manifest"], bundle_paths)

    def test_verify_private_calibration_predictions_rejects_partial_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_paths = write_private_bundle(Path(tmpdir) / "bundle")
            private_rows = json.loads(bundle_paths["rows"].read_text(encoding="utf-8"))
            answer_key = load_private_answer_key(bundle_paths["answer_key"])
            _summary, episode_targets, _episode_generators, _episode_annotations = verify_private_answer_key(answer_key, private_rows)
            predictions = load_private_calibration_predictions(bundle_paths["predictions"])
            predictions["models"][0]["episodes"] = predictions["models"][0]["episodes"][:-1]
            with self.assertRaisesRegex(RuntimeError, "missing episode_ids"):
                verify_private_calibration_predictions(predictions, private_rows, episode_targets)

    def test_verify_private_calibration_predictions_rejects_invalid_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_paths = write_private_bundle(Path(tmpdir) / "bundle")
            private_rows = json.loads(bundle_paths["rows"].read_text(encoding="utf-8"))
            answer_key = load_private_answer_key(bundle_paths["answer_key"])
            _summary, episode_targets, _episode_generators, _episode_annotations = verify_private_answer_key(answer_key, private_rows)
            predictions = load_private_calibration_predictions(bundle_paths["predictions"])
            predictions["models"][0]["episodes"][0]["predicted_labels"][0] = "not_in_vocab"
            with self.assertRaisesRegex(RuntimeError, "invalid predicted_labels"):
                verify_private_calibration_predictions(predictions, private_rows, episode_targets)

    def test_verify_quality_report_requires_three_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_paths = write_private_bundle(Path(tmpdir) / "bundle")
            payload = json.loads(bundle_paths["quality"].read_text(encoding="utf-8"))
            payload["version"] = PRIVATE_QUALITY_REPORT_VERSION
            payload["calibration_summary"]["models"] = payload["calibration_summary"]["models"][:2]
            bundle_paths["quality"].write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "exactly 3 models"):
                verify_quality_report(bundle_paths["quality"])

    def test_verify_quality_report_requires_generator_isolation_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_paths = write_private_bundle(Path(tmpdir) / "bundle")
            payload = json.loads(bundle_paths["quality"].read_text(encoding="utf-8"))
            payload["version"] = PRIVATE_QUALITY_REPORT_VERSION
            payload.pop("generator_isolation_summary")
            bundle_paths["quality"].write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "generator_isolation_summary"):
                verify_quality_report(bundle_paths["quality"])

    def test_verify_private_bundle_rejects_calibration_summary_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_paths = write_private_bundle(Path(tmpdir) / "bundle")
            payload = json.loads(bundle_paths["quality"].read_text(encoding="utf-8"))
            payload["calibration_summary"]["models"][0]["micro_accuracy"] = 0.0
            bundle_paths["quality"].write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            manifest = json.loads(bundle_paths["manifest"].read_text(encoding="utf-8"))
            manifest["sha256"]["private_quality_report.json"] = compute_sha256(bundle_paths["quality"])
            bundle_paths["manifest"].write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "calibration_summary mismatch"):
                verify_private_bundle(Path(tmpdir) / "bundle")

    def test_verify_private_bundle_rejects_empirical_difficulty_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_paths = write_private_bundle(Path(tmpdir) / "bundle")
            rows, answer_key, predictions, manifest, quality = _load_bundle_payloads(bundle_paths)
            replacement = "medium" if rows[0]["analysis"]["difficulty_bin"] == "hard" else "hard"
            rows[0]["analysis"]["difficulty_bin"] = replacement
            answer_key["episodes"][0]["difficulty_bin"] = replacement
            _write_bundle_payloads(bundle_paths, rows, answer_key, predictions, manifest, quality)
            with self.assertRaisesRegex(RuntimeError, "private empirical difficulty mismatch"):
                verify_private_bundle(Path(tmpdir) / "bundle")

    def test_verify_private_bundle_rejects_attack_suite_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_paths = write_private_bundle(Path(tmpdir) / "bundle")
            payload = json.loads(bundle_paths["quality"].read_text(encoding="utf-8"))
            payload["attack_suite"]["difficulty_bin"]["hard"]["models"][0]["macro_accuracy"] = 0.0
            bundle_paths["quality"].write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            manifest = json.loads(bundle_paths["manifest"].read_text(encoding="utf-8"))
            manifest["sha256"]["private_quality_report.json"] = compute_sha256(bundle_paths["quality"])
            bundle_paths["manifest"].write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "attack_suite mismatch"):
                verify_private_bundle(Path(tmpdir) / "bundle")

    def test_build_private_quality_report_uses_present_tasks_for_attack_suite_macro(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_paths = write_private_bundle(Path(tmpdir) / "bundle")
            rows = json.loads(bundle_paths["rows"].read_text(encoding="utf-8"))
            answer_key = json.loads(bundle_paths["answer_key"].read_text(encoding="utf-8"))
            predictions = json.loads(bundle_paths["predictions"].read_text(encoding="utf-8"))

            quality = build_private_quality_report(rows, answer_key, predictions, public_rows=public_fixture()[0])

            suite_task_summary = quality["attack_suite"]["suite_task_id"]["explicit_rule_update"]["models"][0]
            self.assertAlmostEqual(suite_task_summary["micro_accuracy"], 0.8)
            self.assertAlmostEqual(suite_task_summary["macro_accuracy"], 0.8)
            self.assertEqual(
                suite_task_summary["per_task_accuracy"],
                {
                    "explicit_rule_update": 0.8,
                    "latent_rule_update": 0.0,
                    "context_binding": 0.0,
                    "trial_cued_switch": 0.0,
                },
            )

            shift_mode_summary = quality["attack_suite"]["shift_mode"]["explicit_instruction"]["models"][0]
            self.assertAlmostEqual(shift_mode_summary["micro_accuracy"], 0.8)
            self.assertAlmostEqual(shift_mode_summary["macro_accuracy"], 0.8)
            self.assertEqual(shift_mode_summary["per_task_accuracy"], suite_task_summary["per_task_accuracy"])

    def test_build_private_quality_report_attack_suite_macro_ignores_absent_tasks_in_multi_task_slice(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_paths = write_private_bundle(Path(tmpdir) / "bundle")
            rows, answer_key, predictions, _manifest, _quality = _load_bundle_payloads(bundle_paths)
            keep_tasks = {"explicit_rule_update", "latent_rule_update"}
            rows = [row for row in rows if row["analysis"]["suite_task_id"] in keep_tasks]
            remaining_episode_ids = {row["episode_id"] for row in rows}
            answer_key["episodes"] = [
                episode for episode in answer_key["episodes"] if episode["episode_id"] in remaining_episode_ids
            ]
            for model in predictions["models"]:
                model["episodes"] = [
                    episode
                    for episode in model["episodes"]
                    if episode["episode_id"] in remaining_episode_ids
                ]
            normalized_models = [
                {
                    "name": str(model["name"]),
                    "episodes": {
                        str(episode["episode_id"]): tuple(str(label) for label in episode["predicted_labels"])
                        for episode in model["episodes"]
                    },
                }
                for model in predictions["models"]
            ]
            episode_targets = {
                str(episode["episode_id"]): tuple(str(label) for label in episode["final_probe_targets"])
                for episode in answer_key["episodes"]
            }
            difficulty_entries = empirical_difficulty_entries_from_predictions(episode_targets, normalized_models)
            for row in rows:
                row["analysis"]["difficulty_bin"] = str(difficulty_entries[str(row["episode_id"])]["difficulty_bin"])
            for episode in answer_key["episodes"]:
                episode["difficulty_bin"] = str(difficulty_entries[str(episode["episode_id"])]["difficulty_bin"])

            quality = build_private_quality_report(rows, answer_key, predictions, public_rows=public_fixture()[0])

            hard_summary = quality["attack_suite"]["difficulty_bin"]["hard"]["models"][0]
            self.assertAlmostEqual(hard_summary["micro_accuracy"], 0.764706)
            self.assertEqual(
                hard_summary["per_task_accuracy"],
                {
                    "explicit_rule_update": 0.785714,
                    "latent_rule_update": 0.666667,
                    "context_binding": 0.0,
                    "trial_cued_switch": 0.0,
                },
            )
            self.assertAlmostEqual(hard_summary["macro_accuracy"], 0.72619)

    def test_verify_private_bundle_rejects_semantic_isolation_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_paths = write_private_bundle(Path(tmpdir) / "bundle")
            payload = json.loads(bundle_paths["quality"].read_text(encoding="utf-8"))
            payload["semantic_isolation_summary"]["near_duplicate_overlap_count"] = 1
            bundle_paths["quality"].write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            manifest = json.loads(bundle_paths["manifest"].read_text(encoding="utf-8"))
            manifest["sha256"]["private_quality_report.json"] = compute_sha256(bundle_paths["quality"])
            bundle_paths["manifest"].write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "semantic_isolation_summary mismatch"):
                verify_private_bundle(Path(tmpdir) / "bundle")

    def test_verify_private_bundle_rejects_generator_isolation_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_paths = write_private_bundle(Path(tmpdir) / "bundle")
            payload = json.loads(bundle_paths["quality"].read_text(encoding="utf-8"))
            payload["generator_isolation_summary"]["family_ids"].append("tampered_family")
            bundle_paths["quality"].write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            manifest = json.loads(bundle_paths["manifest"].read_text(encoding="utf-8"))
            manifest["sha256"]["private_quality_report.json"] = compute_sha256(bundle_paths["quality"])
            bundle_paths["manifest"].write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "generator_isolation_summary mismatch"):
                verify_private_bundle(Path(tmpdir) / "bundle")

    def test_verify_private_bundle_rejects_generator_family_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_paths = write_private_bundle(Path(tmpdir) / "bundle")
            rows, answer_key, predictions, manifest, quality = _load_bundle_payloads(bundle_paths)
            answer_key["episodes"][0]["generator"]["family_id"] = public_generator_reference()["family_ids"][0]
            _write_bundle_payloads(bundle_paths, rows, answer_key, predictions, manifest, quality)
            with self.assertRaisesRegex(RuntimeError, "generator family_id overlap"):
                verify_private_bundle(Path(tmpdir) / "bundle")

    def test_verify_private_bundle_rejects_generator_template_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_paths = write_private_bundle(Path(tmpdir) / "bundle")
            rows, answer_key, predictions, manifest, quality = _load_bundle_payloads(bundle_paths)
            answer_key["episodes"][0]["generator"]["template_id"] = public_generator_reference()["template_ids"][0]
            _write_bundle_payloads(bundle_paths, rows, answer_key, predictions, manifest, quality)
            with self.assertRaisesRegex(RuntimeError, "generator template_id overlap"):
                verify_private_bundle(Path(tmpdir) / "bundle")

    def test_verify_private_bundle_rejects_generator_operator_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_paths = write_private_bundle(Path(tmpdir) / "bundle")
            rows, answer_key, predictions, manifest, quality = _load_bundle_payloads(bundle_paths)
            answer_key["episodes"][0]["generator"]["operator_class"] = public_generator_reference()["operator_classes"][0]
            _write_bundle_payloads(bundle_paths, rows, answer_key, predictions, manifest, quality)
            with self.assertRaisesRegex(RuntimeError, "generator operator_class overlap"):
                verify_private_bundle(Path(tmpdir) / "bundle")

    def test_verify_private_bundle_rejects_missing_structure_family(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_paths = write_private_bundle(Path(tmpdir) / "bundle")
            rows, answer_key, predictions, manifest, quality = _load_bundle_payloads(bundle_paths)
            target_family = "interleaved_context_rebinding"
            rows = [row for row in rows if row["analysis"]["structure_family_id"] != target_family]
            answer_key["episodes"] = [
                episode for episode in answer_key["episodes"] if episode["structure_family_id"] != target_family
            ]
            remaining_episode_ids = {episode["episode_id"] for episode in answer_key["episodes"]}
            for model in predictions["models"]:
                model["episodes"] = [
                    episode
                    for episode in model["episodes"]
                    if episode["episode_id"] in remaining_episode_ids
                ]
            normalized_models = [
                {
                    "name": str(model["name"]),
                    "episodes": {
                        str(episode["episode_id"]): tuple(str(label) for label in episode["predicted_labels"])
                        for episode in model["episodes"]
                    },
                }
                for model in predictions["models"]
            ]
            episode_targets = {
                str(episode["episode_id"]): tuple(str(label) for label in episode["final_probe_targets"])
                for episode in answer_key["episodes"]
            }
            difficulty_entries = empirical_difficulty_entries_from_predictions(episode_targets, normalized_models)
            for row in rows:
                row["analysis"]["difficulty_bin"] = str(difficulty_entries[str(row["episode_id"])]["difficulty_bin"])
            for episode in answer_key["episodes"]:
                episode["difficulty_bin"] = str(difficulty_entries[str(episode["episode_id"])]["difficulty_bin"])
            summary = _private_row_summary(rows)
            quality = build_private_quality_report(rows, answer_key, predictions, public_rows=public_fixture()[0])
            quality["row_count"] = len(rows)
            quality["difficulty_bin_counts"] = summary["difficulty_bin_counts"]
            quality["structure_family_counts"] = summary["structure_family_counts"]
            quality["turn_count_distribution"] = summary["turn_count_distribution"]
            quality["probe_count_distribution"] = summary["probe_count_distribution"]
            quality["label_vocab_size_distribution"] = summary["label_vocab_size_distribution"]
            quality["stimulus_space_summary"] = summary["stimulus_space_summary"]
            _write_bundle_payloads(bundle_paths, rows, answer_key, predictions, manifest, quality)
            with self.assertRaisesRegex(RuntimeError, "missing required structure families"):
                verify_private_bundle(Path(tmpdir) / "bundle")
