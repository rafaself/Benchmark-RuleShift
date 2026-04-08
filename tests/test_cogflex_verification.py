import contextlib
import io
import json
import re
import tempfile
import unittest
from pathlib import Path
from typing import Callable

import scripts.build_cogflex_dataset as build_mod
from cogflex_fixtures import _private_turns, write_private_bundle
from scripts.build_cogflex_dataset import (
    PRIVATE_QUALITY_REPORT_VERSION,
    PRIVATE_RELEASE_MANIFEST_FILENAME,
    PRIVATE_ROWS_FILENAME,
    PRIVATE_ANSWER_KEY_FILENAME,
    PRIVATE_QUALITY_REPORT_FILENAME,
    PRIVATE_BUNDLE_VERSION,
    build_public_artifacts,
    compute_sha256,
)
from scripts.verify_cogflex import (
    attach_private_scoring,
    load_private_answer_key,
    structural_overlap_score,
    verify_public_attack_suite,
    verify_public_diversity_summary,
    verify_manifest,
    verify_private_answer_key,
    verify_private_bundle,
    verify_public_split,
    verify_public_switching_summary,
    verify_quality_report,
    verify_split_isolation,
    verify_switching_requirements,
)


def _load_bundle_payloads(bundle_paths: dict[str, Path]) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    return (
        json.loads(bundle_paths["rows"].read_text(encoding="utf-8")),
        json.loads(bundle_paths["answer_key"].read_text(encoding="utf-8")),
        json.loads(bundle_paths["manifest"].read_text(encoding="utf-8")),
    )


def _write_bundle_payloads(
    bundle_paths: dict[str, Path],
    rows: list[dict[str, object]],
    answer_key: dict[str, object],
    manifest: dict[str, object],
) -> None:
    bundle_paths["rows"].write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    bundle_paths["answer_key"].write_text(json.dumps(answer_key, indent=2) + "\n", encoding="utf-8")
    manifest["sha256"]["private_leaderboard_rows.json"] = compute_sha256(bundle_paths["rows"])
    manifest["sha256"]["private_answer_key.json"] = compute_sha256(bundle_paths["answer_key"])
    bundle_paths["manifest"].write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _remap_shapes(turn: str) -> str:
    remapped = turn.replace("shape=circle", "shape=__circle__")
    remapped = remapped.replace("shape=triangle", "shape=circle")
    remapped = remapped.replace("shape=square", "shape=triangle")
    return remapped.replace("shape=__circle__", "shape=square")


def _make_valid_private_bundle(bundle_dir: Path) -> dict[str, Path]:
    public_rows, _public_answers, _public_report = build_public_artifacts()
    original_derive_seed = build_mod.derive_seed
    alternate_rows: list[dict[str, object]] | None = None
    for salt in ("private_fixture_a", "private_fixture_b", "private_fixture_c"):
        build_mod.derive_seed = lambda *parts, _salt=salt: original_derive_seed(*parts, _salt)
        try:
            candidate_rows, _candidate_answers, _candidate_report = build_mod.build_public_artifacts()
        finally:
            build_mod.derive_seed = original_derive_seed
        try:
            verify_split_isolation(public_rows, candidate_rows)
        except RuntimeError:
            continue
        alternate_rows = candidate_rows
        break
    if alternate_rows is None:
        raise RuntimeError("unable to build structurally isolated private fixture rows")

    bundle_dir.mkdir(parents=True, exist_ok=True)
    private_rows: list[dict[str, object]] = []
    answer_episodes: list[dict[str, object]] = []
    episode_counter = 1
    for replica in range(4):
        for row in alternate_rows:
            episode_id = f"{episode_counter:04d}"
            marker = f"replica_{replica + 1}_{episode_id}"
            turns = _private_turns(list(row["inference"]["turns"]), marker)
            for turn_index, turn in enumerate(turns, start=1):
                old = f"Episode {row['episode_id']}. Turn {turn_index} of 3."
                new = f"Episode {episode_id}. Turn {turn_index} of 3."
                turns[turn_index - 1] = turn.replace(old, new)
            private_rows.append(
                {
                    "episode_id": episode_id,
                    "inference": {"turns": turns},
                    "analysis": dict(row["analysis"]),
                }
            )
            answer_episodes.append(
                {
                    "episode_id": episode_id,
                    "faculty_id": row["analysis"]["faculty_id"],
                    "suite_task_id": row["analysis"]["suite_task_id"],
                    "shift_mode": row["analysis"]["shift_mode"],
                    "difficulty_bin": row["analysis"]["difficulty_bin"],
                    "turns": turns,
                    "final_probe_targets": list(row["scoring"]["final_probe_targets"]),
                }
            )
            episode_counter += 1

    rows_path = bundle_dir / PRIVATE_ROWS_FILENAME
    answer_key_path = bundle_dir / PRIVATE_ANSWER_KEY_FILENAME
    quality_path = bundle_dir / PRIVATE_QUALITY_REPORT_FILENAME
    manifest_path = bundle_dir / PRIVATE_RELEASE_MANIFEST_FILENAME

    rows_path.write_text(json.dumps(private_rows, indent=2) + "\n", encoding="utf-8")
    answer_key_path.write_text(
        json.dumps(
            {
                "version": "cogflex_private_answer_key_v1",
                "split": "private",
                "episodes": answer_episodes,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    quality_path.write_text(
        json.dumps(
            {
                "version": PRIVATE_QUALITY_REPORT_VERSION,
                "split": "private",
                "row_count": 480,
                "episodes_per_task": 120,
                "difficulty_bin_counts": {"hard": 240, "medium": 240},
                "attack_suite": {
                    "dsl_search_accuracy": {
                        "micro_accuracy": 0.55,
                        "per_task_accuracy": {
                            "explicit_rule_update": 0.56,
                            "latent_rule_update": 0.54,
                            "context_binding": 0.55,
                            "trial_cued_switch": 0.55,
                        },
                    }
                },
                "calibration_summary": {
                    "models": [
                        {"name": "panel-model-a", "macro_accuracy": 0.61, "micro_accuracy": 0.60},
                        {"name": "panel-model-b", "macro_accuracy": 0.57, "micro_accuracy": 0.56},
                        {"name": "panel-model-c", "macro_accuracy": 0.52, "micro_accuracy": 0.51},
                    ]
                },
                "semantic_isolation_summary": {
                    "exact_public_overlap_count": 0,
                    "lexicon_overlap_count": 0,
                },
                "public_generator_commit_sha": "0" * 40,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "version": PRIVATE_BUNDLE_VERSION,
        "split": "private",
        "row_count": 480,
        "episodes_per_task": 120,
        "public_generator_commit_sha": "0" * 40,
        "lexicons": {
            "cue_terms": ["quartz", "sable", "lumen", "cinder"],
            "context_terms": ["mesa", "fjord", "delta", "tundra"],
        },
        "sha256": {
            PRIVATE_ROWS_FILENAME: compute_sha256(rows_path),
            PRIVATE_ANSWER_KEY_FILENAME: compute_sha256(answer_key_path),
            PRIVATE_QUALITY_REPORT_FILENAME: compute_sha256(quality_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "rows": rows_path,
        "answer_key": answer_key_path,
        "quality": quality_path,
        "manifest": manifest_path,
    }


def _replace_episode_from_public(
    bundle_paths: dict[str, Path],
    public_row: dict[str, object],
    *,
    episode_index: int | None = None,
    turn_transform: Callable[[list[str]], list[str]] | None = None,
    final_probe_targets: list[str] | None = None,
) -> None:
    rows, answer_key, manifest = _load_bundle_payloads(bundle_paths)
    if episode_index is None:
        episode_index = next(
            index
            for index, row in enumerate(rows)
            if row["analysis"]["suite_task_id"] == public_row["analysis"]["suite_task_id"]
        )
    private_episode_id = str(rows[episode_index]["episode_id"])
    turns = list(public_row["inference"]["turns"])
    for turn_index, turn in enumerate(turns, start=1):
        old = f"Episode {public_row['episode_id']}. Turn {turn_index} of 3."
        new = f"Episode {private_episode_id}. Turn {turn_index} of 3."
        turns[turn_index - 1] = turn.replace(old, new)
    if turn_transform is not None:
        turns = turn_transform(turns)
    rows[episode_index]["analysis"] = dict(public_row["analysis"])
    rows[episode_index]["inference"]["turns"] = turns
    answer = answer_key["episodes"][episode_index]
    answer["faculty_id"] = public_row["analysis"]["faculty_id"]
    answer["suite_task_id"] = public_row["analysis"]["suite_task_id"]
    answer["shift_mode"] = public_row["analysis"]["shift_mode"]
    answer["difficulty_bin"] = public_row["analysis"]["difficulty_bin"]
    answer["turns"] = list(turns)
    answer["final_probe_targets"] = list(final_probe_targets or public_row["scoring"]["final_probe_targets"])
    _write_bundle_payloads(bundle_paths, rows, answer_key, manifest)


def _select_public_row(suite_task_id: str) -> dict[str, object]:
    rows, _answers, _report = build_public_artifacts()
    return next(row for row in rows if row["analysis"]["suite_task_id"] == suite_task_id)


def _rename_turn_attribute_values(turns: list[str], attribute: str, replacement_values: list[str]) -> list[str]:
    pattern = re.compile(rf"{attribute}=([a-z_]+)")
    discovered: list[str] = []
    joined = "\n".join(turns)
    for value in pattern.findall(joined):
        if value not in discovered:
            discovered.append(value)
    mapping = {value: replacement_values[index] for index, value in enumerate(discovered)}
    renamed_turns: list[str] = []
    for turn in turns:
        renamed_turns.append(pattern.sub(lambda match: f"{attribute}={mapping[match.group(1)]}", turn))
    return renamed_turns


class CogflexVerificationTests(unittest.TestCase):
    def test_verify_public_reports_attack_suite_summary(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            verify_public_split()
        payload = json.loads(stdout.getvalue())
        self.assertIn("attack_suite", payload)
        self.assertIn("switching_diagnostics", payload)
        self.assertIn("transition_family_count", payload)
        self.assertIn("suite_task_counts", payload)

    def test_verify_public_attack_suite_rejects_strong_majority_label_shortcut(self) -> None:
        payload = {
            "attack_suite": {
                "majority_label_accuracy": {
                    "micro_accuracy": 0.5001,
                    "per_task_accuracy": {
                        "explicit_rule_update": 0.5,
                        "latent_rule_update": 0.5,
                        "context_binding": 0.5,
                        "trial_cued_switch": 0.5,
                    },
                },
                "previous_rule_accuracy": {
                    "micro_accuracy": 0.25,
                    "per_task_accuracy": {
                        "explicit_rule_update": 0.0,
                        "latent_rule_update": 0.0,
                        "context_binding": 0.25,
                        "trial_cued_switch": 0.25,
                    },
                },
                "cue_agnostic_accuracy": {
                    "micro_accuracy": 0.5,
                    "per_task_accuracy": {
                        "explicit_rule_update": None,
                        "latent_rule_update": None,
                        "context_binding": 0.5,
                        "trial_cued_switch": 0.5,
                    },
                },
            }
        }
        with self.assertRaisesRegex(RuntimeError, "majority_label_accuracy micro_accuracy exceeds ceiling"):
            verify_public_attack_suite(payload)

    def test_verify_public_diversity_summary_rejects_low_pair_pattern_count(self) -> None:
        payload = {
            "rule_inventory_count": 28,
            "rule_family_count": 8,
            "rule_family_rule_counts": {
                "axis_threshold": 4,
                "cross_feature_binding": 4,
                "feature_gate": 3,
                "magnitude": 4,
                "numeric_binding": 4,
                "parity": 3,
                "relational": 3,
                "sign": 3,
            },
            "transition_family_count": 15,
            "selected_initial_family_count": 5,
            "selected_initial_family_usage": {f"family_{index}": 8 for index in range(5)},
            "selected_shift_family_count": 6,
            "selected_shift_family_usage": {f"family_{index}": 8 for index in range(6)},
            "transition_pair_pattern_count": 14,
            "transition_pair_usage": {f"pair_{index}": 8 for index in range(14)},
        }
        with self.assertRaisesRegex(RuntimeError, "transition_pair_pattern_count is too low"):
            verify_public_diversity_summary(payload)

    def test_verify_public_switching_summary_rejects_learn_only_shortcut(self) -> None:
        payload = {
            "switching_diagnostics": {
                "learn_only_max_probe_accuracy": 0.6251,
                "post_shift_prediction_set_size_distribution": {"1": 120},
                "turn2_required_probe_count_distribution": {"4": 60, "8": 60},
            }
        }
        with self.assertRaisesRegex(RuntimeError, "learn_only_max_probe_accuracy exceeds ceiling"):
            verify_public_switching_summary(payload)

    def test_attach_private_scoring_accepts_inference_only_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_paths = write_private_bundle(Path(tmpdir) / "bundle")
            private_rows = json.loads(bundle_paths["rows"].read_text(encoding="utf-8"))
            answer_key = load_private_answer_key(bundle_paths["answer_key"])
            attached = attach_private_scoring(private_rows[:3], {
                **answer_key,
                "episodes": answer_key["episodes"][:3],
            })
        self.assertEqual(len(attached), 3)
        self.assertIn("scoring", attached[0])
        self.assertEqual(len(attached[0]["scoring"]["final_probe_targets"]), 8)

    def test_verify_private_bundle_accepts_valid_external_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, contextlib.redirect_stdout(io.StringIO()):
            bundle_dir = Path(tmpdir) / "bundle"
            _make_valid_private_bundle(bundle_dir)
            verify_private_bundle(bundle_dir)

    def test_verify_split_isolation_rejects_exact_overlap(self) -> None:
        public_rows, _answers, _report = build_public_artifacts()
        private_row = json.loads(json.dumps(public_rows[0]))
        private_row["episode_id"] = "9001"
        with self.assertRaisesRegex(RuntimeError, "semantic overlap"):
            verify_split_isolation(public_rows, [private_row])

    def test_verify_split_isolation_rejects_near_duplicate_overlap(self) -> None:
        public_rows, _answers, _report = build_public_artifacts()
        private_row = json.loads(json.dumps(_select_public_row("explicit_rule_update")))
        private_row["episode_id"] = "9002"
        mutated_turn = re.sub(r"shape=(circle|triangle|square)", "shape=square", private_row["inference"]["turns"][0], count=1)
        private_row["inference"]["turns"][0] = mutated_turn
        self.assertGreaterEqual(structural_overlap_score(_select_public_row("explicit_rule_update"), private_row), 0.95)
        with self.assertRaisesRegex(RuntimeError, "near-duplicate overlap"):
            verify_split_isolation(public_rows, [private_row])

    def test_verify_split_isolation_rejects_structural_isomorphism(self) -> None:
        public_rows, _answers, _report = build_public_artifacts()
        private_row = json.loads(json.dumps(_select_public_row("context_binding")))
        private_row["episode_id"] = "9003"
        private_row["inference"]["turns"] = _rename_turn_attribute_values(
            private_row["inference"]["turns"],
            "context",
            ["mesa", "fjord"],
        )
        with self.assertRaisesRegex(RuntimeError, "structural overlap"):
            verify_split_isolation(public_rows, [private_row])

    def test_verify_private_bundle_rejects_exact_public_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_paths = _make_valid_private_bundle(Path(tmpdir) / "bundle")
            public_row = _select_public_row("explicit_rule_update")
            _replace_episode_from_public(bundle_paths, public_row)
            with self.assertRaisesRegex(RuntimeError, "semantic overlap"):
                verify_private_bundle(Path(tmpdir) / "bundle")

    def test_verify_private_bundle_rejects_near_duplicate_public_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_paths = _make_valid_private_bundle(Path(tmpdir) / "bundle")
            public_row = _select_public_row("explicit_rule_update")

            def mutate(turns: list[str]) -> list[str]:
                turns = list(turns)
                turns[0] = re.sub(r"shape=circle", "shape=square", turns[0], count=1)
                if turns[0] == public_row["inference"]["turns"][0]:
                    turns[0] = re.sub(r"shape=triangle", "shape=circle", turns[0], count=1)
                if turns[0] == public_row["inference"]["turns"][0]:
                    turns[0] = re.sub(r"shape=square", "shape=triangle", turns[0], count=1)
                return turns

            _replace_episode_from_public(bundle_paths, public_row, turn_transform=mutate)
            with self.assertRaisesRegex(RuntimeError, "near-duplicate overlap"):
                verify_private_bundle(Path(tmpdir) / "bundle")

    def test_verify_private_bundle_rejects_structural_isomorphic_public_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_paths = _make_valid_private_bundle(Path(tmpdir) / "bundle")
            public_row = _select_public_row("context_binding")

            def rename_contexts(turns: list[str]) -> list[str]:
                return _rename_turn_attribute_values(turns, "context", ["mesa", "fjord"])

            _replace_episode_from_public(bundle_paths, public_row, turn_transform=rename_contexts)
            with self.assertRaisesRegex(RuntimeError, "structural overlap"):
                verify_private_bundle(Path(tmpdir) / "bundle")

    def test_verify_switching_requirements_rejects_turn1_solvable_row(self) -> None:
        public_rows, _answers, _report = build_public_artifacts()
        row = json.loads(json.dumps(_select_public_row("explicit_rule_update")))
        learn_turn = row["inference"]["turns"][0]
        label_map = {int(match.group(1)): match.group(3) for match in re.finditer(r"(\d+)\.\s+(.+?)\s+->\s+(type_a|type_b)", learn_turn)}
        replacement = [label_map[index] for index in sorted(label_map)]
        row["scoring"]["final_probe_targets"] = replacement + replacement[:2]
        with self.assertRaisesRegex(RuntimeError, "too solvable from turn 1 alone"):
            verify_switching_requirements([row])

    def test_verify_private_bundle_rejects_missing_quality_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_dir = Path(tmpdir) / "bundle"
            bundle_paths = write_private_bundle(bundle_dir)
            bundle_paths["quality"].unlink()
            with self.assertRaisesRegex(RuntimeError, "missing required files"):
                verify_private_bundle(bundle_dir)

    def test_verify_private_answer_key_rejects_analysis_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_dir = Path(tmpdir) / "bundle"
            bundle_paths = write_private_bundle(bundle_dir)
            private_rows = json.loads(bundle_paths["rows"].read_text(encoding="utf-8"))
            answer_key = load_private_answer_key(bundle_paths["answer_key"])
            answer_key["episodes"][0]["difficulty_bin"] = "mismatch"
            with self.assertRaisesRegex(RuntimeError, "difficulty_bin mismatch"):
                verify_private_answer_key(answer_key, private_rows)

    def test_verify_manifest_rejects_digest_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_dir = Path(tmpdir) / "bundle"
            bundle_paths = write_private_bundle(bundle_dir)
            payload = json.loads(bundle_paths["manifest"].read_text(encoding="utf-8"))
            payload["sha256"]["private_leaderboard_rows.json"] = "0" * 64
            bundle_paths["manifest"].write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "digest mismatch"):
                verify_manifest(bundle_paths["manifest"], bundle_paths)

    def test_verify_quality_report_requires_three_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_dir = Path(tmpdir) / "bundle"
            bundle_paths = write_private_bundle(bundle_dir)
            payload = json.loads(bundle_paths["quality"].read_text(encoding="utf-8"))
            payload["calibration_summary"]["models"] = payload["calibration_summary"]["models"][:2]
            bundle_paths["quality"].write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "exactly 3 models"):
                verify_quality_report(bundle_paths["quality"])
