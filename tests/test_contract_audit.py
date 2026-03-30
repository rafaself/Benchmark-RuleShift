"""Tests for the P0 contract audit.

Validates the contract audit against known-good and known-bad artifact shapes.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from maintainer.contract_audit import (
    CANONICAL_BINARY_TASK_NAME,
    CANONICAL_NARRATIVE_TASK_NAME,
    CANONICAL_TASK_ID,
    CANONICAL_TASK_NAME,
    EXPECTED_SPLITS,
    check_manifest_hashes,
    check_notebook_metadata,
    check_run_artifact,
    check_split_episode_counts,
    check_task_artifact,
    is_known_bad_run_shape,
    materialize_task_definition,
    run_contract_audit,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


# ── fixture builders ────────────────────────────────────────────


def _make_good_task() -> dict:
    """Build a task.json dict that passes all contract checks."""
    return {
        "task_id": CANONICAL_TASK_ID,
        "task_name": CANONICAL_TASK_NAME,
        "tasks": [
            {
                "name": CANONICAL_BINARY_TASK_NAME,
                "description": (
                    "Cognitive flexibility benchmark: infer a hidden rule shift "
                    "from sparse contradictory evidence in a sequence of charge "
                    "interactions, then predict four post-shift probe outcomes."
                ),
                "role": "leaderboard_primary",
            },
            {
                "name": CANONICAL_NARRATIVE_TASK_NAME,
                "description": (
                    "Narrative companion for RuleShift Benchmark v1. Same "
                    "episodes as Binary, natural-language prompt format. "
                    "Robustness evidence only — not leaderboard-scored."
                ),
                "role": "companion",
            },
        ],
        "chosen_task": CANONICAL_BINARY_TASK_NAME,
        "splits": list(EXPECTED_SPLITS),
        "episodes_per_split": 18,
        "total_episodes": 54,
        "probe_count": 4,
    }


def _make_good_run() -> dict:
    """Build a minimal run artifact dict that passes all contract checks.

    Mirrors the canonical benchmark run artifact shape consumed by the audit.
    """
    rows = [
        {
            "episode_id": f"ife-r12-{1036 + i}",
            "template_id": "T1",
            "difficulty": "easy",
            "transition": "R_inv_to_R_std",
            "probe_targets": ["attract", "repel", "attract", "repel"],
            "modes": {
                "binary": {
                    "parse_status": "valid",
                    "predicted_labels": ["attract", "repel", "attract", "repel"],
                    "correct_probe_count": 4,
                    "failure_bucket": "correct",
                },
                "narrative": {
                    "parse_status": "valid",
                    "predicted_labels": ["attract", "repel", "attract", "repel"],
                    "correct_probe_count": 4,
                    "failure_bucket": "correct",
                },
            },
        }
        for i in range(18)
    ]

    return {
        "artifact_schema_version": "v1.1",
        "release_id": "R18",
        "provider_name": "test",
        "model_name": "test-model",
        "prompt_modes": ["binary", "narrative"],
        "splits": [
            {"split_name": "dev", "episode_count": 18, "rows": rows},
            {"split_name": "public_leaderboard", "episode_count": 18, "rows": rows},
            {"split_name": "private_leaderboard", "episode_count": 18, "rows": rows},
        ],
        "execution_summary": [
            {
                "scope_type": "overall",
                "scope_label": "all",
                "mode": "Binary",
                "episode_count": 54,
                "completed_count": 54,
            }
        ],
        "diagnostic_summary": [
            {
                "scope_type": "overall",
                "scope_label": "all",
                "mode": "Binary",
                "episode_count": 54,
                "correct_count": 54,
                "correct_rate": 1.0,
            }
        ],
    }


def _make_known_bad_run() -> dict:
    """Build the exact known bad run shape:

    - one conversation
    - conversations[].metrics == {}
    - aggregated results[].numericResult contains only confidenceInterval
    """
    return {
        "conversations": [
            {"metrics": {}}
        ],
        "results": [
            {"numericResult": {"confidenceInterval": 0.5}}
        ],
    }


# ── positive tests ──────────────────────────────────────────────


class TestContractAuditPositive:
    """Passing test using the expected real serialized artifact schema."""

    def test_good_task_passes(self):
        task_json = _make_good_task()
        errors = check_task_artifact(task_json)
        assert errors == []

    def test_good_run_passes(self):
        run_json = _make_good_run()
        errors = check_run_artifact(run_json)
        assert errors == []

    def test_notebook_metadata_passes(self):
        errors = check_notebook_metadata(_REPO_ROOT)
        assert errors == []

    def test_manifest_hashes_pass(self):
        errors = check_manifest_hashes(_REPO_ROOT)
        assert errors == []

    def test_materialize_task_definition_matches_contract(self):
        task_json = materialize_task_definition(_REPO_ROOT)
        errors = check_task_artifact(task_json)
        assert errors == [], f"materialized task failed validation: {errors}"

    def test_full_contract_audit_passes(self):
        """End-to-end: the real repo state must pass the full contract audit."""
        report = run_contract_audit(_REPO_ROOT)
        failing_sections = [
            name
            for name, section in report["checks"].items()
            if not section["passed"]
        ]
        assert report["passed"], (
            f"contract audit failed in: {failing_sections}; "
            f"errors: {report['errors']}"
        )

    def test_expected_dataset_sources_derived_from_canonical_metadata(self):
        """EXPECTED_DATASET_SOURCES must equal dataset_sources in canonical kernel-metadata.json.

        Regression: must not be derived from KAGGLE_USERNAME or any runtime env var.
        """
        from maintainer.contract_audit import EXPECTED_DATASET_SOURCES

        kernel_meta = json.loads(
            (_REPO_ROOT / "packaging" / "kaggle" / "kernel-metadata.json").read_text(
                encoding="utf-8"
            )
        )
        assert list(EXPECTED_DATASET_SOURCES) == kernel_meta["dataset_sources"], (
            "EXPECTED_DATASET_SOURCES diverged from canonical kernel-metadata.json. "
            "It must not be derived from KAGGLE_USERNAME or any runtime value."
        )

    def test_split_episode_counts_use_supplied_repo_root(self, tmp_path):
        manifests_dir = tmp_path / "src" / "frozen_splits"
        manifests_dir.mkdir(parents=True)

        for partition in ("dev", "public_leaderboard"):
            payload = json.loads(
                (_REPO_ROOT / "src" / "frozen_splits" / f"{partition}.json").read_text(
                    encoding="utf-8"
                )
            )
            payload["seeds"] = payload["seeds"][:1]
            (manifests_dir / f"{partition}.json").write_text(
                json.dumps(payload, indent=2) + "\n",
                encoding="utf-8",
            )

        errors = check_split_episode_counts(tmp_path)

        assert errors == [
            "dev manifest has 1 seeds but EXPECTED_EPISODES_PER_SPLIT is 18",
            "public_leaderboard manifest has 1 seeds but EXPECTED_EPISODES_PER_SPLIT is 18",
        ]

    def test_full_contract_audit_reports_split_manifest_load_failures(self, monkeypatch, tmp_path):
        artifact_path = tmp_path / "artifact.json"
        artifact_path.write_text("{}", encoding="utf-8")

        monkeypatch.setattr("maintainer.contract_audit.check_notebook_metadata", lambda repo_root=None: [])
        monkeypatch.setattr(
            "maintainer.contract_audit.materialize_task_definition",
            lambda repo_root=None: _make_good_task(),
        )
        monkeypatch.setattr("maintainer.contract_audit.check_task_artifact", lambda task_json: [])
        monkeypatch.setattr("maintainer.contract_audit.check_manifest_hashes", lambda repo_root=None: [])
        monkeypatch.setattr(
            "maintainer.contract_audit.find_latest_run_artifact",
            lambda repo_root=None: artifact_path,
        )
        monkeypatch.setattr("maintainer.contract_audit.check_run_artifact", lambda run_json: [])

        def fake_load_split_manifest(partition: str, repo_root=None):
            raise FileNotFoundError(f"{partition}.json missing")

        monkeypatch.setattr("core.splits.load_split_manifest", fake_load_split_manifest)

        report = run_contract_audit(_REPO_ROOT)

        assert report["passed"] is False
        assert report["checks"]["split_episode_counts"]["passed"] is False
        assert report["checks"]["split_episode_counts"]["errors"] == [
            "dev manifest could not be loaded: dev.json missing",
            "public_leaderboard manifest could not be loaded: public_leaderboard.json missing",
        ]


# ── canonical metadata structure ─────────────────────────────────


class TestCanonicalMetadataStructure:
    """Validates the canonical Kaggle metadata files that are the repository source of truth.

    These tests catch placeholder reintroduction, missing required fields, and
    cross-reference drift between kernel-metadata.json and dataset-metadata.json.
    """

    _KERNEL_META = json.loads(
        (_REPO_ROOT / "packaging" / "kaggle" / "kernel-metadata.json").read_text(
            encoding="utf-8"
        )
    )
    _DATASET_META = json.loads(
        (_REPO_ROOT / "packaging" / "kaggle" / "dataset-metadata.json").read_text(
            encoding="utf-8"
        )
    )

    def test_canonical_files_exist(self):
        assert (_REPO_ROOT / "packaging" / "kaggle" / "kernel-metadata.json").is_file()
        assert (_REPO_ROOT / "packaging" / "kaggle" / "dataset-metadata.json").is_file()

    def test_kernel_metadata_has_required_fields(self):
        for field in ("id", "title", "code_file", "dataset_sources"):
            assert self._KERNEL_META.get(field), (
                f"kernel-metadata.json missing required field {field!r}"
            )
        assert isinstance(self._KERNEL_META["dataset_sources"], list)
        assert len(self._KERNEL_META["dataset_sources"]) > 0

    def test_dataset_metadata_has_required_fields(self):
        for field in ("id", "title", "licenses"):
            assert self._DATASET_META.get(field), (
                f"dataset-metadata.json missing required field {field!r}"
            )
        assert isinstance(self._DATASET_META["licenses"], list)
        assert len(self._DATASET_META["licenses"]) > 0

    def test_metadata_ids_are_owner_slug_format(self):
        """id fields must be owner/slug — no template tokens, no missing owner."""
        for name, meta in [
            ("kernel-metadata.json", self._KERNEL_META),
            ("dataset-metadata.json", self._DATASET_META),
        ]:
            parts = meta.get("id", "").split("/")
            assert len(parts) == 2 and all(parts), (
                f"{name} id must be owner/slug format, got {meta.get('id')!r}"
            )

    def test_canonical_metadata_contains_no_placeholders(self):
        """Fail if KAGGLE_USERNAME is reintroduced as a value in either metadata file."""
        for name, meta in [
            ("kernel-metadata.json", self._KERNEL_META),
            ("dataset-metadata.json", self._DATASET_META),
        ]:
            assert "KAGGLE_USERNAME" not in json.dumps(meta), (
                f"{name} contains the placeholder string 'KAGGLE_USERNAME'"
            )

    def test_kernel_dataset_sources_reference_canonical_dataset(self):
        """kernel-metadata.json dataset_sources must include the canonical dataset id."""
        assert self._DATASET_META["id"] in self._KERNEL_META["dataset_sources"], (
            "kernel dataset_sources and dataset-metadata.json id are out of sync"
        )


# ── negative test: canonical metadata placeholder detection ──────


class TestNegativeCanonicalMetadataPlaceholders:
    """Verify that contract validation catches placeholder values in dataset_sources."""

    def test_check_notebook_metadata_rejects_placeholder_dataset_source(self, monkeypatch):
        """Regression: if EXPECTED_DATASET_SOURCES contained a placeholder owner, the
        contract audit should flag it as missing from the canonical metadata file."""
        monkeypatch.setattr(
            "maintainer.contract_audit.EXPECTED_DATASET_SOURCES",
            ("KAGGLE_USERNAME/ruleshift-runtime",),
        )
        errors = check_notebook_metadata(_REPO_ROOT)
        assert any("dataset_sources" in e for e in errors), (
            "check_notebook_metadata did not flag the placeholder dataset source"
        )


# ── negative test: wrong task name ──────────────────────────────


class TestNegativeWrongTaskName:
    def test_wrong_task_id(self):
        task_json = _make_good_task()
        task_json["task_id"] = "wrong_benchmark_v1"
        errors = check_task_artifact(task_json)
        assert any("task_id" in e for e in errors)

    def test_wrong_task_name(self):
        task_json = _make_good_task()
        task_json["task_name"] = "Wrong Benchmark v1"
        errors = check_task_artifact(task_json)
        assert any("task_name" in e for e in errors)


# ── negative test: missing Narrative companion evidence ─────────


class TestNegativeMissingNarrative:
    def test_run_without_narrative_prompt_mode(self):
        run_json = _make_good_run()
        run_json["prompt_modes"] = ["binary"]
        errors = check_run_artifact(run_json)
        assert any("Narrative" in e for e in errors)

    def test_task_without_narrative_definition(self):
        task_json = _make_good_task()
        task_json["tasks"] = [
            t for t in task_json["tasks"]
            if t["name"] != CANONICAL_NARRATIVE_TASK_NAME
        ]
        errors = check_task_artifact(task_json)
        assert any("companion task" in e for e in errors)


# ── negative test: missing required aggregated result evidence ──


class TestNegativeMissingAggregatedResults:
    def test_run_without_any_summary(self):
        run_json = _make_good_run()
        del run_json["diagnostic_summary"]
        del run_json["execution_summary"]
        errors = check_run_artifact(run_json)
        assert any("aggregated result evidence" in e for e in errors)

    def test_run_with_empty_summaries(self):
        run_json = _make_good_run()
        run_json["diagnostic_summary"] = []
        run_json["execution_summary"] = []
        errors = check_run_artifact(run_json)
        assert any("aggregated result evidence" in e for e in errors)

    def test_run_with_only_one_episode(self):
        run_json = _make_good_run()
        run_json["splits"] = [
            {"split_name": "dev", "episode_count": 1, "rows": [run_json["splits"][0]["rows"][0]]}
        ]
        errors = check_run_artifact(run_json)
        assert any("at most 1 episode" in e for e in errors)


# ── negative test: manifest/hash mismatch ───────────────────────


class TestNegativeManifestHashMismatch:
    def test_manifest_validation_failure_detected(self, monkeypatch):
        def fake_validate(*args, **kwargs):
            raise ValueError("sha256 mismatch: expected abc123, got def456")

        monkeypatch.setattr(
            "maintainer.contract_audit.validate_kaggle_staging_manifest",
            fake_validate,
        )
        errors = check_manifest_hashes(_REPO_ROOT)
        assert len(errors) == 1
        assert "mismatch" in errors[0]

    def test_manifest_missing_file_detected(self, monkeypatch):
        def fake_validate(*args, **kwargs):
            raise FileNotFoundError("kbench_notebook points to missing file")

        monkeypatch.setattr(
            "maintainer.contract_audit.validate_kaggle_staging_manifest",
            fake_validate,
        )
        errors = check_manifest_hashes(_REPO_ROOT)
        assert len(errors) == 1
        assert "missing file" in errors[0]


# ── negative test: exact known bad run shape ────────────────────


class TestNegativeKnownBadRunShape:
    def test_known_bad_shape_detected(self):
        bad_run = _make_known_bad_run()
        assert is_known_bad_run_shape(bad_run) is True

    def test_known_bad_shape_rejected_by_check(self):
        bad_run = _make_known_bad_run()
        errors = check_run_artifact(bad_run)
        assert any("known bad run shape" in e for e in errors)

    def test_known_bad_shape_also_fails_aggregated_check(self):
        """The known bad shape has only confidenceInterval — no real evidence."""
        bad_run = _make_known_bad_run()
        errors = check_run_artifact(bad_run)
        assert any("aggregated result evidence" in e for e in errors)

    def test_good_run_is_not_known_bad_shape(self):
        good_run = _make_good_run()
        assert is_known_bad_run_shape(good_run) is False

    def test_known_bad_shape_variants(self):
        """Two conversations escapes the one-conversation check."""
        two_conv = {
            "conversations": [{"metrics": {}}, {"metrics": {}}],
            "results": [{"numericResult": {"confidenceInterval": 0.5}}],
        }
        assert is_known_bad_run_shape(two_conv) is False

    def test_non_empty_metrics_escapes_known_bad_check(self):
        run = _make_known_bad_run()
        run["conversations"][0]["metrics"] = {"accuracy": 0.5}
        assert is_known_bad_run_shape(run) is False
