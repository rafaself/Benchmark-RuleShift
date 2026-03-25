"""Tests for the P0 contract audit.

Validates the contract audit against known-good and known-bad artifact shapes.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from core.contract_audit import (
    CANONICAL_BINARY_TASK_NAME,
    CANONICAL_NARRATIVE_TASK_NAME,
    CANONICAL_TASK_ID,
    CANONICAL_TASK_NAME,
    EXPECTED_SPLITS,
    check_manifest_hashes,
    check_notebook_metadata,
    check_run_artifact,
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
        "episodes_per_split": 16,
        "total_episodes": 48,
        "probe_count": 4,
    }


def _make_good_run() -> dict:
    """Build a minimal run artifact dict that passes all contract checks.

    Mirrors the real artifact.json shape produced by the panel runner.
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
        for i in range(16)
    ]

    return {
        "artifact_schema_version": "v1.1",
        "release_id": "R18",
        "provider_name": "test",
        "model_name": "test-model",
        "prompt_modes": ["binary", "narrative"],
        "splits": [
            {"split_name": "dev", "episode_count": 16, "rows": rows},
            {"split_name": "public_leaderboard", "episode_count": 16, "rows": rows},
            {"split_name": "private_leaderboard", "episode_count": 16, "rows": rows},
        ],
        "execution_summary": [
            {
                "scope_type": "overall",
                "scope_label": "all",
                "mode": "Binary",
                "episode_count": 48,
                "completed_count": 48,
            }
        ],
        "diagnostic_summary": [
            {
                "scope_type": "overall",
                "scope_label": "all",
                "mode": "Binary",
                "episode_count": 48,
                "correct_count": 48,
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
            "core.contract_audit.validate_kaggle_staging_manifest",
            fake_validate,
        )
        errors = check_manifest_hashes(_REPO_ROOT)
        assert len(errors) == 1
        assert "mismatch" in errors[0]

    def test_manifest_missing_file_detected(self, monkeypatch):
        def fake_validate(*args, **kwargs):
            raise FileNotFoundError("kbench_notebook points to missing file")

        monkeypatch.setattr(
            "core.contract_audit.validate_kaggle_staging_manifest",
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
