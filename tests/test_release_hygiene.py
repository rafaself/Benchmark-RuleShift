from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


_REPO_ROOT = Path(__file__).resolve().parents[1]
_README_PATH = _REPO_ROOT / "README.md"
_CANONICAL_M1_REPORT_PATH = (
    _REPO_ROOT
    / "reports"
    / "live"
    / "gemini-first-panel"
    / "binary-vs-narrative"
    / "latest"
    / "report.md"
)
_CANONICAL_M1_ARTIFACT_PATH = (
    _REPO_ROOT
    / "reports"
    / "live"
    / "gemini-first-panel"
    / "binary-vs-narrative"
    / "latest"
    / "artifact.json"
)
_M1_ALIAS_REPORT_PATH = (
    _REPO_ROOT / "reports" / "m1_binary_vs_narrative_robustness_report.md"
)
_M1_ALIAS_ARTIFACT_PATH = (
    _REPO_ROOT / "reports" / "m1_binary_vs_narrative_robustness_report.json"
)
_CANONICAL_M1_METADATA_PATH = (
    _REPO_ROOT
    / "reports"
    / "live"
    / "gemini-first-panel"
    / "binary-vs-narrative"
    / "latest"
    / "metadata.json"
)
_CANONICAL_BINARY_ONLY_METADATA_PATH = (
    _REPO_ROOT
    / "reports"
    / "live"
    / "gemini-first-panel"
    / "binary-only"
    / "latest"
    / "metadata.json"
)
_CANONICAL_COMPARISON_METADATA_PATH = (
    _REPO_ROOT
    / "reports"
    / "live"
    / "gemini-first-panel"
    / "comparison"
    / "latest"
    / "metadata.json"
)


def test_readme_does_not_report_stale_m3_status():
    text = _README_PATH.read_text(encoding="utf-8")

    assert "- **M3**: Not started." not in text


def test_readme_is_the_single_development_source_of_truth_doc():
    text = _README_PATH.read_text(encoding="utf-8")

    assert "main development source of truth" in text
    assert "FROZEN_BENCHMARK_SPEC.md" not in text


def test_public_repo_does_not_commit_private_split_artifacts():
    assert not (_REPO_ROOT / "src" / "frozen_splits" / "private_leaderboard.json").exists()
    assert not (_REPO_ROOT / "packaging" / "kaggle" / "private").exists()


def test_public_private_isolation_guard_passes():
    completed = subprocess.run(
        [sys.executable, "scripts/check_public_private_isolation.py"],
        cwd=_REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_committed_m1_aliases_match_canonical_latest_surfaces():
    assert _M1_ALIAS_REPORT_PATH.read_text(encoding="utf-8") == (
        _CANONICAL_M1_REPORT_PATH.read_text(encoding="utf-8")
    )
    assert _M1_ALIAS_ARTIFACT_PATH.read_text(encoding="utf-8") == (
        _CANONICAL_M1_ARTIFACT_PATH.read_text(encoding="utf-8")
    )


def test_committed_m1_artifact_uses_current_diagnostic_schema():
    payload = json.loads(_CANONICAL_M1_ARTIFACT_PATH.read_text(encoding="utf-8"))

    assert payload["artifact_schema_version"] == "v1.1"
    assert "execution_summary" in payload
    assert "diagnostic_summary" in payload
    assert "diagnostic_episode_rows" in payload
    assert payload["prompt_modes"] == ["binary", "narrative"]

    first_mode_payload = payload["splits"][0]["rows"][0]["modes"]["binary"]
    assert "response_text" not in first_mode_payload
    assert "error_message" not in first_mode_payload


def test_committed_m1_report_exposes_current_diagnostic_sections():
    text = _CANONICAL_M1_REPORT_PATH.read_text(encoding="utf-8")

    assert "Binary-only headline metric" in text
    assert "## Execution Provenance (diagnostic-only)" in text
    assert "## Failure Decomposition (diagnostic-only)" in text
    assert "## Direct Disagreement Diagnostics (diagnostic-only)" in text
    assert "## Diagnostic Failure Slices (diagnostic-only)" in text


def test_current_latest_metadata_surfaces_do_not_advertise_stale_benchmark_versions():
    expected_versions = {
        "schema_version": "v1",
        "generator_version": "R13",
        "template_family_version": "v2",
        "parser_version": "v1",
        "metric_version": "v1",
        "difficulty_version": "R13",
        "artifact_schema_version": "v1.1",
    }

    for path in (
        _CANONICAL_M1_METADATA_PATH,
        _CANONICAL_BINARY_ONLY_METADATA_PATH,
        _CANONICAL_COMPARISON_METADATA_PATH,
    ):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["benchmark_versions"] == expected_versions


def test_current_latest_m1_metadata_uses_mount_only_private_artifact_path():
    payload = json.loads(_CANONICAL_M1_METADATA_PATH.read_text(encoding="utf-8"))
    private_record = next(
        record
        for record in payload["frozen_artifacts"]["split_manifests"]
        if record["split_name"] == "private_leaderboard"
    )

    assert private_record["path"] == "<private_dataset_root>/private_episodes.json"
