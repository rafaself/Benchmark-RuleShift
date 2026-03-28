from __future__ import annotations

from pathlib import Path
import subprocess
import sys


_REPO_ROOT = Path(__file__).resolve().parents[1]
_README_PATH = _REPO_ROOT / "README.md"


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


def test_live_provider_panel_subsystem_is_out_of_mvp_scope():
    removed_paths = (
        _REPO_ROOT / "src" / "core" / "panel_runner.py",
        _REPO_ROOT / "src" / "core" / "anthropic_panel.py",
        _REPO_ROOT / "src" / "core" / "gemini_panel.py",
        _REPO_ROOT / "src" / "core" / "openai_panel.py",
        _REPO_ROOT / "tests" / "test_panel_runner.py",
        _REPO_ROOT / "tests" / "test_anthropic_panel.py",
        _REPO_ROOT / "tests" / "test_gemini_panel.py",
        _REPO_ROOT / "tests" / "test_openai_panel.py",
        _REPO_ROOT / "tests" / "test_anthropic_provider.py",
        _REPO_ROOT / "tests" / "test_gemini_provider.py",
        _REPO_ROOT / "tests" / "test_openai_provider.py",
        _REPO_ROOT / "tests" / "test_provider_registry.py",
        _REPO_ROOT / "tests" / "test_model_runner.py",
        _REPO_ROOT / "tests" / "test_intra_family_comparison.py",
        _REPO_ROOT / "tests" / "test_remote_verification.py",
    )

    for path in removed_paths:
        assert not path.exists(), f"unexpected live panel/provider path: {path}"

    providers_dir = _REPO_ROOT / "src" / "core" / "providers"
    assert not providers_dir.exists()


def test_public_private_isolation_guard_passes():
    completed = subprocess.run(
        [sys.executable, "scripts/check_public_private_isolation.py"],
        cwd=_REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


# ---------------------------------------------------------------------------
# Artifact policy guardrails (Release 2)
# ---------------------------------------------------------------------------


def test_benchmark_result_json_is_not_git_tracked():
    """benchmark_result.json is a transient notebook debug output and must not be tracked."""
    completed = subprocess.run(
        ["git", "ls-files", "--cached", "benchmark_result.json"],
        cwd=_REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.stdout.strip() == "", (
        "benchmark_result.json must not be tracked by git; "
        "it is a transient local debug output — see reports/ARTIFACT_POLICY.md"
    )


def test_benchmark_result_json_is_gitignored():
    """benchmark_result.json must be covered by .gitignore so it is never accidentally committed."""
    completed = subprocess.run(
        ["git", "check-ignore", "-q", "benchmark_result.json"],
        cwd=_REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, (
        "benchmark_result.json must be covered by .gitignore"
    )


def test_reports_live_directory_is_absent():
    """reports/live/ is reserved for transient local outputs and must not be tracked."""
    assert not (_REPO_ROOT / "reports" / "live").exists(), (
        "reports/live/ must not be committed; it is reserved for local run outputs "
        "covered by .gitignore — see reports/ARTIFACT_POLICY.md"
    )


def test_gitignore_covers_transient_local_outputs():
    """.gitignore must exclude benchmark_result.json and reports/live/."""
    gitignore = (_REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "benchmark_result.json" in gitignore
    assert "reports/live/" in gitignore


def test_artifact_policy_document_exists():
    """reports/ARTIFACT_POLICY.md must exist and classify canonical vs generated files."""
    policy_path = _REPO_ROOT / "reports" / "ARTIFACT_POLICY.md"
    assert policy_path.exists(), "reports/ARTIFACT_POLICY.md must exist"
    text = policy_path.read_text(encoding="utf-8")
    assert "Canonical inputs" in text
    assert "Transient local outputs" in text
    assert "Legacy" in text
