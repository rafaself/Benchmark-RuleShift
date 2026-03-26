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
