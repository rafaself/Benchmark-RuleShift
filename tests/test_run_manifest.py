from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from core.kaggle.manifest import (
    RUN_MANIFEST_FILENAME,
    write_run_manifest,
)
from core.kaggle import manifest as manifest_module

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_manifest(tmp_path: Path) -> dict[str, object]:
    manifest_path = write_run_manifest(
        run_id="run-manifest-001",
        output_dir=tmp_path / "manifest-run",
        repo_root=_REPO_ROOT,
        started_at="2026-03-30T00:00:00Z",
        finished_at="2026-03-30T00:01:00Z",
    )
    assert manifest_path == (tmp_path / "manifest-run" / RUN_MANIFEST_FILENAME)
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def test_write_run_manifest_uses_ci_commit_when_available(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GITHUB_SHA", "ci-commit-sha")
    monkeypatch.setattr(manifest_module.subprocess, "run", _unexpected_git_call)

    manifest = _write_manifest(tmp_path)

    notebook_path = _REPO_ROOT / "packaging" / "kaggle" / "ruleshift_notebook_task.ipynb"
    expected_hash = hashlib.sha256(notebook_path.read_bytes()).hexdigest()

    assert manifest["run_id"] == "run-manifest-001"
    assert manifest["git_commit"] == "ci-commit-sha"
    assert manifest["benchmark_version"] == "R14"
    assert manifest["parser_version"] == "v2"
    assert manifest["metrics_version"] == "v1"
    assert manifest["notebook_bundle_hash"] == expected_hash
    assert manifest["runtime_dataset_id"] == "raptorengineer/ruleshift-runtime"
    assert manifest["runtime_dataset_version"] is None
    assert manifest["provider"] == "unknown"
    assert manifest["model"] == "unknown"
    assert manifest["started_at"] == "2026-03-30T00:00:00Z"
    assert manifest["finished_at"] == "2026-03-30T00:01:00Z"


def test_write_run_manifest_uses_local_git_commit_when_ci_sha_missing(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    monkeypatch.setattr(manifest_module.subprocess, "run", _successful_git_rev_parse)

    manifest = _write_manifest(tmp_path)

    assert manifest["git_commit"] == "local-git-sha"


def test_write_run_manifest_allows_non_git_execution(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    monkeypatch.setattr(manifest_module.subprocess, "run", _failing_git_rev_parse)

    manifest = _write_manifest(tmp_path)

    assert manifest["git_commit"] is None


def _unexpected_git_call(*args, **kwargs):
    raise AssertionError("git should not be invoked when GITHUB_SHA is set")


def _successful_git_rev_parse(*args, **kwargs):
    return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="local-git-sha\n", stderr="")


def _failing_git_rev_parse(*args, **kwargs):
    raise subprocess.CalledProcessError(returncode=128, cmd=args[0], stderr="not a git repository")
