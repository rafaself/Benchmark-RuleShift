from __future__ import annotations

import subprocess
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_preflight_script_runs_successfully():
    result = subprocess.run(
        [sys.executable, "scripts/preflight_kaggle.py"],
        cwd=_REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert "=== RuleShift Kaggle Preflight ===" in result.stdout
    assert "[preflight] import runtime modules: ok" in result.stdout
    assert "[preflight] exercise minimal binary task path: ok" in result.stdout
    assert "[preflight] surface structural provider failures: ok" in result.stdout
    assert "=== Preflight Passed ===" in result.stdout
