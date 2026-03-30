from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_pre_deploy_check_runs_successfully(tmp_path: Path):
    runtime_dir = tmp_path / "runtime-package"
    kernel_dir = tmp_path / "kernel-bundle"
    env = {
        **os.environ,
        "PYTHON": sys.executable,
        "RUNTIME_BUILD_DIR": str(runtime_dir),
        "KERNEL_BUILD_DIR": str(kernel_dir),
    }

    result = subprocess.run(
        ["bash", "scripts/pre_deploy_check.sh"],
        cwd=_REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert "[gate] phase-1 environment sanity: ok" in result.stdout
    assert "[gate] phase-2 preflight: ok" in result.stdout
    assert "[gate] phase-3 targeted schema/runtime tests: ok" in result.stdout
    assert "[gate] runtime dataset artifact consistency: ok" in result.stdout
    assert "[gate] kernel bundle consistency: ok" in result.stdout
    assert "=== Pre-deploy gate passed ===" in result.stdout
