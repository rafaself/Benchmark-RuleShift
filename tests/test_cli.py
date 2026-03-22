from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from core import cli


def test_validity_command_emits_current_gate_report(capsys: pytest.CaptureFixture[str]):
    exit_code = cli.main(["validity"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["release_id"] == "R13"
    assert payload["passed"] is False
    assert payload["gate_splits"] == ["public_leaderboard", "private_leaderboard"]


def test_reaudit_command_emits_current_release_report(capsys: pytest.CaptureFixture[str]):
    exit_code = cli.main(["reaudit"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["release_id"] == "R15"
    assert payload["difficulty_labels_missing"] == ["hard"]
    assert payload["model_summaries"] == []


def test_integrity_command_can_write_output_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    output_path = tmp_path / "integrity.json"

    exit_code = cli.main(["integrity", "--output", str(output_path)])

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    captured = capsys.readouterr()

    assert exit_code == 0
    assert payload["overlap_check_passed"] is True
    assert payload["kaggle_manifest_valid"] is True
    assert payload["audit_issue_count"] == 0
    assert json.loads(captured.out) == payload


def test_test_command_returns_pytest_exit_code(monkeypatch: pytest.MonkeyPatch):
    def fake_run_pytest(pytest_args: list[str]) -> subprocess.CompletedProcess[str]:
        assert pytest_args == ["-q"]
        return subprocess.CompletedProcess(
            args=[sys.executable, "-m", "pytest", "-q"],
            returncode=0,
        )

    monkeypatch.setattr(cli, "_run_pytest", fake_run_pytest)

    assert cli.main(["test", "-q"]) == 0


def test_evidence_pass_stops_after_failed_tests(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    monkeypatch.setattr(
        cli,
        "_run_pytest",
        lambda pytest_args: subprocess.CompletedProcess(
            args=[sys.executable, "-m", "pytest", *pytest_args],
            returncode=2,
        ),
    )

    exit_code = cli.main(["evidence-pass", "-q"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 2
    assert payload == {
        "tests": {
            "passed": False,
            "exit_code": 2,
            "command": [sys.executable, "-m", "pytest", "-q"],
        }
    }


def test_entrypoint_aliases_dispatch_to_expected_commands(
    monkeypatch: pytest.MonkeyPatch,
):
    commands: list[list[str]] = []

    def fake_main(argv: list[str] | None = None) -> int:
        commands.append([] if argv is None else list(argv))
        return 0

    monkeypatch.setattr(cli, "main", fake_main)

    assert cli.test_entrypoint() == 0
    assert cli.validity_entrypoint() == 0
    assert cli.reaudit_entrypoint() == 0
    assert cli.integrity_entrypoint() == 0
    assert cli.evidence_pass_entrypoint() == 0

    assert commands == [
        ["test"],
        ["validity"],
        ["reaudit"],
        ["integrity"],
        ["evidence-pass"],
    ]
