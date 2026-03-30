from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pytest

from maintainer import cli
from core.private_split import PRIVATE_DATASET_ROOT_ENV_VAR


def test_build_parser_exposes_only_canonical_cli_commands():
    parser = cli.build_parser()
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )

    assert set(subparsers.choices) == {
        "test",
        "validity",
        "reaudit",
        "integrity",
        "evidence-pass",
        "contract-audit",
        "doctor",
    }


def test_validity_command_emits_current_gate_report(capsys: pytest.CaptureFixture[str]):
    exit_code = cli.main(["validity"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["release_id"] == "R13"
    assert payload["passed"] is True
    assert payload["gate_splits"] == ["public_leaderboard", "private_leaderboard"]


def test_reaudit_command_emits_current_release_report(capsys: pytest.CaptureFixture[str]):
    exit_code = cli.main(["reaudit"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["release_id"] == "R15"
    assert payload["difficulty_labels_missing"] == []
    assert payload["model_summaries"] == []


def test_integrity_command_can_write_output_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
):
    def fake_write_text_with_timestamped_snapshot(path: Path, text: str):
        path.write_text(text, encoding="utf-8")
        snapshot_path = path.parent / f"{path.stem}__20260322_202500{path.suffix}"
        snapshot_path.write_text(text, encoding="utf-8")
        return path, snapshot_path

    monkeypatch.setattr(
        "maintainer.cli.write_text_with_timestamped_snapshot",
        fake_write_text_with_timestamped_snapshot,
    )
    monkeypatch.setattr(
        cli,
        "_build_integrity_payload",
        lambda: {
            "overlap_check_passed": True,
            "kaggle_manifest_valid": True,
            "audit_issue_count": 0,
        },
    )
    output_path = tmp_path / "integrity.json"

    exit_code = cli.main(["integrity", "--output", str(output_path)])

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    captured = capsys.readouterr()

    assert exit_code == 0
    assert payload["overlap_check_passed"] is True
    assert payload["kaggle_manifest_valid"] is True
    assert payload["audit_issue_count"] == 0
    assert (tmp_path / "integrity__20260322_202500.json").is_file()
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
    assert cli.contract_audit_entrypoint() == 0
    assert cli.doctor_entrypoint() == 0

    assert commands == [
        ["test"],
        ["validity"],
        ["reaudit"],
        ["integrity"],
        ["evidence-pass"],
        ["contract-audit"],
        ["doctor"],
    ]


def test_removed_panel_commands_are_rejected():
    with pytest.raises(SystemExit):
        cli.main(["gemini-first-panel"])

    with pytest.raises(SystemExit):
        cli.main(["anthropic-panel"])

    with pytest.raises(SystemExit):
        cli.main(["openai-panel"])


def test_main_returns_130_on_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    monkeypatch.setattr(
        cli,
        "_command_validity",
        lambda args: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    exit_code = cli.main(["validity"])

    captured = capsys.readouterr()

    assert exit_code == 130
    assert captured.out == ""
    assert captured.err.strip() == "Interrupted."


# ---------------------------------------------------------------------------
# doctor command
# ---------------------------------------------------------------------------


def test_doctor_exits_zero_in_public_only_environment(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    monkeypatch.delenv(PRIVATE_DATASET_ROOT_ENV_VAR, raising=False)

    exit_code = cli.main(["doctor"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "not mounted" in captured.out
    assert "public-only" in captured.out
    assert PRIVATE_DATASET_ROOT_ENV_VAR in captured.out


def test_doctor_reports_unavailable_for_private_commands_in_public_only_environment(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    monkeypatch.delenv(PRIVATE_DATASET_ROOT_ENV_VAR, raising=False)

    cli.main(["doctor"])

    captured = capsys.readouterr()
    assert "unavailable" in captured.out
    assert "make validity" in captured.out
    assert "make reaudit" in captured.out
    assert "make integrity" in captured.out


def test_doctor_exits_zero_with_private_split_mounted(
    capsys: pytest.CaptureFixture[str],
):
    # autouse fixture provides the private split via env var
    exit_code = cli.main(["doctor"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "mounted at" in captured.out
    assert "unavailable" not in captured.out


# ---------------------------------------------------------------------------
# fail-fast preflight for private-required commands
# ---------------------------------------------------------------------------


def _assert_private_required_fail(
    command: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv(PRIVATE_DATASET_ROOT_ENV_VAR, raising=False)

    exit_code = cli.main([command])

    captured = capsys.readouterr()
    assert exit_code == 1, f"{command!r} should exit 1 without private split"
    assert "private_leaderboard split is not mounted" in captured.err
    assert PRIVATE_DATASET_ROOT_ENV_VAR in captured.err
    assert "make doctor" in captured.err
    assert captured.out == ""


def test_validity_fails_fast_without_private_split(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    _assert_private_required_fail("validity", monkeypatch, capsys)


def test_reaudit_fails_fast_without_private_split(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    _assert_private_required_fail("reaudit", monkeypatch, capsys)


def test_integrity_fails_fast_without_private_split(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    _assert_private_required_fail("integrity", monkeypatch, capsys)


def test_evidence_pass_fails_fast_without_private_split(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    _assert_private_required_fail("evidence-pass", monkeypatch, capsys)
