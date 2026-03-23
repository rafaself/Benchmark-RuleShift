from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from core import cli
from core.anthropic_panel import AnthropicPanelArtifacts
from core.gemini_panel import GeminiFirstPanelArtifacts
from core.model_execution import ModelMode
from core.openai_panel import OpenAIPanelArtifacts


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
    assert payload["difficulty_labels_missing"] == ["hard"]
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
        "core.cli.write_text_with_timestamped_snapshot",
        fake_write_text_with_timestamped_snapshot,
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

    assert commands == [
        ["test"],
        ["validity"],
        ["reaudit"],
        ["integrity"],
        ["evidence-pass"],
    ]


def test_gemini_first_panel_command_fails_clearly_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("core.providers.gemini._repo_root", lambda: tmp_path)

    exit_code = cli.main(["gemini-first-panel"])

    captured = capsys.readouterr()

    assert exit_code == 2
    assert "GEMINI_API_KEY" in captured.err


def test_gemini_first_panel_command_emits_report_metadata(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
):
    class _StubReport:
        release_id = "R18"

    report_path = tmp_path / "gemini_first_panel_report.md"
    monkeypatch.setattr(
        cli,
        "run_gemini_first_panel",
        lambda **_: GeminiFirstPanelArtifacts(
            provider_name="gemini",
            model_name="gemini-2.5-flash-001",
            prompt_modes=(ModelMode.BINARY,),
            release_report=_StubReport(),
            report_markdown="# report\n",
            report_path=report_path,
            artifact_payload={"prompt_modes": ["binary"]},
            artifact_path=report_path.with_suffix(".json"),
            snapshot_report_path=tmp_path / "gemini_first_panel_report__20260322_203000.md",
            snapshot_artifact_path=tmp_path / "gemini_first_panel_report__20260322_203000.json",
        ),
    )

    exit_code = cli.main(["gemini-first-panel", "--report-path", str(report_path)])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload == {
        "release_id": "R18",
        "provider_name": "gemini",
        "model_name": "gemini-2.5-flash-001",
        "prompt_modes": ["binary"],
        "report_path": str(report_path),
        "artifact_path": str(report_path.with_suffix(".json")),
        "snapshot_report_path": str(
            tmp_path / "gemini_first_panel_report__20260322_203000.md"
        ),
        "snapshot_artifact_path": str(
            tmp_path / "gemini_first_panel_report__20260322_203000.json"
        ),
    }


def test_gemini_first_panel_command_emits_narrative_mode_when_requested(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
):
    class _StubReport:
        release_id = "R18"

    report_path = tmp_path / "m1_binary_vs_narrative_robustness_report.md"
    monkeypatch.setattr(
        cli,
        "run_gemini_first_panel",
        lambda **_: GeminiFirstPanelArtifacts(
            provider_name="gemini",
            model_name="gemini-2.5-flash-001",
            prompt_modes=(ModelMode.BINARY, ModelMode.NARRATIVE),
            release_report=_StubReport(),
            report_markdown="# report\n",
            report_path=report_path,
            artifact_payload={"prompt_modes": ["binary", "narrative"]},
            artifact_path=report_path.with_suffix(".json"),
            snapshot_report_path=tmp_path / "m1_binary_vs_narrative_robustness_report__20260322_203500.md",
            snapshot_artifact_path=tmp_path / "m1_binary_vs_narrative_robustness_report__20260322_203500.json",
        ),
    )

    exit_code = cli.main(
        [
            "gemini-first-panel",
            "--include-narrative",
            "--report-path",
            str(report_path),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["prompt_modes"] == ["binary", "narrative"]
    assert payload["artifact_path"] == str(report_path.with_suffix(".json"))
    assert payload["snapshot_report_path"] == str(
        tmp_path / "m1_binary_vs_narrative_robustness_report__20260322_203500.md"
    )


def test_gemini_first_panel_command_rejects_unpinned_model_ids(
    capsys: pytest.CaptureFixture[str],
):
    exit_code = cli.main(["gemini-first-panel", "--model", "gemini-2.5-flash"])

    captured = capsys.readouterr()

    assert exit_code == 2
    assert "pinned model ID" in captured.err


def test_anthropic_panel_command_fails_clearly_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("core.providers.anthropic._repo_root", lambda: tmp_path)

    exit_code = cli.main(["anthropic-panel"])

    captured = capsys.readouterr()

    assert exit_code == 2
    assert "ANTHROPIC_API_KEY" in captured.err


def test_anthropic_panel_command_emits_report_metadata(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
):
    class _StubReport:
        release_id = "R18"

    report_path = tmp_path / "anthropic_panel_report.md"
    monkeypatch.setattr(
        cli,
        "run_anthropic_panel",
        lambda **_: AnthropicPanelArtifacts(
            provider_name="anthropic",
            model_name="claude-3-5-haiku-20241022",
            prompt_modes=(ModelMode.BINARY,),
            release_report=_StubReport(),
            report_markdown="# report\n",
            report_path=report_path,
            artifact_payload={"prompt_modes": ["binary"]},
            artifact_path=report_path.with_suffix(".json"),
            snapshot_report_path=tmp_path / "anthropic_panel_report__20260322_210000.md",
            snapshot_artifact_path=tmp_path / "anthropic_panel_report__20260322_210000.json",
        ),
    )

    exit_code = cli.main(["anthropic-panel", "--report-path", str(report_path)])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload == {
        "release_id": "R18",
        "provider_name": "anthropic",
        "model_name": "claude-3-5-haiku-20241022",
        "prompt_modes": ["binary"],
        "report_path": str(report_path),
        "artifact_path": str(report_path.with_suffix(".json")),
        "snapshot_report_path": str(
            tmp_path / "anthropic_panel_report__20260322_210000.md"
        ),
        "snapshot_artifact_path": str(
            tmp_path / "anthropic_panel_report__20260322_210000.json"
        ),
    }


def test_anthropic_panel_command_rejects_unpinned_model_ids(
    capsys: pytest.CaptureFixture[str],
):
    exit_code = cli.main(["anthropic-panel", "--model", "claude-3-5-haiku-latest"])

    captured = capsys.readouterr()

    assert exit_code == 2
    assert "pinned model ID" in captured.err


def test_openai_panel_command_fails_clearly_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("core.providers.openai._repo_root", lambda: tmp_path)

    exit_code = cli.main(["openai-panel"])

    captured = capsys.readouterr()

    assert exit_code == 2
    assert "OPENAI_API_KEY" in captured.err


def test_openai_panel_command_emits_report_metadata(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
):
    class _StubReport:
        release_id = "R18"

    report_path = tmp_path / "openai_panel_report.md"
    monkeypatch.setattr(
        cli,
        "run_openai_panel",
        lambda **_: OpenAIPanelArtifacts(
            provider_name="openai",
            model_name="gpt-5-mini-2025-08-07",
            prompt_modes=(ModelMode.BINARY,),
            release_report=_StubReport(),
            report_markdown="# report\n",
            report_path=report_path,
            artifact_payload={"prompt_modes": ["binary"]},
            artifact_path=report_path.with_suffix(".json"),
            snapshot_report_path=tmp_path / "openai_panel_report__20260322_220000.md",
            snapshot_artifact_path=tmp_path / "openai_panel_report__20260322_220000.json",
        ),
    )

    exit_code = cli.main(["openai-panel", "--report-path", str(report_path)])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload == {
        "release_id": "R18",
        "provider_name": "openai",
        "model_name": "gpt-5-mini-2025-08-07",
        "prompt_modes": ["binary"],
        "report_path": str(report_path),
        "artifact_path": str(report_path.with_suffix(".json")),
        "snapshot_report_path": str(
            tmp_path / "openai_panel_report__20260322_220000.md"
        ),
        "snapshot_artifact_path": str(
            tmp_path / "openai_panel_report__20260322_220000.json"
        ),
    }


def test_openai_panel_command_rejects_unpinned_model_ids(
    capsys: pytest.CaptureFixture[str],
):
    exit_code = cli.main(["openai-panel", "--model", "gpt-5-mini"])

    captured = capsys.readouterr()

    assert exit_code == 2
    assert "pinned model ID" in captured.err


def test_main_returns_130_on_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    monkeypatch.setattr(
        cli,
        "run_gemini_first_panel",
        lambda **_: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    exit_code = cli.main(["gemini-first-panel"])

    captured = capsys.readouterr()

    assert exit_code == 130
    assert captured.out == ""
    assert captured.err.strip() == "Interrupted."
