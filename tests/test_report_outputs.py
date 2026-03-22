from __future__ import annotations

from pathlib import Path

from core.report_outputs import (
    build_latest_report_path,
    build_timestamped_snapshot_path,
    write_text_with_timestamped_snapshot,
)


def test_snapshot_for_latest_path_is_written_to_history_directory(tmp_path: Path):
    report_path = tmp_path / "reports" / "live" / "gemini-first-panel" / "binary-only" / "latest" / "report.md"

    canonical_path, snapshot_path = write_text_with_timestamped_snapshot(
        report_path,
        "# report\n",
        timestamp="20260322_210000",
    )

    assert canonical_path == report_path
    assert snapshot_path == (
        tmp_path
        / "reports"
        / "live"
        / "gemini-first-panel"
        / "binary-only"
        / "history"
        / "report__20260322_210000.md"
    )
    assert snapshot_path.read_text(encoding="utf-8") == "# report\n"


def test_snapshot_for_non_latest_path_stays_beside_canonical_file(tmp_path: Path):
    output_path = tmp_path / "integrity.json"

    snapshot_path = build_timestamped_snapshot_path(
        output_path,
        timestamp="20260322_210500",
    )

    assert snapshot_path == tmp_path / "integrity__20260322_210500.json"


def test_build_latest_report_path_uses_reports_root(monkeypatch):
    fake_root = Path("/tmp/fake-repo")
    monkeypatch.setattr("core.report_outputs._reports_root", lambda: fake_root / "reports")

    result = build_latest_report_path(
        "live",
        "gemini-first-panel",
        "binary-vs-narrative",
        filename="report.md",
    )

    assert result == (
        fake_root
        / "reports"
        / "live"
        / "gemini-first-panel"
        / "binary-vs-narrative"
        / "latest"
        / "report.md"
    )
