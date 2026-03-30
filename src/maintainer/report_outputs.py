from __future__ import annotations

from datetime import datetime
from pathlib import Path

__all__ = [
    "build_timestamped_snapshot_path",
    "current_report_timestamp",
    "write_json_with_timestamped_snapshot",
    "write_text_with_timestamped_snapshot",
]

_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
_LATEST_DIRNAME = "latest"
_HISTORY_DIRNAME = "history"


def current_report_timestamp() -> str:
    return datetime.now().astimezone().strftime(_TIMESTAMP_FORMAT)


def build_timestamped_snapshot_path(path: Path, *, timestamp: str) -> Path:
    history_dir = _snapshot_directory_for(path)
    snapshot_path = history_dir / f"{path.stem}__{timestamp}{path.suffix}"
    if not snapshot_path.exists():
        return snapshot_path

    counter = 1
    while True:
        candidate = history_dir / (
            f"{path.stem}__{timestamp}_{counter:02d}{path.suffix}"
        )
        if not candidate.exists():
            return candidate
        counter += 1


def write_text_with_timestamped_snapshot(
    path: Path,
    text: str,
    *,
    timestamp: str | None = None,
) -> tuple[Path, Path]:
    resolved_timestamp = current_report_timestamp() if timestamp is None else timestamp
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

    snapshot_path = build_timestamped_snapshot_path(path, timestamp=resolved_timestamp)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(text, encoding="utf-8")
    return path, snapshot_path


def write_json_with_timestamped_snapshot(
    path: Path,
    payload: str,
    *,
    timestamp: str | None = None,
) -> tuple[Path, Path]:
    return write_text_with_timestamped_snapshot(
        path,
        payload,
        timestamp=timestamp,
    )


def _snapshot_directory_for(path: Path) -> Path:
    if path.parent.name == _LATEST_DIRNAME:
        return path.parent.parent / _HISTORY_DIRNAME
    return path.parent
