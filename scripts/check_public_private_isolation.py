#!/usr/bin/env python3
"""Fail when private split artifacts appear in public repo or public packaging paths."""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
KAGGLE_DIR = REPO_ROOT / "packaging" / "kaggle"
FORBIDDEN_PATHS = (
    SRC_DIR / "frozen_splits" / "private_leaderboard.json",
    KAGGLE_DIR / "private" / "private_episodes.json",
    KAGGLE_DIR / "private" / "dataset-metadata.json",
)
FORBIDDEN_FILENAMES = (
    "private_leaderboard.json",
    "private_episodes.json",
)
_IGNORED_REPO_DIRS = {".git", ".venv", ".pytest_cache", "__pycache__"}


def _collect_public_location_errors() -> list[str]:
    errors: list[str] = []

    for path in FORBIDDEN_PATHS:
        if path.exists():
            errors.append(f"forbidden private artifact present: {path.relative_to(REPO_ROOT)}")

    private_packaging_dir = KAGGLE_DIR / "private"
    if private_packaging_dir.exists():
        errors.append(f"forbidden public packaging directory present: {private_packaging_dir.relative_to(REPO_ROOT)}")

    for search_root in (SRC_DIR / "frozen_splits", KAGGLE_DIR):
        if not search_root.exists():
            continue
        for filename in FORBIDDEN_FILENAMES:
            for path in search_root.rglob(filename):
                errors.append(f"forbidden filename in public location: {path.relative_to(REPO_ROOT)}")

    for filename in FORBIDDEN_FILENAMES:
        for path in REPO_ROOT.rglob(filename):
            if any(part in _IGNORED_REPO_DIRS for part in path.parts):
                continue
            errors.append(f"forbidden filename committed to repo: {path.relative_to(REPO_ROOT)}")

    manifest_path = KAGGLE_DIR / "frozen_artifacts_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frozen_split_manifests = manifest.get("frozen_split_manifests", {})
    if "private_leaderboard" in frozen_split_manifests:
        errors.append("packaging/kaggle/frozen_artifacts_manifest.json exposes private_leaderboard")

    notebook_path = KAGGLE_DIR / "ruleshift_notebook_task.ipynb"
    notebook_text = notebook_path.read_text(encoding="utf-8")
    if "packaging/kaggle/private/private_episodes.json" in notebook_text:
        errors.append("official notebook still contains a repo-local private dataset fallback")

    return errors


def _collect_notebook_compliance_errors() -> list[str]:
    """Static compliance checks on the official notebook source."""
    errors: list[str] = []
    notebook_path = KAGGLE_DIR / "ruleshift_notebook_task.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    all_sources = "\n".join(
        "".join(cell.get("source", ()))
        for cell in notebook["cells"]
    )

    # Requirement 3: private split must be loaded only through authorized resolution
    if "resolve_private_dataset_root" not in all_sources:
        errors.append(
            "official notebook does not call resolve_private_dataset_root; "
            "private split must be loaded through the authorized flow"
        )

    # Requirement 4: leaderboard evaluation must explicitly exclude dev
    if '_LEADERBOARD_PARTITIONS = ("public_leaderboard", "private_leaderboard")' not in all_sources:
        errors.append(
            "official notebook does not define "
            '_LEADERBOARD_PARTITIONS = ("public_leaderboard", "private_leaderboard"); '
            "dev must be explicitly excluded from leaderboard evaluation"
        )

    # Requirement 5: final cell must select the single main task only
    last_code_source = ""
    for cell in reversed(notebook["cells"]):
        if cell.get("cell_type") == "code":
            last_code_source = "".join(cell.get("source", ()))
            break
    magic_lines = [
        line.strip()
        for line in last_code_source.splitlines()
        if line.strip().startswith("%")
    ]
    if magic_lines != ["%choose ruleshift_benchmark_v1_binary"]:
        errors.append(
            "last notebook code cell must contain exactly "
            "'%choose ruleshift_benchmark_v1_binary'; "
            f"got magic lines: {magic_lines!r}"
        )

    return errors

def main() -> int:
    errors = [
        *_collect_public_location_errors(),
        *_collect_notebook_compliance_errors(),
    ]
    if errors:
        print("Public/private isolation and compliance check FAILED:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print("Public/private isolation and compliance check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
