#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
from types import ModuleType

from scripts.private_local_loader import load_private_local_module
from scripts.private_release_paths import resolve_private_release_dirs

_PRIVATE_MODULE: ModuleType | None = None
ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROWS_DATASET_DIR, PRIVATE_SCORING_DATASET_DIR = resolve_private_release_dirs(ROOT)
PRIVATE_DATASET_DIR = PRIVATE_ROWS_DATASET_DIR


def _module() -> ModuleType:
    global _PRIVATE_MODULE
    if _PRIVATE_MODULE is None:
        _PRIVATE_MODULE = load_private_local_module(
            "build_private_cogflex_dataset.py",
            "scripts.private_local.build_private_cogflex_dataset",
        )
    return _PRIVATE_MODULE


def build_private_bundle(
    rows_dir: Path | None = None,
    scoring_dir: Path | None = None,
):
    resolved_rows_dir, resolved_scoring_dir = resolve_private_release_dirs(
        ROOT,
        rows_dir=rows_dir,
        scoring_dir=scoring_dir,
    )
    return _module().build_private_bundle(resolved_rows_dir, resolved_scoring_dir)


def main() -> None:
    build_private_bundle()


if __name__ == "__main__":
    main()
