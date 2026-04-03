#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
KAGGLE_DIR = REPO_ROOT / "kaggle"
SRC_DIR = REPO_ROOT / "src"
DATASET_RUNTIME_FILES = (
    Path("frozen_splits/public_leaderboard_rows.json"),
)


def build_kaggle_package(output_dir: Path) -> tuple[Path, Path]:
    output_dir = output_dir.resolve()
    kernel_dir = output_dir / "kernel"
    dataset_dir = output_dir / "dataset"

    if output_dir.exists():
        shutil.rmtree(output_dir)

    kernel_dir.mkdir(parents=True)
    shutil.copy2(KAGGLE_DIR / "ruleshift_notebook_task.ipynb", kernel_dir)
    shutil.copy2(KAGGLE_DIR / "kernel-metadata.json", kernel_dir)

    dataset_src_dir = dataset_dir / "src"
    for rel_path in DATASET_RUNTIME_FILES:
        destination = dataset_src_dir / rel_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SRC_DIR / rel_path, destination)
    shutil.copy2(KAGGLE_DIR / "dataset-metadata.json", dataset_dir)

    return kernel_dir, dataset_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the Kaggle notebook and benchmark dataset package.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/ruleshift-kaggle-build"),
    )
    args = parser.parse_args()
    kernel_dir, dataset_dir = build_kaggle_package(args.output_dir)
    print(kernel_dir.parent)
    print(kernel_dir)
    print(dataset_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
