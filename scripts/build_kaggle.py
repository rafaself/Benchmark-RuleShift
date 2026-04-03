#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
KAGGLE_DIR = REPO_ROOT / "kaggle"
SRC_DIR = REPO_ROOT / "src"


def build_kaggle_package(output_dir: Path) -> tuple[Path, Path]:
    output_dir = output_dir.resolve()
    kernel_dir = output_dir / "kernel"
    dataset_dir = output_dir / "dataset"

    if output_dir.exists():
        shutil.rmtree(output_dir)

    kernel_dir.mkdir(parents=True)
    shutil.copy2(KAGGLE_DIR / "ruleshift_notebook_task.ipynb", kernel_dir)
    shutil.copy2(KAGGLE_DIR / "kernel-metadata.json", kernel_dir)

    shutil.copytree(
        SRC_DIR,
        dataset_dir / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(KAGGLE_DIR / "dataset-metadata.json", dataset_dir)

    return kernel_dir, dataset_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the Kaggle notebook and runtime dataset package.",
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
