#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from scripts.build_cogflex_dataset import PRIVATE_DATASET_ID, dataset_metadata  # noqa: E402
from scripts.private_cogflex_bundle import write_private_bundle  # noqa: E402

PRIVATE_DATASET_DIR = ROOT / "kaggle/dataset/private"


def build_private_bundle(output_dir: Path = PRIVATE_DATASET_DIR) -> dict[str, Path]:
    """Build the private CogFlex dataset bundle on disk.

    Args:
        output_dir: Directory where the private bundle files should be written.

    Returns:
        A mapping from bundle artifact names to the paths written on disk.

    """
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_paths = write_private_bundle(output_dir)
    metadata_path = output_dir / "dataset-metadata.json"
    metadata_path.write_text(
        json.dumps(dataset_metadata(PRIVATE_DATASET_ID, "CogFlex Suite Runtime Private"), indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        **bundle_paths,
        "metadata": metadata_path,
    }


def main() -> None:
    """Build the default private dataset bundle.

    Returns:
        None.

    """
    build_private_bundle()


if __name__ == "__main__":
    main()
