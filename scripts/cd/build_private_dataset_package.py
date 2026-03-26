#!/usr/bin/env python3
"""Build the private dataset package from an already-generated private artifact.

This flow is intentionally separate from the public deploy build. It handles
only the mounted private artifact and private dataset metadata.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.private_split import PRIVATE_EPISODES_FILENAME, load_private_split_manifest_info


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the private dataset package from a private_episodes.json artifact.",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        required=True,
        help="Path to the generated private_episodes.json artifact.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for the private dataset package. Must be outside the public repo tree.",
    )
    parser.add_argument(
        "--dataset-id",
        required=True,
        help="Fully qualified Kaggle dataset id for the private dataset package.",
    )
    parser.add_argument(
        "--title",
        default="RuleShift Private Evaluation Dataset",
        help="Dataset title to write into dataset-metadata.json.",
    )
    return parser


def _require_outside_repo(path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError:
        return resolved
    raise ValueError(
        f"{label} must be outside the public repository tree"
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    artifact_path = args.artifact.resolve()
    output_dir = _require_outside_repo(args.output_dir, label="output-dir")

    if artifact_path.name != PRIVATE_EPISODES_FILENAME:
        raise ValueError(
            f"artifact must point to {PRIVATE_EPISODES_FILENAME}"
        )
    if not artifact_path.is_file():
        raise FileNotFoundError(f"artifact not found: {artifact_path}")

    artifact_root = artifact_path.parent
    private_manifest = load_private_split_manifest_info(artifact_root)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    shutil.copy2(artifact_path, output_dir / PRIVATE_EPISODES_FILENAME)
    dataset_metadata = {
        "title": args.title,
        "id": args.dataset_id,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (output_dir / "dataset-metadata.json").write_text(
        json.dumps(dataset_metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "CONTENTS.md").write_text(
        "\n".join(
            [
                "# RuleShift Private Dataset Package",
                "",
                "Copied items:",
                "",
                f"- `{PRIVATE_EPISODES_FILENAME}`",
                "- `dataset-metadata.json`",
                "",
                "Private artifact metadata:",
                "",
                f"- benchmark_version: `{private_manifest['benchmark_version']}`",
                f"- schema_version: `{private_manifest['schema_version']}`",
                f"- artifact_checksum: `{private_manifest['artifact_checksum']}`",
                "",
                "This package is private-only and must not be merged into the public runtime package.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
