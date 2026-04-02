#!/usr/bin/env python3
"""Build the private leaderboard dataset attachment from the local seed manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a private dataset attachment containing private_episodes.json.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Output directory for the private dataset attachment.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Optional path to the ignored private_leaderboard.json seed manifest.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    from _private_builder import write_private_dataset_artifact

    args = build_parser().parse_args(argv)
    episodes_path = write_private_dataset_artifact(
        args.output_dir,
        manifest_path=args.manifest_path,
    )
    print(episodes_path.parent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
