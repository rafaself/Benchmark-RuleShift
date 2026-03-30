from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
KAGGLE_DIR = REPO_ROOT / "packaging" / "kaggle"


def build_output_parser(*, description: str, help_text: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=help_text,
    )
    return parser


def load_required_json(path: Path, *, required_fields: tuple[str, ...]) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"{path.name} not found at {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    missing = [field for field in required_fields if not payload.get(field)]
    if missing:
        raise ValueError(f"{path.name} is missing required fields: {missing}")
    return payload


def reset_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()
