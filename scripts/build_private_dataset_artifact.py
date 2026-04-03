#!/usr/bin/env python3
"""Build the private leaderboard dataset attachment from the local seed manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Final

_MANIFEST_FIELD_ORDER: Final[tuple[str, ...]] = (
    "partition",
    "episode_split",
    "manifest_version",
    "seed_bank_version",
    "spec_version",
    "generator_version",
    "template_set_version",
    "difficulty_version",
    "seeds",
)
_DEFAULT_MANIFEST: Final[Path] = (
    Path(__file__).resolve().parents[1] / "src" / "frozen_splits" / "private_leaderboard.json"
)
_PRIVATE_EPISODES_FILENAME: Final[str] = "private_episodes.json"
_PRIVATE_ARTIFACT_SCHEMA: Final[str] = "private_split_artifact.v1"


def _to_jsonable(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _load_manifest(manifest_path: Path | None):
    from tasks.ruleshift_benchmark.runtime import FrozenSplitManifest, Split

    path = _DEFAULT_MANIFEST if manifest_path is None else manifest_path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if tuple(payload) != _MANIFEST_FIELD_ORDER:
        raise ValueError(
            "private manifest fields must exactly match the canonical order: "
            + ", ".join(_MANIFEST_FIELD_ORDER)
        )
    return FrozenSplitManifest(
        partition=payload["partition"],
        episode_split=Split(payload["episode_split"]),
        manifest_version=payload["manifest_version"],
        seed_bank_version=payload["seed_bank_version"],
        spec_version=payload["spec_version"],
        generator_version=payload["generator_version"],
        template_set_version=payload["template_set_version"],
        difficulty_version=payload["difficulty_version"],
        seeds=tuple(payload["seeds"]),
    )


def _build_payload(manifest) -> dict[str, object]:
    from tasks.ruleshift_benchmark.runtime import Split, _generate_episode

    if manifest.partition != "private_leaderboard":
        raise ValueError("private artifact generation requires the private_leaderboard manifest")
    if manifest.episode_split is not Split.PRIVATE:
        raise ValueError("private artifact generation requires Split.PRIVATE seeds")

    episodes = [
        {"seed": seed, "episode": _to_jsonable(_generate_episode(seed, split=Split.PRIVATE))}
        for seed in manifest.seeds
    ]
    payload: dict[str, object] = {
        "partition": "private_leaderboard",
        "episode_split": "private",
        "benchmark_version": manifest.manifest_version,
        "schema_version": _PRIVATE_ARTIFACT_SCHEMA,
        "episodes": episodes,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**payload, "artifact_checksum": hashlib.sha256(encoded).hexdigest()}


def _write_artifact(output_dir: Path, *, manifest_path: Path | None = None) -> Path:
    manifest = _load_manifest(manifest_path)
    payload = _build_payload(manifest)
    out_dir = output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = out_dir / _PRIVATE_EPISODES_FILENAME
    episodes_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return episodes_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a private dataset attachment containing private_episodes.json.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest-path", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    episodes_path = _write_artifact(args.output_dir, manifest_path=args.manifest_path)
    print(episodes_path.parent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
