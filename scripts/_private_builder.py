"""Private episode builder — development tooling only.

Generates private_episodes.json from the ignored private seed manifest.
This module is NOT included in the public runtime package; it is used
only by scripts/build_private_dataset_artifact.py for local private
dataset construction.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Final

from tasks.ruleshift_benchmark.splits import (
    FrozenSplitManifest,
    PRIVATE_EPISODES_FILENAME,
    PRIVATE_SPLIT_ARTIFACT_SCHEMA_VERSION,
)
from tasks.ruleshift_benchmark.generator import generate_episode
from tasks.ruleshift_benchmark.protocol import Split

__all__ = [
    "PRIVATE_SPLIT_MANIFEST_FILENAME",
    "build_private_episodes_payload",
    "default_private_manifest_path",
    "load_private_seed_manifest",
    "write_private_dataset_artifact",
]

PRIVATE_SPLIT_MANIFEST_FILENAME: Final[str] = "private_leaderboard.json"
_PRIVATE_MANIFEST_FIELD_ORDER: Final[tuple[str, ...]] = (
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


def default_private_manifest_path() -> Path:
    return Path(__file__).resolve().parents[1] / "src" / "frozen_splits" / PRIVATE_SPLIT_MANIFEST_FILENAME


def load_private_seed_manifest(
    manifest_path: Path | str | None = None,
) -> FrozenSplitManifest:
    path = default_private_manifest_path() if manifest_path is None else Path(manifest_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    actual_fields = tuple(payload)
    if actual_fields != _PRIVATE_MANIFEST_FIELD_ORDER:
        raise ValueError(
            "private manifest fields must exactly match the canonical order: "
            + ", ".join(_PRIVATE_MANIFEST_FIELD_ORDER)
        )
    return FrozenSplitManifest(
        partition=payload["partition"],
        episode_split=payload["episode_split"],
        manifest_version=payload["manifest_version"],
        seed_bank_version=payload["seed_bank_version"],
        spec_version=payload["spec_version"],
        generator_version=payload["generator_version"],
        template_set_version=payload["template_set_version"],
        difficulty_version=payload["difficulty_version"],
        seeds=tuple(payload["seeds"]),
    )


def build_private_episodes_payload(
    manifest: FrozenSplitManifest,
) -> dict[str, object]:
    if manifest.partition != "private_leaderboard":
        raise ValueError("private artifact generation requires the private_leaderboard manifest")
    if manifest.episode_split is not Split.PRIVATE:
        raise ValueError("private artifact generation requires Split.PRIVATE seeds")

    episodes = [
        {
            "seed": seed,
            "episode": _to_jsonable(generate_episode(seed, split=Split.PRIVATE)),
        }
        for seed in manifest.seeds
    ]
    payload: dict[str, object] = {
        "partition": "private_leaderboard",
        "episode_split": "private",
        "benchmark_version": manifest.manifest_version,
        "schema_version": PRIVATE_SPLIT_ARTIFACT_SCHEMA_VERSION,
        "episodes": episodes,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        **payload,
        "artifact_checksum": hashlib.sha256(encoded).hexdigest(),
    }


def write_private_dataset_artifact(
    output_dir: Path | str,
    *,
    manifest_path: Path | str | None = None,
) -> Path:
    manifest = load_private_seed_manifest(manifest_path)
    payload = build_private_episodes_payload(manifest)

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = out_dir / PRIVATE_EPISODES_FILENAME
    episodes_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return episodes_path


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
