import hashlib
import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
import sys

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(_REPO_ROOT / "src"))

from tasks.ruleshift_benchmark.runtime import (  # noqa: E402
    DIFFICULTY_VERSION,
    GENERATOR_VERSION,
    PRIVATE_DATASET_ROOT_ENV_VAR,
    SPEC_VERSION,
    TEMPLATE_SET_VERSION,
    FrozenSplitManifest,
    Split,
    _generate_episode,
)

_TEST_PRIVATE_SEEDS = tuple(range(37800, 38070))
_TEST_PRIVATE_MANIFEST = FrozenSplitManifest(
    partition="private_leaderboard",
    episode_split=Split.PRIVATE,
    manifest_version="R14",
    seed_bank_version="R14-private-5",
    spec_version=SPEC_VERSION,
    generator_version=GENERATOR_VERSION,
    template_set_version=TEMPLATE_SET_VERSION,
    difficulty_version=DIFFICULTY_VERSION,
    seeds=_TEST_PRIVATE_SEEDS,
)

_PRIVATE_EPISODES_FILENAME = "private_episodes.json"
_PRIVATE_ARTIFACT_SCHEMA = "private_split_artifact.v1"


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


def _build_private_episodes_payload(manifest: FrozenSplitManifest) -> dict[str, object]:
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


@pytest.fixture(scope="session")
def mounted_private_dataset_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    dataset_root = tmp_path_factory.mktemp("private-dataset")
    episodes_path = dataset_root / _PRIVATE_EPISODES_FILENAME
    episodes_payload = _build_private_episodes_payload(_TEST_PRIVATE_MANIFEST)
    episodes_path.write_text(
        json.dumps(episodes_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return dataset_root


@pytest.fixture(autouse=True)
def _mounted_private_dataset_env(
    mounted_private_dataset_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        PRIVATE_DATASET_ROOT_ENV_VAR,
        str(mounted_private_dataset_root),
    )
    yield
    monkeypatch.delenv(PRIVATE_DATASET_ROOT_ENV_VAR, raising=False)
