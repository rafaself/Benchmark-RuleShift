import json
from pathlib import Path
import sys

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from _private_builder import build_private_episodes_payload  # noqa: E402
from tasks.ruleshift_benchmark.splits import (  # noqa: E402
    FrozenSplitManifest,
    PRIVATE_DATASET_ROOT_ENV_VAR,
    PRIVATE_EPISODES_FILENAME,
)
from tasks.ruleshift_benchmark.protocol import Split  # noqa: E402
from tasks.ruleshift_benchmark.schema import (  # noqa: E402
    DIFFICULTY_VERSION,
    GENERATOR_VERSION,
    SPEC_VERSION,
    TEMPLATE_SET_VERSION,
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


@pytest.fixture(scope="session")
def mounted_private_dataset_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    dataset_root = tmp_path_factory.mktemp("private-dataset")
    episodes_path = dataset_root / PRIVATE_EPISODES_FILENAME
    episodes_payload = build_private_episodes_payload(_TEST_PRIVATE_MANIFEST)
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
