import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.private_split import (  # noqa: E402
    PRIVATE_DATASET_ROOT_ENV_VAR,
    PRIVATE_EPISODES_FILENAME,
    build_private_split_artifact,
)
from core.splits import MANIFEST_VERSION  # noqa: E402

_TEST_PRIVATE_SEEDS = (
    40001,
    40016,
    40017,
    40018,
    40000,
    40004,
    40006,
    40008,
    40003,
    40005,
    40007,
    40012,
    40002,
    40010,
    40011,
    40015,
)


@pytest.fixture(scope="session")
def mounted_private_dataset_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    dataset_root = tmp_path_factory.mktemp("private-dataset")
    episodes_path = dataset_root / PRIVATE_EPISODES_FILENAME
    episodes_payload = build_private_split_artifact(
        benchmark_version=MANIFEST_VERSION,
        seeds=_TEST_PRIVATE_SEEDS,
    )
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
