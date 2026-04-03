from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
_NOTEBOOK_PATH = _REPO_ROOT / "kaggle" / "ruleshift_notebook_task.ipynb"
_PRIVATE_EPISODES_FILENAME = "private_episodes.json"
_PRIVATE_ARTIFACT_SCHEMA = "private_split_artifact.v1"
_EXPECTED_PUBLIC_EPISODES = 54
_EXPECTED_PRIVATE_EPISODES = 270

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from scripts.build_kaggle import build_kaggle_package  # noqa: E402
from tasks.ruleshift_benchmark.runtime import (  # noqa: E402
    DIFFICULTY_VERSION,
    GENERATOR_VERSION,
    MANIFEST_VERSION,
    PRIVATE_DATASET_ROOT_ENV_VAR,
    SPEC_VERSION,
    TEMPLATE_SET_VERSION,
    FrozenSplitManifest,
    Split,
    _generate_episode,
)


class _LLMStub:
    def prompt(self, text: str, *, schema: Any = None) -> dict[str, str]:
        return {
            "probe_6": "type_a",
            "probe_7": "type_b",
            "probe_8": "type_a",
            "probe_9": "type_b",
        }


class _KBenchShim:
    llm = _LLMStub()

    @staticmethod
    def task(*, name: str, description: str):
        def decorator(fn):
            def wrapped(*args, **kwargs):
                return fn(*args, **kwargs)

            wrapped.__name__ = fn.__name__
            return wrapped

        return decorator


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


def _write_private_dataset(root: Path) -> None:
    manifest = FrozenSplitManifest(
        partition="private_leaderboard",
        episode_split=Split.PRIVATE,
        manifest_version=MANIFEST_VERSION,
        seed_bank_version="R14-private-5",
        spec_version=SPEC_VERSION,
        generator_version=GENERATOR_VERSION,
        template_set_version=TEMPLATE_SET_VERSION,
        difficulty_version=DIFFICULTY_VERSION,
        seeds=tuple(range(37800, 38070)),
    )
    episodes = [
        {
            "seed": seed,
            "episode": _to_jsonable(_generate_episode(seed, split=Split.PRIVATE)),
        }
        for seed in manifest.seeds
    ]
    payload: dict[str, object] = {
        "partition": manifest.partition,
        "episode_split": manifest.episode_split.value,
        "benchmark_version": manifest.manifest_version,
        "schema_version": _PRIVATE_ARTIFACT_SCHEMA,
        "episodes": episodes,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["artifact_checksum"] = hashlib.sha256(encoded).hexdigest()
    (root / _PRIVATE_EPISODES_FILENAME).write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def _execute_notebook() -> dict[str, object]:
    notebook = json.loads(_NOTEBOOK_PATH.read_text(encoding="utf-8"))
    namespace: dict[str, object] = {"__builtins__": __import__("builtins")}
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", ()))
        source = "\n".join(
            line for line in source.splitlines() if not line.strip().startswith("%")
        )
        if source.strip():
            exec(compile(source, f"<{cell['id']}>", "exec"), namespace)  # noqa: S102
    return namespace


def test_kaggle_build_and_notebook_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel_dir, dataset_dir = build_kaggle_package(tmp_path / "build")
    assert (kernel_dir / "ruleshift_notebook_task.ipynb").is_file()
    assert (dataset_dir / "src" / "tasks" / "ruleshift_benchmark" / "runtime.py").is_file()
    assert (dataset_dir / "src" / "frozen_splits" / "public_leaderboard.json").is_file()

    private_dataset_root = tmp_path / "private-dataset"
    private_dataset_root.mkdir()
    _write_private_dataset(private_dataset_root)

    monkeypatch.setenv(PRIVATE_DATASET_ROOT_ENV_VAR, str(private_dataset_root))
    monkeypatch.setitem(sys.modules, "kaggle_benchmarks", _KBenchShim())

    original_is_dir = Path.is_dir

    def kaggle_runtime_is_dir(path: Path) -> bool:
        if str(path) == "/kaggle/input/datasets/raptorengineer/ruleshift-runtime/src":
            return True
        return original_is_dir(path)

    monkeypatch.setattr(Path, "is_dir", kaggle_runtime_is_dir)

    namespace = _execute_notebook()

    assert namespace["PRIVATE_DATASET_ROOT"] is not None
    assert [p["partition"] for p in namespace["bundle"]["partitions"]] == [
        "public_leaderboard",
        "private_leaderboard",
    ]
    assert list(namespace["partition_df"]["episodes"]) == [
        _EXPECTED_PUBLIC_EPISODES,
        _EXPECTED_PRIVATE_EPISODES,
    ]
    assert len(namespace["leaderboard_df"]) == _EXPECTED_PUBLIC_EPISODES + _EXPECTED_PRIVATE_EPISODES
    assert namespace["score"] == (
        namespace["result"]["numerator"],
        namespace["result"]["denominator"],
    )
