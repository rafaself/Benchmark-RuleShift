from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
_NOTEBOOK_PATH = _REPO_ROOT / "kaggle" / "ruleshift_notebook_task.ipynb"
_PRIVATE_ROWS_FILENAME = "private_leaderboard_rows.json"
_EXPECTED_PUBLIC_EPISODES = 54
_EXPECTED_PRIVATE_EPISODES = 270
_PRIVATE_SEEDS = range(37800, 38070)

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from scripts.build_kaggle import build_kaggle_package  # noqa: E402
from tasks.ruleshift_benchmark.runtime import (  # noqa: E402
    PRIVATE_DATASET_ROOT_ENV_VAR,
    format_public_label,
    load_public_rows,
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


def _write_private_dataset(root: Path) -> None:
    rows = [
        {
            "episode_id": f"ife-r13-{seed}",
            "split": "private",
            "prompt_binary": "Classify the probes. Return type_a or type_b for each.",
            "probe_targets": [
                format_public_label("zark"),
                format_public_label("blim"),
                format_public_label("zark"),
                format_public_label("blim"),
            ],
        }
        for seed in _PRIVATE_SEEDS
    ]
    (root / _PRIVATE_ROWS_FILENAME).write_text(
        json.dumps(rows, indent=2) + "\n",
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
    assert (dataset_dir / "src" / "frozen_splits" / "public_leaderboard_rows.json").is_file()

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
    assert len(load_public_rows()) == _EXPECTED_PUBLIC_EPISODES
    assert list(namespace["partition_df"]["episodes"]) == [
        _EXPECTED_PUBLIC_EPISODES,
        _EXPECTED_PRIVATE_EPISODES,
    ]
    assert len(namespace["leaderboard_df"]) == _EXPECTED_PUBLIC_EPISODES + _EXPECTED_PRIVATE_EPISODES
    assert namespace["score"] == (
        namespace["result"]["numerator"],
        namespace["result"]["denominator"],
    )
