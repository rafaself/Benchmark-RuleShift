from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PUBLIC_ROWS_PATH = _REPO_ROOT / "src" / "frozen_splits" / "public_leaderboard_rows.json"
_PRIVATE_ROWS_FILENAME = "private_leaderboard_rows.json"
_ALLOWED_LABELS = {"type_a", "type_b"}

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.build_kaggle import build_kaggle_package  # noqa: E402
from tasks.ruleshift_benchmark.runtime import (  # noqa: E402
    BinaryResponse,
    KaggleExecutionError,
    PRIVATE_DATASET_ROOT_ENV_VAR,
    load_public_rows,
    normalize_binary_response,
    run_binary_task,
)


class _LLMStub:
    def prompt(self, text: str, *, schema: Any = None) -> BinaryResponse:
        return BinaryResponse(
            probe_6="type_a",
            probe_7="type_b",
            probe_8="type_a",
            probe_9="type_b",
        )


class _FailingLLM:
    def prompt(self, text: str, *, schema: Any = None) -> BinaryResponse:
        raise RuntimeError("boom")


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
            "episode_id": "ife-private-1",
            "split": "private",
            "prompt_binary": "Classify the probes. Return type_a or type_b for each.",
            "probe_targets": ["type_a", "type_b", "type_a", "type_b"],
        },
        {
            "episode_id": "ife-private-2",
            "split": "private",
            "prompt_binary": "Classify the probes. Return type_a or type_b for each.",
            "probe_targets": ["type_b", "type_a", "type_b", "type_a"],
        },
    ]
    (root / _PRIVATE_ROWS_FILENAME).write_text(
        json.dumps(rows, indent=2) + "\n",
        encoding="utf-8",
    )


def _execute_notebook(path: Path) -> dict[str, object]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
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


def _purge_runtime_modules() -> None:
    for name in list(sys.modules):
        if name == "tasks" or name.startswith("tasks."):
            sys.modules.pop(name, None)


def test_public_split_file_exists() -> None:
    assert _PUBLIC_ROWS_PATH.is_file()


def test_public_rows_have_four_binary_probe_targets() -> None:
    rows = load_public_rows()
    assert rows
    for row in rows:
        assert len(row["probe_targets"]) == 4
        assert set(row["probe_targets"]) <= _ALLOWED_LABELS


def test_binary_predictions_normalize_to_public_labels() -> None:
    normalized = normalize_binary_response(
        {
            "probe_6": "type_a",
            "probe_7": "type_b",
            "probe_8": "type_a",
            "probe_9": "type_b",
        }
    )
    assert normalized is not None
    assert set(normalized) <= _ALLOWED_LABELS


def test_task_response_failure_is_explicit() -> None:
    with pytest.raises(KaggleExecutionError, match="llm.prompt failed"):
        run_binary_task(
            llm=_FailingLLM(),
            prompt_binary="prompt",
            probe_targets=("type_a", "type_b", "type_a", "type_b"),
        )


def test_kaggle_build_and_notebook_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel_dir, dataset_dir = build_kaggle_package(tmp_path / "build")
    assert sorted(
        str(path.relative_to(kernel_dir))
        for path in kernel_dir.rglob("*")
        if path.is_file()
    ) == ["kernel-metadata.json", "ruleshift_notebook_task.ipynb"]
    assert sorted(
        str(path.relative_to(dataset_dir))
        for path in dataset_dir.rglob("*")
        if path.is_file()
    ) == [
        "dataset-metadata.json",
        "src/frozen_splits/public_leaderboard_rows.json",
        "src/tasks/ruleshift_benchmark/__init__.py",
        "src/tasks/ruleshift_benchmark/runtime.py",
    ]

    private_dataset_root = tmp_path / "private-dataset"
    private_dataset_root.mkdir()
    _write_private_dataset(private_dataset_root)

    monkeypatch.setenv(PRIVATE_DATASET_ROOT_ENV_VAR, str(private_dataset_root))
    monkeypatch.setitem(sys.modules, "kaggle_benchmarks", _KBenchShim())
    monkeypatch.syspath_prepend(str(dataset_dir / "src"))

    original_is_dir = Path.is_dir

    def kaggle_runtime_is_dir(path: Path) -> bool:
        if str(path) == "/kaggle/input/datasets/raptorengineer/ruleshift-runtime/src":
            return True
        return original_is_dir(path)

    monkeypatch.setattr(Path, "is_dir", kaggle_runtime_is_dir)
    _purge_runtime_modules()

    namespace = _execute_notebook(kernel_dir / "ruleshift_notebook_task.ipynb")

    assert len(namespace["leaderboard_rows"]) == len(load_public_rows()) + 2
    assert namespace["score"] == (
        namespace["result"]["numerator"],
        namespace["result"]["denominator"],
    )
    assert namespace["result"]["episodes"] == len(namespace["leaderboard_rows"])
