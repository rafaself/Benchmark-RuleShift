from __future__ import annotations

import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
_KAGGLE_DIR = _REPO_ROOT / "kaggle"
_SRC_DIR = _REPO_ROOT / "src"
_TASK_DIR = _SRC_DIR / "tasks" / "ruleshift_benchmark"

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.build_kaggle import build_kaggle_package

_EXPECTED_TASK_RUNTIME_RELPATHS = {
    "src/tasks/ruleshift_benchmark/protocol.py",
    "src/tasks/ruleshift_benchmark/runner.py",
    "src/tasks/ruleshift_benchmark/schema.py",
    "src/tasks/ruleshift_benchmark/splits.py",
}
_EXPECTED_RUNTIME_RELPATHS = {
    path.relative_to(_REPO_ROOT).as_posix()
    for path in _SRC_DIR.rglob("*")
    if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
}


def test_runtime_task_directory_matches_the_deployed_kaggle_layout() -> None:
    task_files = {
        path.relative_to(_REPO_ROOT).as_posix()
        for path in _TASK_DIR.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }

    assert task_files == _EXPECTED_TASK_RUNTIME_RELPATHS


def test_build_kaggle_package_copies_the_runtime_tree_directly(tmp_path: Path) -> None:
    output_dir = tmp_path / "kaggle-build"
    stale_path = output_dir / "stale.txt"
    stale_path.parent.mkdir(parents=True)
    stale_path.write_text("old", encoding="utf-8")

    kernel_dir, dataset_dir = build_kaggle_package(output_dir)

    assert output_dir.exists()
    assert not stale_path.exists()
    assert kernel_dir == output_dir / "kernel"
    assert dataset_dir == output_dir / "dataset"

    assert sorted(path.relative_to(kernel_dir).as_posix() for path in kernel_dir.rglob("*") if path.is_file()) == [
        "kernel-metadata.json",
        "ruleshift_notebook_task.ipynb",
    ]
    assert (
        kernel_dir / "ruleshift_notebook_task.ipynb"
    ).read_text(encoding="utf-8") == (
        _KAGGLE_DIR / "ruleshift_notebook_task.ipynb"
    ).read_text(encoding="utf-8")
    assert json.loads((kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8")) == json.loads(
        (_KAGGLE_DIR / "kernel-metadata.json").read_text(encoding="utf-8")
    )

    dataset_files = {
        path.relative_to(dataset_dir).as_posix()
        for path in dataset_dir.rglob("*")
        if path.is_file()
    }
    assert dataset_files == _EXPECTED_RUNTIME_RELPATHS | {"dataset-metadata.json"}

    assert json.loads((dataset_dir / "dataset-metadata.json").read_text(encoding="utf-8")) == json.loads(
        (_KAGGLE_DIR / "dataset-metadata.json").read_text(encoding="utf-8")
    )
