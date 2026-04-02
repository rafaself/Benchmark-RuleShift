from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
_NOTEBOOK_PATH = _REPO_ROOT / "packaging" / "kaggle" / "ruleshift_notebook_task.ipynb"

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import tests.kbench_shim as _shim  # noqa: E402

sys.modules["kaggle_benchmarks"] = _shim  # type: ignore[assignment]
import kaggle_benchmarks as kbench  # noqa: E402

_EXPECTED_PUBLIC_EPISODES = 54
_EXPECTED_PRIVATE_EPISODES = 270


def _execute_notebook() -> dict:
    notebook = json.loads(_NOTEBOOK_PATH.read_text(encoding="utf-8"))
    ns: dict = {"__builtins__": __import__("builtins")}
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", ()))
        filtered = "\n".join(
            line for line in source.splitlines() if not line.strip().startswith("%")
        )
        if filtered.strip():
            exec(compile(filtered, f"<{cell['id']}>", "exec"), ns)  # noqa: S102
    return ns


@pytest.fixture(autouse=True)
def _patch_kaggle_runtime_path(monkeypatch: pytest.MonkeyPatch) -> None:
    original_is_dir = Path.is_dir

    def _kaggle_aware_is_dir(self: Path) -> bool:
        if str(self) == "/kaggle/input/datasets/raptorengineer/ruleshift-runtime/src":
            return True
        return original_is_dir(self)

    monkeypatch.setattr(Path, "is_dir", _kaggle_aware_is_dir)


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    kbench.reset_registry()
    yield
    kbench.reset_registry()


@pytest.fixture(autouse=True)
def _run_output_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RULESHIFT_RUN_OUTPUT_DIR", str(tmp_path / "notebook-run"))
    yield


def test_notebook_executes_end_to_end_with_minimal_pipeline() -> None:
    ns = _execute_notebook()

    assert ns["PRIVATE_DATASET_ROOT"] is not None
    assert [partition["partition"] for partition in ns["bundle"]["partitions"]] == [
        "public_leaderboard",
        "private_leaderboard",
    ]
    assert list(ns["partition_df"]["episodes"]) == [
        _EXPECTED_PUBLIC_EPISODES,
        _EXPECTED_PRIVATE_EPISODES,
    ]
    assert len(ns["leaderboard_df"]) == _EXPECTED_PUBLIC_EPISODES + _EXPECTED_PRIVATE_EPISODES
    assert ns["score"] == (ns["result"]["numerator"], ns["result"]["denominator"])
    assert ns["_RULESHIFT_RESULT"] == ns["result"]
    assert ns["_RULESHIFT_BINARY_DF"] is not None
    assert len(ns["_RULESHIFT_BINARY_DF"]) == len(ns["leaderboard_df"])
    assert set(kbench.get_registry()) == {"ruleshift_benchmark_v1_binary"}
