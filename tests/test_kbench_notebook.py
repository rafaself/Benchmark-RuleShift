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

from core.kaggle import validate_kaggle_payload  # noqa: E402

import tests.kbench_shim as _shim  # noqa: E402

sys.modules["kaggle_benchmarks"] = _shim  # type: ignore[assignment]
import kaggle_benchmarks as kbench  # noqa: E402


def _read_notebook_sources() -> str:
    notebook = json.loads(_NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", ())) for cell in notebook["cells"])


def _execute_notebook_cells() -> dict:
    cells = json.loads(_NOTEBOOK_PATH.read_text(encoding="utf-8"))["cells"]
    ns: dict = {
        "__builtins__": __import__("builtins"),
        "display": lambda *args, **kwargs: None,
    }
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", ()))
        filtered = "\n".join(line for line in source.splitlines() if not line.strip().startswith("%"))
        if filtered.strip():
            exec(compile(filtered, f"<{cell['id']}>", "exec"), ns)  # noqa: S102
    return ns


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    kbench.reset_registry()
    yield
    kbench.reset_registry()


@pytest.fixture(autouse=True)
def _run_output_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RULESHIFT_RUN_OUTPUT_DIR", str(tmp_path / "notebook-run"))
    monkeypatch.setenv("RULESHIFT_RUN_ID", "test-run")
    yield
    monkeypatch.delenv("RULESHIFT_RUN_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("RULESHIFT_RUN_ID", raising=False)


def test_notebook_source_keeps_binary_only_leaderboard_surface():
    source = _read_notebook_sources()

    assert 'name="ruleshift_benchmark_v1_binary_row"' in source
    assert "store_task=False" in source
    assert '@kbench.task(\n    name="ruleshift_benchmark_v1_binary"' in source
    assert '@kbench.task(\n    name="ruleshift_benchmark_v1_narrative"' not in source
    assert "load_leaderboard_dataframe" in source
    assert "run_binary_task" in source
    assert "_ruleshift_benchmark_v1_binary_row.evaluate(" in source
    assert "packaging/kaggle/private/private_episodes.json" not in source
    assert "NotebookStatus" not in source
    assert "discover_private_dataset_root" not in source
    assert "load_private_split" not in source
    assert "run_narrative_episode" not in source
    assert "write_diagnostics_summary" not in source
    assert "write_run_manifest" not in source
    assert "benchmark_result.json" not in source
    assert "dev_df" not in source
    assert "%choose ruleshift_benchmark_v1_binary" in source


def test_notebook_executes_end_to_end_with_private_mount():
    ns = _execute_notebook_cells()

    assert set(ns["frozen_splits"]) == {"public_leaderboard", "private_leaderboard"}
    assert set(ns["leaderboard_df"]["split"]) == {"public_leaderboard", "private_leaderboard"}

    registry = kbench.get_registry()
    assert set(registry) == {"ruleshift_benchmark_v1_binary"}
    assert "_ruleshift_benchmark_v1_binary_row" in ns
    assert ns["_ruleshift_benchmark_v1_binary_row"].store_task is False
    assert ns["_ruleshift_benchmark_v1_binary_row"].evaluate_call_count == 1
    assert ns["_ruleshift_benchmark_v1_binary_row"].last_evaluation_data is not None
    assert set(ns["_ruleshift_benchmark_v1_binary_row"].last_evaluation_data["split"]) == {
        "public_leaderboard",
        "private_leaderboard",
    }

    validate_kaggle_payload(ns["payload"])
    assert set(ns["payload"]) == {
        "score",
        "numerator",
        "denominator",
        "total_episodes",
        "benchmark_version",
        "split",
        "manifest_version",
    }
    assert ns["payload"]["total_episodes"] == len(ns["leaderboard_df"])
    assert ns["payload"]["split"] == "frozen_leaderboard"
    assert "RUN_LOG_PATH" not in ns
    assert "DIAGNOSTICS_SUMMARY_PATH" not in ns
    assert "RUN_MANIFEST_PATH" not in ns
    assert "RUN_EPISODE_LEDGER_PATH" not in ns


def test_notebook_executes_public_only_without_private_mount(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("RULESHIFT_PRIVATE_DATASET_ROOT", raising=False)
    ns = _execute_notebook_cells()

    assert ns["PRIVATE_DATASET_ROOT"] is None
    assert set(ns["frozen_splits"]) == {"public_leaderboard"}
    assert set(ns["leaderboard_df"]["split"]) == {"public_leaderboard"}
    assert ns["payload"]["total_episodes"] == len(ns["leaderboard_df"])
    assert ns["payload"]["split"] == "public_leaderboard"


def test_last_code_cell_selects_binary_task():
    notebook = json.loads(_NOTEBOOK_PATH.read_text(encoding="utf-8"))
    last_code = next(cell for cell in reversed(notebook["cells"]) if cell.get("cell_type") == "code")
    magic_lines = [line.strip() for line in "".join(last_code.get("source", ())).splitlines() if line.strip().startswith("%")]

    assert magic_lines == ["%choose ruleshift_benchmark_v1_binary"]
