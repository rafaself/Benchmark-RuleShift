"""Local pre-deploy smoke tests for the official Kaggle notebook.

Validates the notebook's bootstrap assumptions, module imports, frozen split
loading, evaluation dataframe construction, @kbench.task registration, and
per-episode return shapes — all without Kaggle UI or real LLM calls.

The kbench_shim module stands in for kaggle_benchmarks locally.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
_NOTEBOOK_PATH = _REPO_ROOT / "packaging" / "kaggle" / "ruleshift_benchmark_v1_kbench.ipynb"

# Ensure src/ is on sys.path (same as conftest.py).
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# Install the kbench shim as ``kaggle_benchmarks`` before any notebook code runs.
import tests.kbench_shim as _shim  # noqa: E402

sys.modules["kaggle_benchmarks"] = _shim  # type: ignore[assignment]
import kaggle_benchmarks as kbench  # noqa: E402


# ── helpers ──────────────────────────────────────────────────────────────────


def _read_notebook_sources() -> list[str]:
    notebook = json.loads(_NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return ["".join(cell.get("source", ())) for cell in notebook["cells"]]


# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clean_kbench_registry():
    """Reset the shim registry between tests."""
    kbench.reset_registry()
    yield
    kbench.reset_registry()


# ── notebook bootstrap ───────────────────────────────────────────────────────


class TestNotebookBootstrap:
    """Cell 2: _find_repo_root and sys.path setup."""

    def test_repo_root_resolves_from_cwd(self):
        """The notebook's _find_repo_root must find this repo when cwd is the repo root."""
        # Extract and exec cell 2 logic
        sources = _read_notebook_sources()
        cell2 = sources[2]
        ns: dict = {"Path": Path, "sys": sys}
        exec(compile(cell2, "<cell-2>", "exec"), ns)
        repo_root = ns["REPO_ROOT"]
        assert repo_root == _REPO_ROOT
        assert (repo_root / "src").is_dir()
        assert (repo_root / "packaging" / "kaggle" / "frozen_artifacts_manifest.json").is_file()

    def test_src_added_to_sys_path(self):
        assert str(_SRC_DIR) in sys.path


# ── module imports ───────────────────────────────────────────────────────────


class TestNotebookImports:
    """Cell 3: all imports the notebook needs from the local package."""

    def test_kaggle_module_exports(self):
        from kaggle import (  # noqa: F401
            BinaryResponse,
            normalize_binary_response,
            normalize_narrative_response,
            score_episode,
        )

    def test_splits_module_exports(self):
        from splits import PARTITIONS, load_frozen_split  # noqa: F401
        assert len(PARTITIONS) == 3

    def test_parser_version_exported(self):
        from parser import PARSER_VERSION  # noqa: F401
        assert isinstance(PARSER_VERSION, str) and PARSER_VERSION

    def test_metric_version_exported(self):
        from metrics import METRIC_VERSION  # noqa: F401
        assert isinstance(METRIC_VERSION, str) and METRIC_VERSION

    def test_render_functions_exported(self):
        from render import render_binary_prompt, render_narrative_prompt  # noqa: F401


# ── frozen split loading ─────────────────────────────────────────────────────


class TestFrozenSplitLoading:
    """Cell 3: load_frozen_split for every partition."""

    def test_all_partitions_load(self):
        from splits import PARTITIONS, load_frozen_split

        for partition in PARTITIONS:
            records = load_frozen_split(partition)
            assert len(records) > 0, f"{partition} returned no episodes"

    def test_episodes_have_required_attributes(self):
        from splits import load_frozen_split

        records = load_frozen_split("dev")
        for record in records:
            ep = record.episode
            assert hasattr(ep, "episode_id")
            assert hasattr(ep, "difficulty")
            assert hasattr(ep, "template_id")
            assert hasattr(ep, "probe_targets")
            assert len(ep.probe_targets) == 4


# ── evaluation dataframe construction ────────────────────────────────────────


def _build_eval_df():
    """Reproduce notebook cell 4 logic."""
    import pandas as pd
    from splits import PARTITIONS, load_frozen_split
    from render import render_binary_prompt, render_narrative_prompt

    frozen_splits = {p: load_frozen_split(p) for p in PARTITIONS}

    rows: list[dict] = []
    for partition in PARTITIONS:
        for record in frozen_splits[partition]:
            ep = record.episode
            rows.append({
                "episode_id": ep.episode_id,
                "split": partition,
                "difficulty": ep.difficulty.value,
                "template_id": ep.template_id.value,
                "prompt_binary": render_binary_prompt(ep),
                "prompt_narrative": render_narrative_prompt(ep),
                "probe_targets": tuple(t.value for t in ep.probe_targets),
            })

    return pd.DataFrame(rows)


class TestEvalDataframe:
    """Cell 4: evaluation dataframe shape and contents."""

    def test_eval_df_has_expected_columns(self):
        df = _build_eval_df()
        required = {"episode_id", "split", "difficulty", "template_id",
                     "prompt_binary", "prompt_narrative", "probe_targets"}
        assert required.issubset(set(df.columns))

    def test_eval_df_row_count_matches_frozen_splits(self):
        from splits import PARTITIONS, load_frozen_split

        expected = sum(len(load_frozen_split(p)) for p in PARTITIONS)
        df = _build_eval_df()
        assert len(df) == expected

    def test_probe_targets_are_string_tuples_of_length_4(self):
        df = _build_eval_df()
        for targets in df["probe_targets"]:
            assert isinstance(targets, tuple)
            assert len(targets) == 4
            assert all(isinstance(t, str) for t in targets)

    def test_prompts_are_nonempty_strings(self):
        df = _build_eval_df()
        for col in ("prompt_binary", "prompt_narrative"):
            for value in df[col]:
                assert isinstance(value, str) and len(value) > 0


# ── @kbench.task registration ────────────────────────────────────────────────


def _register_tasks():
    """Reproduce notebook cells 8-9: register Binary and Narrative tasks."""
    from kaggle import (
        BinaryResponse,
        normalize_binary_response,
        normalize_narrative_response,
        score_episode,
    )

    @kbench.task(
        name="ruleshift_benchmark_v1_binary",
        description=(
            "Cognitive flexibility benchmark: infer a hidden rule shift from "
            "sparse contradictory evidence in a sequence of charge interactions, "
            "then predict four post-shift probe outcomes."
        ),
    )
    def ruleshift_benchmark_v1_binary(
        llm,
        prompt_binary: str,
        probe_targets: tuple,
    ) -> tuple[int, int]:
        try:
            response = llm.prompt(prompt_binary, schema=BinaryResponse)
            predictions = normalize_binary_response(response)
        except Exception:
            predictions = None
        return score_episode(predictions, probe_targets)

    @kbench.task(
        name="ruleshift_benchmark_v1_narrative",
        description=(
            "Narrative companion for RuleShift Benchmark v1. Same episodes as Binary, "
            "natural-language prompt format. Robustness evidence only — not leaderboard-scored."
        ),
    )
    def ruleshift_benchmark_v1_narrative(
        llm,
        prompt_narrative: str,
        probe_targets: tuple,
    ) -> tuple[int, int]:
        try:
            response = llm.prompt(prompt_narrative)
            predictions = normalize_narrative_response(response)
        except Exception:
            predictions = None
        return score_episode(predictions, probe_targets)

    return ruleshift_benchmark_v1_binary, ruleshift_benchmark_v1_narrative


class TestTaskRegistration:
    """Cells 8-9: @kbench.task decorator and registration."""

    def test_both_tasks_register(self):
        _register_tasks()
        registry = kbench.get_registry()
        assert "ruleshift_benchmark_v1_binary" in registry
        assert "ruleshift_benchmark_v1_narrative" in registry

    def test_task_names_and_descriptions(self):
        _register_tasks()
        registry = kbench.get_registry()
        binary = registry["ruleshift_benchmark_v1_binary"]
        narrative = registry["ruleshift_benchmark_v1_narrative"]
        assert binary.name == "ruleshift_benchmark_v1_binary"
        assert "cognitive flexibility" in binary.description.lower()
        assert narrative.name == "ruleshift_benchmark_v1_narrative"
        assert "narrative companion" in narrative.description.lower()

    def test_task_functions_are_callable(self):
        binary_task, narrative_task = _register_tasks()
        assert callable(binary_task)
        assert callable(narrative_task)


# ── dry-run evaluation ───────────────────────────────────────────────────────


class TestDryRunEvaluation:
    """End-to-end: register tasks, build eval_df, run evaluate() with stub LLM."""

    def test_binary_evaluate_returns_result_set(self):
        binary_task, _ = _register_tasks()
        df = _build_eval_df()
        result = binary_task.evaluate(llm=[kbench.llm], evaluation_data=df)
        result_df = result.as_dataframe()
        assert len(result_df) == len(df)

    def test_narrative_evaluate_returns_result_set(self):
        _, narrative_task = _register_tasks()
        df = _build_eval_df()
        result = narrative_task.evaluate(llm=[kbench.llm], evaluation_data=df)
        result_df = result.as_dataframe()
        assert len(result_df) == len(df)

    def test_stub_llm_yields_zero_scores(self):
        """With a stub LLM that returns None, every episode should score (0, 4)."""
        binary_task, _ = _register_tasks()
        df = _build_eval_df()
        result = binary_task.evaluate(llm=[kbench.llm], evaluation_data=df)
        result_df = result.as_dataframe()
        assert (result_df["num_correct"] == 0).all()
        assert (result_df["total"] == 4).all()

    def test_return_shape_is_int_pair(self):
        """Each task call must return (int, int)."""
        binary_task, narrative_task = _register_tasks()
        row = _build_eval_df().iloc[0]
        for task_fn in (binary_task, narrative_task):
            col = "prompt_binary" if "binary" in task_fn.name else "prompt_narrative"
            score = task_fn(kbench.llm, **{col: row[col], "probe_targets": row["probe_targets"]})
            assert isinstance(score, tuple) and len(score) == 2
            assert all(isinstance(v, int) for v in score)


# ── scoring contract ─────────────────────────────────────────────────────────


class TestScoringContract:
    """Cell 11: dry-run scoring checks — same assertions as the notebook."""

    def test_perfect_prediction(self):
        from kaggle import score_episode
        df = _build_eval_df()
        targets = df.iloc[0]["probe_targets"]
        assert score_episode(targets, targets) == (4, 4)

    def test_invalid_prediction(self):
        from kaggle import score_episode
        df = _build_eval_df()
        targets = df.iloc[0]["probe_targets"]
        assert score_episode(None, targets) == (0, 4)

    def test_normalize_binary_roundtrip(self):
        from kaggle import BinaryResponse, Label, normalize_binary_response
        df = _build_eval_df()
        targets = df.iloc[0]["probe_targets"]
        # Text path
        text = ", ".join(targets)
        assert normalize_binary_response(text) == targets
        # Structured path
        labels = [Label(t) for t in targets]
        structured = BinaryResponse(*labels)
        assert normalize_binary_response(structured) == targets

    def test_malformed_response_scores_zero(self):
        from kaggle import normalize_binary_response, score_episode
        df = _build_eval_df()
        targets = df.iloc[0]["probe_targets"]
        malformed = normalize_binary_response("I don't know")
        assert score_episode(malformed, targets) == (0, 4)


# ── notebook source fidelity ─────────────────────────────────────────────────


class TestNotebookSourceFidelity:
    """Verify the notebook source matches expected task signatures."""

    def test_notebook_file_exists(self):
        assert _NOTEBOOK_PATH.is_file()

    def test_notebook_contains_binary_task_decorator(self):
        sources = _read_notebook_sources()
        joined = "\n".join(sources)
        assert '@kbench.task(\n    name="ruleshift_benchmark_v1_binary"' in joined

    def test_notebook_contains_narrative_task_decorator(self):
        sources = _read_notebook_sources()
        joined = "\n".join(sources)
        assert '@kbench.task(\n    name="ruleshift_benchmark_v1_narrative"' in joined

    def test_notebook_selects_binary_for_leaderboard(self):
        sources = _read_notebook_sources()
        joined = "\n".join(sources)
        assert "%choose ruleshift_benchmark_v1_binary" in joined

    def test_notebook_does_not_choose_narrative(self):
        sources = _read_notebook_sources()
        joined = "\n".join(sources)
        assert "%choose ruleshift_benchmark_v1_narrative" not in joined

    def test_binary_task_signature_matches_eval_df_columns(self):
        """The function params (excluding llm) must match eval_df column names."""
        import inspect
        binary_task, _ = _register_tasks()
        sig = inspect.signature(binary_task.fn)
        param_names = [p for p in sig.parameters if p != "llm"]
        df = _build_eval_df()
        for name in param_names:
            assert name in df.columns, f"Binary task param '{name}' not in eval_df columns"

    def test_narrative_task_signature_matches_eval_df_columns(self):
        import inspect
        _, narrative_task = _register_tasks()
        sig = inspect.signature(narrative_task.fn)
        param_names = [p for p in sig.parameters if p != "llm"]
        df = _build_eval_df()
        for name in param_names:
            assert name in df.columns, f"Narrative task param '{name}' not in eval_df columns"
