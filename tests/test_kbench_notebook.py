"""Local pre-deploy smoke tests for the official Kaggle notebook.

Validates the notebook's bootstrap assumptions, module imports, frozen split
loading, evaluation dataframe construction, @kbench.task registration, and
per-episode return shapes — all without Kaggle UI or real LLM calls.

Also contains TestNotebookEndToEnd, which executes every code cell in the
notebook in order (using the kbench shim) and verifies all data-boundary
contracts:
  - ruleshift_benchmark_v1_binary is the only kbench task (Narrative is plain)
  - binary .evaluate() and the Narrative loop run over leaderboard_df only
  - local validation cells run over dev_df only
  - dev rows never reach build_kaggle_payload
  - %choose selects ruleshift_benchmark_v1_binary as the last cell

The kbench_shim module stands in for kaggle_benchmarks locally.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
_NOTEBOOK_PATH = _REPO_ROOT / "packaging" / "kaggle" / "ruleshift_notebook_task.ipynb"

# Ensure src/ is on sys.path (same as conftest.py).
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# Install the kbench shim as ``kaggle_benchmarks`` before any notebook code runs.
import tests.kbench_shim as _shim  # noqa: E402

sys.modules["kaggle_benchmarks"] = _shim  # type: ignore[assignment]
import kaggle_benchmarks as kbench  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _read_notebook_sources() -> list[str]:
    notebook = json.loads(_NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return ["".join(cell.get("source", ())) for cell in notebook["cells"]]


def _cell_source(cell_id: str) -> str:
    """Return the source of the cell with the given id."""
    notebook = json.loads(_NOTEBOOK_PATH.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell.get("id") == cell_id:
            return "".join(cell.get("source", ()))
    raise KeyError(f"Cell id {cell_id!r} not found in notebook")


def _execute_notebook_cells() -> dict:
    """Execute every code cell in notebook order (Jupyter magic lines skipped).

    Returns the final execution namespace.  Stubs out Jupyter's display()
    so the dev-validation cells run without a Jupyter kernel.
    """
    cells = json.loads(_NOTEBOOK_PATH.read_text(encoding="utf-8"))["cells"]
    ns: dict = {
        "__builtins__": __import__("builtins"),
        "display": lambda *a, **kw: None,   # stub Jupyter display()
    }
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", ()))
        # Strip Jupyter magic lines (%choose, %matplotlib, etc.) so the
        # remaining Python is valid for exec().
        filtered = "\n".join(
            line for line in source.splitlines() if not line.strip().startswith("%")
        )
        if not filtered.strip():
            continue
        exec(compile(filtered, f"<{cell['id']}>", "exec"), ns)  # noqa: S102
    return ns


def _build_eval_df():
    """Reproduce the combined-splits eval_df (all partitions) for unit tests."""
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


def _register_binary_task():
    """Register only the Binary task as @kbench.task — matching the notebook."""
    from kaggle import (
        BinaryResponse,
        normalize_binary_response,
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

    return ruleshift_benchmark_v1_binary


def _make_narrative_fn():
    """Return the Narrative function as a plain callable — not a kbench task."""
    from kaggle import normalize_narrative_response, score_episode

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

    return ruleshift_benchmark_v1_narrative


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_kbench_registry():
    """Reset the shim registry between tests."""
    kbench.reset_registry()
    yield
    kbench.reset_registry()


# ---------------------------------------------------------------------------
# notebook bootstrap
# ---------------------------------------------------------------------------


class TestNotebookBootstrap:
    """Cell 2: _find_repo_root and sys.path setup."""

    def test_repo_root_resolves_from_cwd(self):
        """The notebook's _find_repo_root must find this repo when cwd is the repo root."""
        cell2 = _cell_source("cell-2")
        ns: dict = {"Path": Path, "sys": sys}
        exec(compile(cell2, "<cell-2>", "exec"), ns)  # noqa: S102
        repo_root = ns["REPO_ROOT"]
        assert repo_root == _REPO_ROOT
        assert (repo_root / "src").is_dir()
        assert (repo_root / "packaging" / "kaggle" / "frozen_artifacts_manifest.json").is_file()

    def test_src_added_to_sys_path(self):
        assert str(_SRC_DIR) in sys.path

    def test_missing_private_dataset_raises_clear_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        cell2 = _cell_source("cell-2")
        ns: dict = {"Path": Path, "sys": sys}
        monkeypatch.delenv("RULESHIFT_PRIVATE_DATASET_ROOT", raising=False)
        with pytest.raises(
            FileNotFoundError,
            match="Private evaluation dataset is not attached",
        ):
            exec(compile(cell2, "<cell-2>", "exec"), ns)  # noqa: S102


# ---------------------------------------------------------------------------
# module imports
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# frozen split loading
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# evaluation dataframe construction
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# @kbench.task registration — Binary only
# ---------------------------------------------------------------------------


class TestTaskRegistration:
    """cell-binary-def: only Binary is registered as a kbench task."""

    def test_only_binary_task_is_registered(self):
        _register_binary_task()
        registry = kbench.get_registry()
        assert "ruleshift_benchmark_v1_binary" in registry
        assert "ruleshift_benchmark_v1_narrative" not in registry, (
            "Narrative must not be registered as a kbench task"
        )

    def test_binary_task_name_and_description(self):
        _register_binary_task()
        registry = kbench.get_registry()
        binary = registry["ruleshift_benchmark_v1_binary"]
        assert binary.name == "ruleshift_benchmark_v1_binary"
        assert "cognitive flexibility" in binary.description.lower()

    def test_binary_task_is_callable(self):
        binary_task = _register_binary_task()
        assert callable(binary_task)

    def test_narrative_is_callable_as_plain_function(self):
        narrative_fn = _make_narrative_fn()
        assert callable(narrative_fn)
        assert not hasattr(narrative_fn, "evaluate"), (
            "Narrative must not have .evaluate() — it is not a kbench task"
        )


# ---------------------------------------------------------------------------
# dry-run evaluation
# ---------------------------------------------------------------------------


class TestDryRunEvaluation:
    """End-to-end: register Binary task, build eval_df, run evaluate() with stub LLM."""

    def test_binary_evaluate_returns_result_set(self):
        binary_task = _register_binary_task()
        df = _build_eval_df()
        result = binary_task.evaluate(llm=[kbench.llm], evaluation_data=df)
        result_df = result.as_dataframe()
        assert len(result_df) == len(df)

    def test_stub_llm_yields_zero_scores(self):
        """With a stub LLM that returns None, every episode should score (0, 4)."""
        binary_task = _register_binary_task()
        df = _build_eval_df()
        result = binary_task.evaluate(llm=[kbench.llm], evaluation_data=df)
        result_df = result.as_dataframe()
        assert (result_df["num_correct"] == 0).all()
        assert (result_df["total"] == 4).all()

    def test_binary_return_shape_is_int_pair(self):
        """Each binary task call must return (int, int)."""
        binary_task = _register_binary_task()
        row = _build_eval_df().iloc[0]
        score = binary_task(kbench.llm, prompt_binary=row["prompt_binary"],
                            probe_targets=row["probe_targets"])
        assert isinstance(score, tuple) and len(score) == 2
        assert all(isinstance(v, int) for v in score)

    def test_narrative_plain_function_return_shape(self):
        """Narrative plain function must also return (int, int)."""
        narrative_fn = _make_narrative_fn()
        row = _build_eval_df().iloc[0]
        score = narrative_fn(kbench.llm, prompt_narrative=row["prompt_narrative"],
                             probe_targets=row["probe_targets"])
        assert isinstance(score, tuple) and len(score) == 2
        assert all(isinstance(v, int) for v in score)


# ---------------------------------------------------------------------------
# scoring contract
# ---------------------------------------------------------------------------


class TestScoringContract:
    """cell-11: dry-run scoring checks — same assertions as the notebook."""

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


# ---------------------------------------------------------------------------
# notebook source fidelity
# ---------------------------------------------------------------------------


class TestNotebookSourceFidelity:
    """Verify the notebook source matches the implemented flow."""

    def test_notebook_file_exists(self):
        assert _NOTEBOOK_PATH.is_file()

    def test_notebook_contains_binary_task_decorator(self):
        sources = _read_notebook_sources()
        joined = "\n".join(sources)
        assert '@kbench.task(\n    name="ruleshift_benchmark_v1_binary"' in joined

    def test_notebook_narrative_is_not_a_kbench_task(self):
        """Narrative must be a plain function — no @kbench.task decorator."""
        sources = _read_notebook_sources()
        joined = "\n".join(sources)
        assert '@kbench.task(\n    name="ruleshift_benchmark_v1_narrative"' not in joined, (
            "Narrative must NOT be decorated with @kbench.task"
        )

    def test_notebook_narrative_plain_function_is_defined(self):
        """Narrative must still exist as a callable plain function."""
        sources = _read_notebook_sources()
        joined = "\n".join(sources)
        assert "def ruleshift_benchmark_v1_narrative(" in joined

    def test_notebook_selects_binary_for_leaderboard(self):
        sources = _read_notebook_sources()
        joined = "\n".join(sources)
        assert "%choose ruleshift_benchmark_v1_binary" in joined

    def test_notebook_does_not_choose_narrative(self):
        sources = _read_notebook_sources()
        joined = "\n".join(sources)
        assert "%choose ruleshift_benchmark_v1_narrative" not in joined

    def test_notebook_uses_mount_only_private_dataset_resolution(self):
        joined = "\n".join(_read_notebook_sources())
        assert "resolve_private_dataset_root" in joined
        assert 'frozen_splits["private_leaderboard"] = load_private_split(PRIVATE_DATASET_ROOT)' in joined
        assert "packaging/kaggle/private/private_episodes.json" not in joined

    def test_notebook_defines_explicit_leaderboard_partition_boundary(self):
        joined = "\n".join(_read_notebook_sources())
        assert '_DEV_PARTITION = "dev"' in joined
        assert '_LEADERBOARD_PARTITIONS = ("public_leaderboard", "private_leaderboard")' in joined
        assert '_PUBLIC_RUNTIME_PARTITIONS = (_DEV_PARTITION, "public_leaderboard")' in joined
        assert 'for _partition in _LEADERBOARD_PARTITIONS' in joined

    def test_binary_task_signature_matches_eval_df_columns(self):
        """The function params (excluding llm) must match eval_df column names."""
        import inspect
        binary_task = _register_binary_task()
        sig = inspect.signature(binary_task.fn)
        param_names = [p for p in sig.parameters if p != "llm"]
        df = _build_eval_df()
        for name in param_names:
            assert name in df.columns, f"Binary task param '{name}' not in eval_df columns"

    def test_narrative_function_signature_matches_eval_df_columns(self):
        """Narrative plain function params (excluding llm) must match eval_df columns."""
        import inspect
        narrative_fn = _make_narrative_fn()
        sig = inspect.signature(narrative_fn)
        param_names = [p for p in sig.parameters if p != "llm"]
        df = _build_eval_df()
        for name in param_names:
            assert name in df.columns, f"Narrative param '{name}' not in eval_df columns"


# ---------------------------------------------------------------------------
# end-to-end notebook execution — data-boundary verification
# ---------------------------------------------------------------------------


class TestNotebookEndToEnd:
    """Execute every notebook cell in order and verify all data-flow boundaries.

    Uses _execute_notebook_cells() which runs the real notebook source with
    the kbench shim (stub LLM, no API calls).

    Boundaries verified:
      1. dev_df contains only dev rows
      2. leaderboard_df contains no dev rows, no overlap with dev_df
      3. Binary .evaluate() receives a DataFrame without a split column
      4. Binary result row count == leaderboard_df row count
      5. Narrative loop result row count == leaderboard_df row count
      6. Narrative result DataFrame has no split column
      7. Dev-validation binary result count == dev_df row count
      8. Dev-validation narrative result count == dev_df row count
      9. Only ruleshift_benchmark_v1_binary is in the kbench registry
     10. validate_kaggle_payload passes on the emitted payload
     11. Payload episode counts match leaderboard_df
     12. %choose cell is last and selects ruleshift_benchmark_v1_binary
    """

    @pytest.fixture
    def ns(self):
        """Execute all notebook cells and return the namespace."""
        return _execute_notebook_cells()

    # ── 1-2: dataframe construction boundaries ────────────────────────────────

    def test_dev_df_contains_only_dev_rows(self, ns):
        dev_df = ns["dev_df"]
        assert len(dev_df) > 0
        assert (dev_df["split"] == "dev").all(), (
            "dev_df must contain only dev-split rows"
        )

    def test_leaderboard_df_contains_no_dev_rows(self, ns):
        leaderboard_df = ns["leaderboard_df"]
        assert len(leaderboard_df) > 0
        assert "dev" not in leaderboard_df["split"].values, (
            "leaderboard_df must never contain dev rows"
        )

    def test_leaderboard_df_contains_only_public_and_private_rows(self, ns):
        leaderboard_df = ns["leaderboard_df"]
        assert set(leaderboard_df["split"]) == {
            "public_leaderboard",
            "private_leaderboard",
        }

    def test_leaderboard_df_has_no_episode_overlap_with_dev_df(self, ns):
        dev_ids = set(ns["dev_df"]["episode_id"])
        lb_ids = set(ns["leaderboard_df"]["episode_id"])
        assert dev_ids.isdisjoint(lb_ids), (
            "dev_df and leaderboard_df must have no episode_id overlap"
        )

    # ── 3-6: official leaderboard evaluation boundaries ───────────────────────

    def test_binary_eval_df_has_no_split_column(self, ns):
        """split must be dropped from the DataFrame passed to Binary .evaluate()."""
        _binary_eval_df = ns["_binary_eval_df"]
        assert "split" not in _binary_eval_df.columns, (
            "split column must be absent from _binary_eval_df; "
            "leaderboard membership is set at construction time, not by split values"
        )

    def test_official_binary_result_count_matches_leaderboard(self, ns):
        result_df = ns["binary_results"].as_dataframe()
        assert len(result_df) == len(ns["leaderboard_df"]), (
            "Binary result row count must equal leaderboard_df row count"
        )

    def test_official_narrative_result_count_matches_leaderboard(self, ns):
        narrative_results_df = ns["narrative_results_df"]
        assert narrative_results_df is not None
        assert len(narrative_results_df) == len(ns["leaderboard_df"]), (
            "Narrative result row count must equal leaderboard_df row count"
        )

    def test_narrative_results_df_has_no_split_column(self, ns):
        """Narrative loop selects only prompt_narrative and probe_targets; no split."""
        narrative_results_df = ns["narrative_results_df"]
        assert "split" not in narrative_results_df.columns, (
            "narrative_results_df must not carry a split column"
        )

    # ── 7-8: local dev-validation boundaries ─────────────────────────────────

    def test_dev_binary_validation_count_matches_dev_df(self, ns):
        _dev_binary_results = ns["_dev_binary_results"]
        assert _dev_binary_results is not None, (
            "Dev binary validation must produce results"
        )
        dev_binary_df = _dev_binary_results.as_dataframe()
        assert len(dev_binary_df) == len(ns["dev_df"]), (
            "Dev binary validation row count must equal dev_df row count"
        )

    def test_dev_narrative_validation_count_matches_dev_df(self, ns):
        _dev_narrative_df = ns["_dev_narrative_df"]
        assert _dev_narrative_df is not None, (
            "Dev narrative validation must produce results"
        )
        assert len(_dev_narrative_df) == len(ns["dev_df"]), (
            "Dev narrative validation row count must equal dev_df row count"
        )

    # ── 9: kbench registry boundary ──────────────────────────────────────────

    def test_only_binary_task_in_kbench_registry(self):
        """After executing the notebook, only Binary is a registered kbench task."""
        _execute_notebook_cells()
        registry = kbench.get_registry()
        assert "ruleshift_benchmark_v1_binary" in registry, (
            "ruleshift_benchmark_v1_binary must be in the kbench registry"
        )
        assert "ruleshift_benchmark_v1_narrative" not in registry, (
            "Narrative must not appear in the kbench registry"
        )

    # ── 10-11: payload boundaries ─────────────────────────────────────────────

    def test_canonical_payload_validates_cleanly(self, ns):
        from kaggle import validate_kaggle_payload
        validate_kaggle_payload(ns["payload"])  # must not raise

    def test_payload_episode_counts_match_leaderboard(self, ns):
        payload = ns["payload"]
        n = len(ns["leaderboard_df"])
        assert payload["primary_result"]["total_episodes"] == n, (
            "payload primary_result episode count must equal leaderboard_df row count"
        )
        assert payload["narrative_result"]["total_episodes"] == n, (
            "payload narrative_result episode count must equal leaderboard_df row count"
        )

    def test_payload_binary_and_narrative_episode_counts_are_aligned(self, ns):
        comp = ns["payload"]["comparison"]
        assert comp["episode_count_aligned"] is True
        assert comp["binary_total_episodes"] == comp["narrative_total_episodes"]

    # ── 12: %choose boundary ─────────────────────────────────────────────────

    def test_choose_cell_is_last_and_selects_binary(self):
        cells = json.loads(_NOTEBOOK_PATH.read_text(encoding="utf-8"))["cells"]
        last_code = next(
            c for c in reversed(cells) if c.get("cell_type") == "code"
        )
        source = "".join(last_code.get("source", ()))
        magic_lines = [
            line.strip() for line in source.splitlines()
            if line.strip().startswith("%")
        ]
        assert magic_lines == ["%choose ruleshift_benchmark_v1_binary"], (
            f"Last code cell must contain exactly '%choose ruleshift_benchmark_v1_binary', got magic lines: {magic_lines!r}"
        )
