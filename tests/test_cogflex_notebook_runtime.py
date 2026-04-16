import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.private_cogflex_bundle import write_private_bundle


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "kaggle/notebook/cogflex_notebook_task.ipynb"
PUBLIC_ROWS_PATH = ROOT / "kaggle/dataset/public/public_leaderboard_rows.json"
TEST_ROWS_PATH = ROOT / "kaggle/dataset/test/test_leaderboard_rows.json"


class _BenchStub:
    @staticmethod
    def task(*args, **kwargs):
        def decorator(fn):
            return fn

        return decorator


class FakeLLM:
    def __init__(self, final_response: object, *, final_call_index: int) -> None:
        self.final_response = final_response
        self.final_call_index = final_call_index
        self.calls: list[str] = []

    def prompt(self, prompt: str, schema: object | None = None) -> object:
        self.calls.append(prompt)
        return self.final_response if len(self.calls) == self.final_call_index else "ack"


class _FakeResultsFrame:
    def __init__(self, results: list[dict[str, object]]) -> None:
        self.result = results

    def reset_index(self, drop: bool = False) -> "_FakeResultsFrame":
        return self

    def __len__(self) -> int:
        return len(self.result)


class FakeRuns:
    def __init__(self, results: list[dict[str, object]]) -> None:
        self._results = results

    def as_dataframe(self) -> _FakeResultsFrame:
        return _FakeResultsFrame(self._results)


class RecordingTaskRunner:
    def __init__(self, results: list[dict[str, object]]) -> None:
        self._results = results
        self.call_args: list[dict[str, object]] = []
        self.evaluate_calls: list[dict[str, object]] = []

    def __call__(
        self,
        llm: object,
        turns: list[str],
        response_spec: dict[str, object],
        final_probe_targets: tuple[str, ...],
        probe_metadata: tuple[dict[str, object], ...] | None = None,
        probe_annotations: tuple[str, ...] | None = None,
    ) -> dict[str, object]:
        result = self._results[len(self.call_args)]
        self.call_args.append(
            {
                "llm": llm,
                "turns": turns,
                "response_spec": response_spec,
                "final_probe_targets": final_probe_targets,
                "probe_metadata": probe_metadata,
                "probe_annotations": probe_annotations,
            }
        )
        return result

    def evaluate(self, llm: list[object], evaluation_data: object) -> FakeRuns:
        self.evaluate_calls.append({"llm": llm, "evaluation_data": evaluation_data})
        return FakeRuns(self._results)


def _load_code_cells() -> dict[str, str]:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return {
        cell["id"]: "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    }


def load_notebook_namespace() -> dict[str, object]:
    code_cells = _load_code_cells()
    fake_kbench = types.ModuleType("kaggle_benchmarks")
    fake_kbench.task = _BenchStub.task
    fake_pd = types.ModuleType("pandas")
    fake_pd.DataFrame = lambda rows: rows
    namespace: dict[str, object] = {"Path": Path}
    with patch.dict(sys.modules, {"kaggle_benchmarks": fake_kbench, "pandas": fake_pd}):
        exec(code_cells["cell-bootstrap"], namespace)
        namespace["EVAL_SPLIT"] = "public"
        namespace["DATASET_ROOT"] = PUBLIC_ROWS_PATH.parent
        namespace["ROWS_PATH"] = PUBLIC_ROWS_PATH
        exec(code_cells["cell-runtime-types"], namespace)
        exec(code_cells["cell-runtime-normalize"], namespace)
        exec(code_cells["cell-runtime-parse"], namespace)
        exec(code_cells["cell-runtime-validate"], namespace)
        exec(code_cells["cell-runtime-score"], namespace)
        runtime_load_prefix = code_cells["cell-runtime-load"].split("leaderboard_rows = load_selected_rows()", 1)[0]
        exec(runtime_load_prefix, namespace)
        exec(code_cells["cell-task"], namespace)
    return namespace


class CogflexNotebookRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.namespace = load_notebook_namespace()

    @staticmethod
    def _sample_results() -> list[dict[str, object]]:
        return [
            {
                "numerator": 5,
                "denominator": 5,
                "incongruent_numerator": 2,
                "incongruent_denominator": 2,
                "congruent_numerator": 3,
                "congruent_denominator": 3,
                "first_probe_numerator": 1,
                "first_probe_denominator": 1,
                "shift_window_numerator": 2,
                "shift_window_denominator": 2,
                "obsolete_rule_error_numerator": 0,
                "obsolete_rule_error_denominator": 5,
                "requires_switch_numerator": 2,
                "requires_switch_denominator": 2,
                "scorable": True,
            },
            {
                "numerator": 3,
                "denominator": 5,
                "incongruent_numerator": 1,
                "incongruent_denominator": 2,
                "congruent_numerator": 2,
                "congruent_denominator": 3,
                "first_probe_numerator": 0,
                "first_probe_denominator": 1,
                "shift_window_numerator": 1,
                "shift_window_denominator": 2,
                "obsolete_rule_error_numerator": 1,
                "obsolete_rule_error_denominator": 5,
                "requires_switch_numerator": 1,
                "requires_switch_denominator": 2,
                "scorable": True,
            },
        ]

    def _public_rows(self) -> list[dict[str, object]]:
        return self.namespace["_load_rows"](PUBLIC_ROWS_PATH)

    def _test_rows(self) -> list[dict[str, object]]:
        self.namespace["EVAL_SPLIT"] = "test"
        try:
            return self.namespace["_load_rows"](TEST_ROWS_PATH)
        finally:
            self.namespace["EVAL_SPLIT"] = "public"

    def test_run_flexible_task_ignores_schema_metadata_in_response_spec(self) -> None:
        row = self._public_rows()[0]
        response_spec_with_metadata = json.loads(json.dumps(row["inference"]["response_spec"]))
        response_spec_without_metadata = {
            key: value
            for key, value in response_spec_with_metadata.items()
            if key not in {"schema_version", "output_schema"}
        }
        expected_labels = list(row["scoring"]["final_probe_targets"])
        probe_metadata = row["scoring"].get("probe_metadata")
        probe_annotations = row["scoring"].get("probe_annotations")

        llm_with_metadata = FakeLLM({"ordered_labels": expected_labels}, final_call_index=len(row["inference"]["turns"]))
        llm_without_metadata = FakeLLM(
            {"ordered_labels": expected_labels},
            final_call_index=len(row["inference"]["turns"]),
        )

        result_with_metadata = self.namespace["run_flexible_task"](
            llm_with_metadata,
            row["inference"]["turns"],
            response_spec_with_metadata,
            tuple(expected_labels),
            probe_metadata=None if probe_metadata is None else tuple(probe_metadata),
            probe_annotations=None if probe_annotations is None else tuple(probe_annotations),
        )
        result_without_metadata = self.namespace["run_flexible_task"](
            llm_without_metadata,
            row["inference"]["turns"],
            response_spec_without_metadata,
            tuple(expected_labels),
            probe_metadata=None if probe_metadata is None else tuple(probe_metadata),
            probe_annotations=None if probe_annotations is None else tuple(probe_annotations),
        )

        self.assertEqual(result_with_metadata, result_without_metadata)
        self.assertEqual(llm_with_metadata.calls[-1], llm_without_metadata.calls[-1])

    def test_attach_private_scoring_uses_runtime_signature_instead_of_literal_inference_match(self) -> None:
        self.namespace["EVAL_SPLIT"] = "private"
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                bundle_dir = Path(tmpdir) / "bundle"
                bundle_paths = write_private_bundle(bundle_dir)
                private_rows = self.namespace["_load_rows"](bundle_paths["rows"])
                answer_key = json.loads(bundle_paths["answer_key"].read_text(encoding="utf-8"))

                response_spec = answer_key["episodes"][0]["inference"]["response_spec"]
                response_spec["schema_version"] = "banished"
                response_spec.pop("output_schema", None)
                response_spec["unused_runtime_metadata"] = "ignored"
                bundle_paths["answer_key"].write_text(json.dumps(answer_key, indent=2) + "\n", encoding="utf-8")

                self.namespace["PRIVATE_DATASET_ROOT"] = bundle_dir
                self.namespace["PRIVATE_SCORING_DATASET_ROOT"] = bundle_dir
                attached_rows = self.namespace["_attach_private_scoring"](private_rows)
                self.assertIn("scoring", attached_rows[0])

                answer_key["episodes"][0]["inference"]["response_spec"]["probe_count"] += 1
                bundle_paths["answer_key"].write_text(json.dumps(answer_key, indent=2) + "\n", encoding="utf-8")

                with self.assertRaisesRegex(RuntimeError, "answer key inference mismatch"):
                    self.namespace["_attach_private_scoring"](private_rows)
        finally:
            self.namespace["EVAL_SPLIT"] = "public"
            self.namespace["PRIVATE_DATASET_ROOT"] = self.namespace["DEFAULT_PRIVATE_DATASET_ROOT"]
            self.namespace["PRIVATE_SCORING_DATASET_ROOT"] = self.namespace["DEFAULT_PRIVATE_SCORING_DATASET_ROOT"]

    def test_load_selected_rows_enforces_public_row_count(self) -> None:
        rows = self._public_rows()
        with tempfile.TemporaryDirectory() as tmpdir:
            rows_path = Path(tmpdir) / "public-rows.json"
            rows_path.write_text(json.dumps(rows[:-1], indent=2) + "\n", encoding="utf-8")
            self.namespace["EVAL_SPLIT"] = "public"
            self.namespace["ROWS_PATH"] = rows_path

            with self.assertRaisesRegex(RuntimeError, "public split row count mismatch: expected 20, found 19"):
                self.namespace["load_selected_rows"]()

            rows_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
            loaded_rows = self.namespace["load_selected_rows"]()

        self.assertEqual(len(loaded_rows), 20)

    def test_load_selected_rows_enforces_private_row_count(self) -> None:
        self.namespace["EVAL_SPLIT"] = "private"
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                bundle_dir = Path(tmpdir) / "bundle"
                bundle_paths = write_private_bundle(bundle_dir)
                private_rows = self.namespace["_load_rows"](bundle_paths["rows"])
                rows_path = Path(tmpdir) / "private-rows.json"
                rows_path.write_text(json.dumps(private_rows[:-1], indent=2) + "\n", encoding="utf-8")
                self.namespace["ROWS_PATH"] = rows_path

                with self.assertRaisesRegex(RuntimeError, "private split row count mismatch: expected 96, found 95"):
                    self.namespace["load_selected_rows"]()

                rows_path.write_text(json.dumps(private_rows, indent=2) + "\n", encoding="utf-8")
                loaded_rows = self.namespace["load_selected_rows"]()
        finally:
            self.namespace["EVAL_SPLIT"] = "public"
            self.namespace["ROWS_PATH"] = PUBLIC_ROWS_PATH

        self.assertEqual(len(loaded_rows), 96)

    def test_load_selected_rows_enforces_test_row_count(self) -> None:
        rows = self._test_rows()
        with tempfile.TemporaryDirectory() as tmpdir:
            rows_path = Path(tmpdir) / "test-rows.json"
            rows_path.write_text(json.dumps(rows * 2, indent=2) + "\n", encoding="utf-8")
            self.namespace["EVAL_SPLIT"] = "test"
            self.namespace["ROWS_PATH"] = rows_path

            with self.assertRaisesRegex(RuntimeError, "test split row count mismatch: expected 1, found 2"):
                self.namespace["load_selected_rows"]()

            rows_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
            loaded_rows = self.namespace["load_selected_rows"]()

        self.namespace["EVAL_SPLIT"] = "public"
        self.namespace["ROWS_PATH"] = PUBLIC_ROWS_PATH
        self.assertEqual(len(loaded_rows), 1)

    def test_summarize_suite_benchmark_compact_summary_matches_debug_score(self) -> None:
        rows = self._public_rows()
        selected_rows = [rows[0], next(row for row in rows if row["analysis"]["suite_task_id"] != rows[0]["analysis"]["suite_task_id"])]
        results = self._sample_results()

        compact_summary = self.namespace["summarize_suite_benchmark"](FakeRuns(results), selected_rows)
        debug_summary = self.namespace["summarize_suite_benchmark"](FakeRuns(results), selected_rows, include_debug=True)

        self.assertEqual(compact_summary["score"], debug_summary["score"])
        self.assertEqual(compact_summary["macro_accuracy"], debug_summary["macro_accuracy"])
        self.assertNotIn("per_task_accuracy", compact_summary)
        self.assertNotIn("per_task_metrics", compact_summary)
        self.assertNotIn("structure_family_accuracy", compact_summary)
        self.assertNotIn("difficulty_bin_accuracy", compact_summary)
        self.assertIn("per_task_accuracy", debug_summary)

    def test_registered_task_uses_tabular_evaluation_for_public_split_and_prints_compact_summary(self) -> None:
        rows = self._public_rows()
        selected_rows = [rows[0], next(row for row in rows if row["analysis"]["suite_task_id"] != rows[0]["analysis"]["suite_task_id"])]
        results = self._sample_results()

        self.namespace["scored_rows"] = selected_rows
        self.namespace["df"] = selected_rows
        runner = RecordingTaskRunner(results)
        self.namespace["run_flexible_task"] = runner

        with patch("builtins.print") as mock_print:
            returned_score = self.namespace["cogflex_suite_flexible"](object())

        self.assertEqual(len(runner.evaluate_calls), 1)
        self.assertEqual(runner.evaluate_calls[0]["evaluation_data"], selected_rows)
        self.assertEqual(runner.call_args, [])
        self.assertEqual(mock_print.call_count, 1)
        printed_summary = json.loads(mock_print.call_args.args[0])
        self.assertEqual(returned_score, printed_summary["score"])
        self.assertEqual(
            set(printed_summary),
            {
                "score",
                "protocol_valid_rate",
                "scorable_episodes",
                "episodes",
                "macro_accuracy",
                "incongruent_accuracy",
                "first_probe_accuracy",
                "obsolete_rule_error_rate",
            },
        )

    def test_registered_task_uses_tabular_evaluation_for_test_split(self) -> None:
        selected_rows = self._test_rows()
        results = [
            {
                "numerator": 5,
                "denominator": 5,
                "incongruent_numerator": 2,
                "incongruent_denominator": 2,
                "congruent_numerator": 3,
                "congruent_denominator": 3,
                "first_probe_numerator": 1,
                "first_probe_denominator": 1,
                "shift_window_numerator": 2,
                "shift_window_denominator": 2,
                "obsolete_rule_error_numerator": 0,
                "obsolete_rule_error_denominator": 5,
                "requires_switch_numerator": 2,
                "requires_switch_denominator": 2,
                "scorable": True,
            }
        ]

        self.namespace["EVAL_SPLIT"] = "test"
        self.namespace["scored_rows"] = selected_rows
        self.namespace["df"] = selected_rows
        runner = RecordingTaskRunner(results)
        self.namespace["run_flexible_task"] = runner

        try:
            with patch("builtins.print") as mock_print:
                returned_score = self.namespace["cogflex_suite_flexible"](object())
        finally:
            self.namespace["EVAL_SPLIT"] = "public"

        self.assertEqual(len(runner.evaluate_calls), 1)
        self.assertEqual(runner.evaluate_calls[0]["evaluation_data"], selected_rows)
        self.assertEqual(runner.call_args, [])
        printed_summary = json.loads(mock_print.call_args.args[0])
        self.assertEqual(returned_score, printed_summary["score"])

    def test_registered_task_runs_private_split_in_memory_without_tabular_evaluation(self) -> None:
        results = [
            {
                "numerator": 5,
                "denominator": 5,
                "incongruent_numerator": 2,
                "incongruent_denominator": 2,
                "congruent_numerator": 3,
                "congruent_denominator": 3,
                "first_probe_numerator": 1,
                "first_probe_denominator": 1,
                "shift_window_numerator": 2,
                "shift_window_denominator": 2,
                "obsolete_rule_error_numerator": 0,
                "obsolete_rule_error_denominator": 5,
                "requires_switch_numerator": 2,
                "requires_switch_denominator": 2,
                "scorable": True,
            },
            {
                "numerator": 3,
                "denominator": 5,
                "incongruent_numerator": 1,
                "incongruent_denominator": 2,
                "congruent_numerator": 2,
                "congruent_denominator": 3,
                "first_probe_numerator": 0,
                "first_probe_denominator": 1,
                "shift_window_numerator": 1,
                "shift_window_denominator": 2,
                "obsolete_rule_error_numerator": 1,
                "obsolete_rule_error_denominator": 5,
                "requires_switch_numerator": 1,
                "requires_switch_denominator": 2,
                "scorable": True,
            },
        ]
        runner = RecordingTaskRunner(results)

        self.namespace["run_flexible_task"] = runner
        self.namespace["EVAL_SPLIT"] = "private"
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                bundle_dir = Path(tmpdir) / "bundle"
                bundle_paths = write_private_bundle(bundle_dir)
                private_rows = self.namespace["_load_rows"](bundle_paths["rows"])
                self.namespace["PRIVATE_DATASET_ROOT"] = bundle_dir
                self.namespace["PRIVATE_SCORING_DATASET_ROOT"] = bundle_dir
                scored_rows = self.namespace["_attach_private_scoring"](private_rows)
                self.namespace["scored_rows"] = scored_rows[:2]
                self.namespace["df"] = object()

                with patch("builtins.print") as mock_print:
                    returned_score = self.namespace["cogflex_suite_flexible"](object())
        finally:
            self.namespace["EVAL_SPLIT"] = "public"
            self.namespace["PRIVATE_DATASET_ROOT"] = self.namespace["DEFAULT_PRIVATE_DATASET_ROOT"]
            self.namespace["PRIVATE_SCORING_DATASET_ROOT"] = self.namespace["DEFAULT_PRIVATE_SCORING_DATASET_ROOT"]

        self.assertEqual(runner.evaluate_calls, [])
        self.assertEqual(len(runner.call_args), 2)
        self.assertEqual(mock_print.call_count, 1)
        printed_summary = json.loads(mock_print.call_args.args[0])
        self.assertEqual(returned_score, printed_summary["score"])
        self.assertEqual(
            set(printed_summary),
            {
                "score",
                "protocol_valid_rate",
                "scorable_episodes",
                "episodes",
                "macro_accuracy",
                "incongruent_accuracy",
                "first_probe_accuracy",
                "obsolete_rule_error_rate",
            },
        )
