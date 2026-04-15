import contextlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.private_cogflex_bundle import public_fixture, write_private_bundle


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "kaggle/notebook/cogflex_notebook_task.ipynb"
PUBLIC_ROWS_PATH = ROOT / "kaggle/dataset/public/public_leaderboard_rows.json"


class _BenchStub:
    @staticmethod
    def task(*args, **kwargs):
        def decorator(fn):
            return fn

        return decorator


def _load_code_cells() -> dict[str, str]:
    """Load notebook code cells keyed by notebook cell ID.

    Returns:
        A mapping from notebook cell IDs to the concatenated code in each cell.

    """
    notebook = _load_notebook()
    return {
        cell["id"]: "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    }


def _load_notebook() -> dict[str, object]:
    """Load the CogFlex notebook JSON payload from disk.

    Returns:
        The parsed notebook document.

    """
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def load_bootstrap_namespace() -> dict[str, object]:
    """Execute the notebook bootstrap cell inside a controlled test namespace.

    Returns:
        The namespace populated by the bootstrap cell execution.

    """
    code_cells = _load_code_cells()
    fake_kbench = types.ModuleType("kaggle_benchmarks")
    fake_kbench.task = _BenchStub.task
    fake_pd = types.ModuleType("pandas")

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_root = Path(tmpdir)
        (dataset_root / "public_leaderboard_rows.json").write_text("[]", encoding="utf-8")
        namespace: dict[str, object] = {}
        with patch.dict(
            sys.modules,
            {"kaggle_benchmarks": fake_kbench, "pandas": fake_pd},
        ), patch.dict(
            os.environ,
            {
                "COGFLEX_EVAL_SPLIT": "public",
                "COGFLEX_DATASET_ROOT": str(dataset_root),
                "COGFLEX_PRIVATE_DATASET_ROOT": "",
                "COGFLEX_PRIVATE_ANSWER_KEY_PATH": "",
            },
            clear=False,
        ):
            exec(code_cells["cell-bootstrap"], namespace)
    return namespace


def load_notebook_namespace() -> dict[str, object]:
    """Execute the notebook runtime support cells for test reuse.

    Returns:
        The namespace populated with the notebook runtime helpers.

    """
    code_cells = _load_code_cells()
    fake_kbench = types.ModuleType("kaggle_benchmarks")
    fake_kbench.task = _BenchStub.task
    fake_pd = types.ModuleType("pandas")
    namespace: dict[str, object] = {"Path": Path}
    with patch.dict(sys.modules, {"kaggle_benchmarks": fake_kbench, "pandas": fake_pd}), patch.dict(
        os.environ,
        {
            "COGFLEX_EVAL_SPLIT": "public",
            "COGFLEX_DATASET_ROOT": str(ROOT / "kaggle/dataset/public"),
            "COGFLEX_EXPECTED_PUBLIC_EPISODE_COUNT": "120",
            "COGFLEX_PRIVATE_DATASET_ROOT": "",
            "COGFLEX_PRIVATE_ANSWER_KEY_PATH": "",
        },
        clear=False,
    ):
        exec(code_cells["cell-bootstrap"], namespace)
        exec(code_cells["cell-runtime-types"], namespace)
        exec(code_cells["cell-runtime-normalize"], namespace)
        exec(code_cells["cell-runtime-parse"], namespace)
        exec(code_cells["cell-runtime-validate"], namespace)
        exec(code_cells["cell-runtime-score"], namespace)
        runtime_load_prefix = code_cells["cell-runtime-load"].split("leaderboard_rows = load_selected_rows()", 1)[0]
        exec(runtime_load_prefix, namespace)
    namespace.update({"EVAL_SPLIT": "public", "ROWS_PATH": PUBLIC_ROWS_PATH, "PRIVATE_ANSWER_KEY_PATH": None})
    return namespace


class FakeLLM:
    def __init__(self, final_response: object, *, final_call_index: int) -> None:
        self.final_response = final_response
        self.final_call_index = final_call_index
        self.calls: list[tuple[str, object | None]] = []

    def prompt(self, prompt: str, schema: object | None = None) -> object:
        self.calls.append((prompt, schema))
        return self.final_response if len(self.calls) == self.final_call_index else "ack"


class FailingLLM:
    def __init__(self, *, fail_on_call: int) -> None:
        self.fail_on_call = fail_on_call
        self.calls: list[tuple[str, object | None]] = []

    def prompt(self, prompt: str, schema: object | None = None) -> object:
        self.calls.append((prompt, schema))
        if len(self.calls) == self.fail_on_call:
            raise RuntimeError("prompt failed")
        return "ack"


class FakeRuns:
    def __init__(self, results: list[dict[str, object]]) -> None:
        self._results = results

    def __bool__(self) -> bool:
        return True

    def as_dataframe(self):
        class _ResultFrame:
            def __init__(self, results):
                self.result = results

            def reset_index(self, drop: bool = True):
                return self

            def __len__(self):
                return len(self.result)

        return _ResultFrame(self._results)


class CogflexNotebookRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bootstrap_namespace = load_bootstrap_namespace()
        cls.namespace = load_notebook_namespace()
        cls.rows, _answers, _report = public_fixture()

    @staticmethod
    def _private_probe_metadata_fixture(episode: dict[str, object]) -> list[dict[str, object]]:
        return [
            {
                "probe_index": index,
                "target_label": str(target),
                "obsolete_rule_label": None,
                "congruency": str(annotation),
                "requires_switch": str(annotation) == "incongruent",
            }
            for index, (target, annotation) in enumerate(
                zip(episode["final_probe_targets"], episode["probe_annotations"]),
                start=1,
            )
        ]

    def test_load_rows_accepts_the_public_split(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            loaded_rows = self.namespace["_load_rows"](PUBLIC_ROWS_PATH)
        row = loaded_rows[0]
        self.assertEqual(len(loaded_rows), 120)
        self.assertEqual(sorted(row["inference"]), ["response_spec", "turn_specs", "turns"])
        self.assertEqual(row["inference"]["turn_specs"][-1]["kind"], "decision")
        self.assertEqual(
            row["inference"]["turn_specs"][-1]["item_count"],
            row["inference"]["response_spec"]["probe_count"],
        )

    def test_bootstrap_uses_expected_kaggle_dataset_roots(self) -> None:
        self.assertEqual(
            self.bootstrap_namespace["DEFAULT_DATASET_ROOT"],
            Path("/kaggle/input/datasets/raptorengineer/cogflex-suite-runtime"),
        )
        self.assertEqual(
            self.bootstrap_namespace["DEFAULT_PRIVATE_DATASET_ROOT"],
            Path("/kaggle/input/datasets/raptorengineer/cogflex-suite-runtime-private"),
        )
        self.assertEqual(self.bootstrap_namespace["EXPECTED_PUBLIC_EPISODE_COUNT"], 120)

    def test_notebook_selects_main_task_with_choose_cell(self) -> None:
        code_cells = _load_code_cells()
        self.assertIn("cell-choose", code_cells)
        self.assertIn("%choose cogflex_suite_flexible", code_cells["cell-choose"])

    def test_notebook_runs_main_task_before_choose(self) -> None:
        code_cells = _load_code_cells()
        self.assertIn("cell-run", code_cells)
        self.assertIn("cogflex_suite_flexible.run(kbench.llm)", code_cells["cell-run"])

    def test_notebook_starts_with_a_markdown_overview_cell(self) -> None:
        notebook = _load_notebook()
        self.assertEqual(notebook["cells"][0]["cell_type"], "markdown")

    def test_every_code_cell_has_an_immediately_preceding_markdown_cell(self) -> None:
        notebook = _load_notebook()
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            self.assertGreater(index, 0, msg=f"code cell {cell['id']} should not be first")
            self.assertEqual(
                notebook["cells"][index - 1]["cell_type"],
                "markdown",
                msg=f"code cell {cell['id']} should have a markdown cell immediately before it",
            )

    def test_notebook_preserves_expected_code_cell_ids_in_refactored_flow(self) -> None:
        notebook = _load_notebook()
        code_cell_ids = [cell["id"] for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertEqual(
            code_cell_ids,
            [
                "cell-bootstrap",
                "cell-runtime-types",
                "cell-runtime-normalize",
                "cell-runtime-parse",
                "cell-runtime-validate",
                "cell-runtime-load",
                "cell-runtime-score",
                "cell-task",
                "cell-run",
                "cell-choose",
            ],
        )

    def test_choose_cell_remains_the_last_code_cell(self) -> None:
        notebook = _load_notebook()
        code_cell_ids = [cell["id"] for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertEqual(code_cell_ids[-1], "cell-choose")

    def test_notebook_task_description_uses_cognitive_flexibility_framing(self) -> None:
        code_cells = _load_code_cells()
        self.assertIn(
            'description="Cognitive flexibility benchmark within executive functions with variable turn structures and label vocabularies."',
            code_cells["cell-task"],
        )

    def test_load_rows_accepts_private_inference_only_split(self) -> None:
        self.namespace["EVAL_SPLIT"] = "private"
        with tempfile.TemporaryDirectory() as tmpdir, contextlib.redirect_stdout(io.StringIO()):
            bundle_dir = Path(tmpdir) / "bundle"
            bundle_paths = write_private_bundle(bundle_dir)
            loaded_rows = self.namespace["_load_rows"](bundle_paths["rows"])
        self.assertEqual(len(loaded_rows), 24)
        self.assertNotIn("scoring", loaded_rows[0])
        self.namespace["EVAL_SPLIT"] = "public"

    def test_attach_private_scoring_joins_by_episode_id(self) -> None:
        self.namespace["EVAL_SPLIT"] = "private"
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_dir = Path(tmpdir) / "bundle"
            bundle_paths = write_private_bundle(bundle_dir)
            private_rows = self.namespace["_load_rows"](bundle_paths["rows"])
            self.namespace["PRIVATE_ANSWER_KEY_PATH"] = bundle_paths["answer_key"]
            attached_rows = self.namespace["_attach_private_scoring"](private_rows)
        self.assertIn("scoring", attached_rows[0])
        self.assertEqual(
            len(attached_rows[0]["scoring"]["final_probe_targets"]),
            attached_rows[0]["inference"]["response_spec"]["probe_count"],
        )
        self.assertIn("probe_annotations", attached_rows[0]["scoring"])
        self.assertEqual(
            len(attached_rows[0]["scoring"]["probe_annotations"]),
            attached_rows[0]["inference"]["response_spec"]["probe_count"],
        )
        self.namespace["PRIVATE_ANSWER_KEY_PATH"] = None
        self.namespace["EVAL_SPLIT"] = "public"

    def test_attach_private_scoring_requires_external_answer_key(self) -> None:
        self.namespace["EVAL_SPLIT"] = "private"
        self.namespace["PRIVATE_ANSWER_KEY_PATH"] = None
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_dir = Path(tmpdir) / "bundle"
            bundle_paths = write_private_bundle(bundle_dir)
            private_rows = self.namespace["_load_rows"](bundle_paths["rows"])
        with self.assertRaisesRegex(RuntimeError, "Private split requires an external answer key"):
            self.namespace["attach_selected_scoring"](private_rows)
        self.namespace["EVAL_SPLIT"] = "public"

    def test_attach_private_scoring_rejects_duplicate_episode_ids_in_answer_key(self) -> None:
        self.namespace["EVAL_SPLIT"] = "private"
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                bundle_dir = Path(tmpdir) / "bundle"
                bundle_paths = write_private_bundle(bundle_dir)
                private_rows = self.namespace["_load_rows"](bundle_paths["rows"])
                answer_key = json.loads(bundle_paths["answer_key"].read_text(encoding="utf-8"))
                answer_key["episodes"].append(json.loads(json.dumps(answer_key["episodes"][0])))
                bundle_paths["answer_key"].write_text(json.dumps(answer_key, indent=2) + "\n", encoding="utf-8")
                self.namespace["PRIVATE_ANSWER_KEY_PATH"] = bundle_paths["answer_key"]
                with self.assertRaisesRegex(RuntimeError, "duplicates episode_id"):
                    self.namespace["_attach_private_scoring"](private_rows)
        finally:
            self.namespace["PRIVATE_ANSWER_KEY_PATH"] = None
            self.namespace["EVAL_SPLIT"] = "public"

    def test_attach_private_scoring_rejects_missing_requires_switch(self) -> None:
        self.namespace["EVAL_SPLIT"] = "private"
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                bundle_dir = Path(tmpdir) / "bundle"
                bundle_paths = write_private_bundle(bundle_dir)
                private_rows = self.namespace["_load_rows"](bundle_paths["rows"])
                answer_key = json.loads(bundle_paths["answer_key"].read_text(encoding="utf-8"))
                answer_key["episodes"][0]["probe_metadata"] = self._private_probe_metadata_fixture(answer_key["episodes"][0])
                answer_key["episodes"][0]["probe_metadata"][0].pop("requires_switch")
                bundle_paths["answer_key"].write_text(json.dumps(answer_key, indent=2) + "\n", encoding="utf-8")
                self.namespace["PRIVATE_ANSWER_KEY_PATH"] = bundle_paths["answer_key"]
                with self.assertRaisesRegex(ValueError, "probe_metadata\\.requires_switch is required"):
                    self.namespace["_attach_private_scoring"](private_rows)
        finally:
            self.namespace["PRIVATE_ANSWER_KEY_PATH"] = None
            self.namespace["EVAL_SPLIT"] = "public"

    def test_attach_private_scoring_rejects_missing_obsolete_rule_label(self) -> None:
        self.namespace["EVAL_SPLIT"] = "private"
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                bundle_dir = Path(tmpdir) / "bundle"
                bundle_paths = write_private_bundle(bundle_dir)
                private_rows = self.namespace["_load_rows"](bundle_paths["rows"])
                answer_key = json.loads(bundle_paths["answer_key"].read_text(encoding="utf-8"))
                answer_key["episodes"][0]["probe_metadata"] = self._private_probe_metadata_fixture(answer_key["episodes"][0])
                answer_key["episodes"][0]["probe_metadata"][0].pop("obsolete_rule_label")
                bundle_paths["answer_key"].write_text(json.dumps(answer_key, indent=2) + "\n", encoding="utf-8")
                self.namespace["PRIVATE_ANSWER_KEY_PATH"] = bundle_paths["answer_key"]
                with self.assertRaisesRegex(ValueError, "probe_metadata\\.obsolete_rule_label is required"):
                    self.namespace["_attach_private_scoring"](private_rows)
        finally:
            self.namespace["PRIVATE_ANSWER_KEY_PATH"] = None
            self.namespace["EVAL_SPLIT"] = "public"

    def test_attach_private_scoring_rejects_probe_metadata_length_mismatch(self) -> None:
        self.namespace["EVAL_SPLIT"] = "private"
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                bundle_dir = Path(tmpdir) / "bundle"
                bundle_paths = write_private_bundle(bundle_dir)
                private_rows = self.namespace["_load_rows"](bundle_paths["rows"])
                answer_key = json.loads(bundle_paths["answer_key"].read_text(encoding="utf-8"))
                answer_key["episodes"][0]["probe_metadata"] = self._private_probe_metadata_fixture(answer_key["episodes"][0])
                answer_key["episodes"][0]["probe_metadata"] = answer_key["episodes"][0]["probe_metadata"][:-1]
                bundle_paths["answer_key"].write_text(json.dumps(answer_key, indent=2) + "\n", encoding="utf-8")
                self.namespace["PRIVATE_ANSWER_KEY_PATH"] = bundle_paths["answer_key"]
                with self.assertRaisesRegex(ValueError, "probe_metadata must align with final_probe_targets"):
                    self.namespace["_attach_private_scoring"](private_rows)
        finally:
            self.namespace["PRIVATE_ANSWER_KEY_PATH"] = None
            self.namespace["EVAL_SPLIT"] = "public"

    def test_attach_private_scoring_rejects_answer_key_episode_set_mismatch(self) -> None:
        self.namespace["EVAL_SPLIT"] = "private"
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                bundle_dir = Path(tmpdir) / "bundle"
                bundle_paths = write_private_bundle(bundle_dir)
                private_rows = self.namespace["_load_rows"](bundle_paths["rows"])
                answer_key = json.loads(bundle_paths["answer_key"].read_text(encoding="utf-8"))
                answer_key["episodes"] = answer_key["episodes"][:-1]
                bundle_paths["answer_key"].write_text(json.dumps(answer_key, indent=2) + "\n", encoding="utf-8")
                self.namespace["PRIVATE_ANSWER_KEY_PATH"] = bundle_paths["answer_key"]
                with self.assertRaisesRegex(RuntimeError, "private answer key episode set mismatch"):
                    self.namespace["_attach_private_scoring"](private_rows)
        finally:
            self.namespace["PRIVATE_ANSWER_KEY_PATH"] = None
            self.namespace["EVAL_SPLIT"] = "public"

    def test_validate_row_rejects_missing_decision_turn(self) -> None:
        row = json.loads(json.dumps(self.rows[0]))
        row["inference"]["turn_specs"][-1]["kind"] = "evidence"
        with self.assertRaisesRegex(ValueError, "expected a final decision turn"):
            self.namespace["_validate_row"](row)

    def test_validate_row_rejects_missing_public_scoring(self) -> None:
        row = json.loads(json.dumps(self.rows[0]))
        row.pop("scoring", None)
        with self.assertRaisesRegex(ValueError, "public rows must include scoring"):
            self.namespace["_validate_row"](row)

    def test_validate_row_rejects_missing_probe_annotations(self) -> None:
        row = json.loads(json.dumps(self.rows[0]))
        row["scoring"].pop("probe_annotations", None)
        with self.assertRaisesRegex(ValueError, "scoring must include probe_annotations"):
            self.namespace["_validate_row"](row)

    def test_validate_row_rejects_private_scoring_payloads(self) -> None:
        row = json.loads(json.dumps(self.rows[0]))
        self.namespace["EVAL_SPLIT"] = "private"
        try:
            with self.assertRaisesRegex(ValueError, "private rows must not include scoring"):
                self.namespace["_validate_row"](row)
        finally:
            self.namespace["EVAL_SPLIT"] = "public"

    def test_run_flexible_task_sends_evidence_turns_before_scored_turn(self) -> None:
        row = self.rows[0]
        llm = FakeLLM(
            {"ordered_labels": list(row["scoring"]["final_probe_targets"])},
            final_call_index=len(row["inference"]["turns"]),
        )
        result = self.namespace["run_flexible_task"](
            llm,
            row["inference"]["turns"],
            row["inference"]["response_spec"],
            tuple(row["scoring"]["final_probe_targets"]),
        )
        self.assertEqual(result["denominator"], row["inference"]["response_spec"]["probe_count"])
        self.assertEqual(len(result["predictions"]), row["inference"]["response_spec"]["probe_count"])
        self.assertEqual(len(llm.calls), len(row["inference"]["turns"]))
        for prompt, schema in llm.calls[:-1]:
            self.assertIsNone(schema)
        final_prompt, final_schema = llm.calls[-1]
        self.assertTrue(final_prompt.startswith(row["inference"]["turns"][-1]))
        self.assertIn('Return only a JSON object of the form {"ordered_labels":[...]}.', final_prompt)
        self.assertIn(
            f'"ordered_labels" must contain exactly {row["inference"]["response_spec"]["probe_count"]} labels in probe order.',
            final_prompt,
        )
        self.assertIn(
            f'Use only labels from: {", ".join(row["inference"]["response_spec"]["label_vocab"])}.',
            final_prompt,
        )
        self.assertIn("No markdown, no code fences, no explanations, no extra keys.", final_prompt)
        self.assertIsNone(final_schema)

    def test_normalize_response_spec_rebuilds_output_schema(self) -> None:
        row = self.rows[0]
        normalized = self.namespace["_normalize_response_spec"](row["inference"]["response_spec"])
        self.assertEqual(normalized["output_schema"]["type"], "object")
        self.assertEqual(normalized["output_schema"]["required"], ["ordered_labels"])
        self.assertNotIn("prompt_schema", normalized)
        ordered_labels = normalized["output_schema"]["properties"]["ordered_labels"]
        self.assertEqual(ordered_labels["minItems"], normalized["probe_count"])
        self.assertEqual(ordered_labels["maxItems"], normalized["probe_count"])
        self.assertEqual(ordered_labels["items"]["enum"], row["inference"]["response_spec"]["label_vocab"])

    def test_score_episode_uses_dynamic_denominator(self) -> None:
        result = self.namespace["score_episode"](("orbit", "anchor"), ("orbit", "orbit"))
        self.assertEqual(result["numerator"], 1)
        self.assertEqual(result["denominator"], 2)
        self.assertEqual(result["score_status"], "cognitive_mismatch")

    def test_score_episode_computes_incongruent_and_congruent_counts(self) -> None:
        result = self.namespace["score_episode"](
            ("orbit", "anchor", "orbit"),
            ("orbit", "orbit", "orbit"),
            probe_metadata=(
                {
                    "probe_index": 1,
                    "target_label": "orbit",
                    "obsolete_rule_label": "anchor",
                    "congruency": "incongruent",
                    "requires_switch": True,
                },
                {
                    "probe_index": 2,
                    "target_label": "orbit",
                    "obsolete_rule_label": "orbit",
                    "congruency": "congruent",
                    "requires_switch": True,
                },
                {
                    "probe_index": 3,
                    "target_label": "orbit",
                    "obsolete_rule_label": "anchor",
                    "congruency": "incongruent",
                    "requires_switch": True,
                },
            ),
        )
        self.assertEqual(result["numerator"], 2)
        self.assertEqual(result["incongruent_numerator"], 2)
        self.assertEqual(result["incongruent_denominator"], 2)
        self.assertEqual(result["congruent_numerator"], 0)
        self.assertEqual(result["congruent_denominator"], 1)
        self.assertEqual(result["first_probe_numerator"], 1)
        self.assertEqual(result["first_probe_denominator"], 1)
        self.assertEqual(result["obsolete_rule_error_numerator"], 0)
        self.assertEqual(result["obsolete_rule_error_denominator"], 3)

    def test_score_episode_detects_obsolete_rule_errors(self) -> None:
        result = self.namespace["score_episode"](
            ("left", "right"),
            ("right", "right"),
            probe_metadata=(
                {
                    "probe_index": 1,
                    "target_label": "right",
                    "obsolete_rule_label": "left",
                    "congruency": "incongruent",
                    "requires_switch": True,
                },
                {
                    "probe_index": 2,
                    "target_label": "right",
                    "obsolete_rule_label": "left",
                    "congruency": "incongruent",
                    "requires_switch": True,
                },
            ),
        )
        self.assertEqual(result["obsolete_rule_error_numerator"], 1)
        self.assertEqual(result["obsolete_rule_error_denominator"], 2)

    def test_score_episode_tracks_first_probe_accuracy_after_shift(self) -> None:
        result = self.namespace["score_episode"](
            ("stay", "switch"),
            ("switch", "switch"),
            probe_metadata=(
                {
                    "probe_index": 1,
                    "target_label": "switch",
                    "obsolete_rule_label": "stay",
                    "congruency": "incongruent",
                    "requires_switch": True,
                },
                {
                    "probe_index": 2,
                    "target_label": "switch",
                    "obsolete_rule_label": "stay",
                    "congruency": "incongruent",
                    "requires_switch": True,
                },
            ),
        )
        self.assertEqual(result["first_probe_numerator"], 0)
        self.assertEqual(result["first_probe_denominator"], 1)

    def test_score_episode_uses_late_switch_block_start_for_first_probe_accuracy(self) -> None:
        result = self.namespace["score_episode"](
            ("left", "left", "left", "left", "right"),
            ("left", "left", "left", "right", "right"),
            probe_metadata=(
                {
                    "probe_index": 1,
                    "target_label": "left",
                    "obsolete_rule_label": None,
                    "congruency": "congruent",
                    "requires_switch": False,
                },
                {
                    "probe_index": 2,
                    "target_label": "left",
                    "obsolete_rule_label": None,
                    "congruency": "congruent",
                    "requires_switch": False,
                },
                {
                    "probe_index": 3,
                    "target_label": "left",
                    "obsolete_rule_label": None,
                    "congruency": "congruent",
                    "requires_switch": False,
                },
                {
                    "probe_index": 4,
                    "target_label": "right",
                    "obsolete_rule_label": "left",
                    "congruency": "incongruent",
                    "requires_switch": True,
                },
                {
                    "probe_index": 5,
                    "target_label": "right",
                    "obsolete_rule_label": "left",
                    "congruency": "incongruent",
                    "requires_switch": True,
                },
            ),
        )
        self.assertEqual(result["first_probe_numerator"], 0)
        self.assertEqual(result["first_probe_denominator"], 1)

    def test_score_episode_counts_each_switch_block_start_for_first_probe_accuracy(self) -> None:
        result = self.namespace["score_episode"](
            ("left", "right", "right", "right", "right", "left"),
            ("left", "right", "right", "right", "left", "left"),
            probe_metadata=(
                {
                    "probe_index": 1,
                    "target_label": "left",
                    "obsolete_rule_label": None,
                    "congruency": "congruent",
                    "requires_switch": False,
                },
                {
                    "probe_index": 2,
                    "target_label": "right",
                    "obsolete_rule_label": "left",
                    "congruency": "incongruent",
                    "requires_switch": True,
                },
                {
                    "probe_index": 3,
                    "target_label": "right",
                    "obsolete_rule_label": "left",
                    "congruency": "incongruent",
                    "requires_switch": True,
                },
                {
                    "probe_index": 4,
                    "target_label": "right",
                    "obsolete_rule_label": None,
                    "congruency": "congruent",
                    "requires_switch": False,
                },
                {
                    "probe_index": 5,
                    "target_label": "left",
                    "obsolete_rule_label": "right",
                    "congruency": "incongruent",
                    "requires_switch": True,
                },
                {
                    "probe_index": 6,
                    "target_label": "left",
                    "obsolete_rule_label": None,
                    "congruency": "congruent",
                    "requires_switch": False,
                },
            ),
        )
        self.assertEqual(result["first_probe_numerator"], 1)
        self.assertEqual(result["first_probe_denominator"], 2)

    def test_score_episode_skips_first_probe_accuracy_when_episode_never_switches(self) -> None:
        result = self.namespace["score_episode"](
            ("left", "right", "left"),
            ("left", "right", "left"),
            probe_metadata=(
                {
                    "probe_index": 1,
                    "target_label": "left",
                    "obsolete_rule_label": None,
                    "congruency": "congruent",
                    "requires_switch": False,
                },
                {
                    "probe_index": 2,
                    "target_label": "right",
                    "obsolete_rule_label": None,
                    "congruency": "congruent",
                    "requires_switch": False,
                },
                {
                    "probe_index": 3,
                    "target_label": "left",
                    "obsolete_rule_label": None,
                    "congruency": "congruent",
                    "requires_switch": False,
                },
            ),
        )
        self.assertEqual(result["first_probe_numerator"], 0)
        self.assertEqual(result["first_probe_denominator"], 0)

    def test_run_flexible_task_scores_invalid_labels_as_zero_instead_of_raising(self) -> None:
        row = self.rows[0]
        llm = FakeLLM({"ordered_labels": ["not_in_vocab"] * len(row["scoring"]["final_probe_targets"])}, final_call_index=len(row["inference"]["turns"]))
        result = self.namespace["run_flexible_task"](
            llm,
            row["inference"]["turns"],
            row["inference"]["response_spec"],
            tuple(row["scoring"]["final_probe_targets"]),
        )
        self.assertEqual(result["numerator"], 0)
        self.assertEqual(result["denominator"], len(row["scoring"]["final_probe_targets"]))
        self.assertEqual(result["predictions"], [""] * len(row["scoring"]["final_probe_targets"]))
        self.assertEqual(result["score_status"], "invalid_label_vocab")

    def test_run_flexible_task_scores_prompt_failures_as_zero_instead_of_raising(self) -> None:
        row = self.rows[0]
        llm = FailingLLM(fail_on_call=len(row["inference"]["turns"]))
        with contextlib.redirect_stderr(io.StringIO()) as stderr:
            result = self.namespace["run_flexible_task"](
                llm,
                row["inference"]["turns"],
                row["inference"]["response_spec"],
                tuple(row["scoring"]["final_probe_targets"]),
            )
        self.assertEqual(result["numerator"], 0)
        self.assertEqual(result["denominator"], len(row["scoring"]["final_probe_targets"]))
        self.assertEqual(result["predictions"], [""] * len(row["scoring"]["final_probe_targets"]))
        self.assertEqual(result["score_status"], "prompt_failure")
        self.assertIn("Final prompt failed: RuntimeError('prompt failed')", stderr.getvalue())

    def test_normalize_ordered_labels_accepts_structured_and_legacy_shapes(self) -> None:
        response_spec = {"format": "ordered_labels", "probe_count": 3, "label_vocab": ["left", "right"]}
        normalized_structured = self.namespace["normalize_ordered_labels"](
            {"ordered_labels": ["left", "right", "left"]},
            response_spec,
        )
        normalized_text = self.namespace["normalize_ordered_labels"]("left, right, left", response_spec)
        normalized_dict = self.namespace["normalize_ordered_labels"](
            {"probe_1": "left", "probe_2": "right", "probe_3": "left"},
            response_spec,
        )
        self.assertEqual(normalized_structured, ("left", "right", "left"))
        self.assertEqual(normalized_text, ("left", "right", "left"))
        self.assertEqual(normalized_dict, ("left", "right", "left"))

    def test_extract_handles_json_string_response(self) -> None:
        response_spec = {"format": "ordered_labels", "probe_count": 3, "label_vocab": ["left", "right"]}
        result = self.namespace["normalize_ordered_labels"](
            '{"ordered_labels": ["left", "right", "left"]}',
            response_spec,
        )
        self.assertEqual(result, ("left", "right", "left"))

    def test_extract_handles_json_array_string_response(self) -> None:
        response_spec = {"format": "ordered_labels", "probe_count": 3, "label_vocab": ["left", "right"]}
        result = self.namespace["normalize_ordered_labels"](
            '["left", "right", "left"]',
            response_spec,
        )
        self.assertEqual(result, ("left", "right", "left"))

    def test_extract_handles_markdown_fenced_json_response(self) -> None:
        response_spec = {"format": "ordered_labels", "probe_count": 3, "label_vocab": ["left", "right"]}
        fenced = '```json\n{"ordered_labels": ["left", "right", "left"]}\n```'
        result = self.namespace["normalize_ordered_labels"](fenced, response_spec)
        self.assertEqual(result, ("left", "right", "left"))

    def test_extract_strips_numbered_list_prefixes(self) -> None:
        response_spec = {"format": "ordered_labels", "probe_count": 3, "label_vocab": ["left", "right"]}
        result = self.namespace["normalize_ordered_labels"]("1. left\n2. right\n3. left", response_spec)
        self.assertEqual(result, ("left", "right", "left"))

    def test_extract_strips_surrounding_quotes_from_text(self) -> None:
        response_spec = {"format": "ordered_labels", "probe_count": 3, "label_vocab": ["left", "right"]}
        result = self.namespace["normalize_ordered_labels"]('"left", "right", "left"', response_spec)
        self.assertEqual(result, ("left", "right", "left"))

    def test_extract_handles_pydantic_style_object(self) -> None:
        response_spec = {"format": "ordered_labels", "probe_count": 2, "label_vocab": ["left", "right"]}

        class PydanticLike:
            def __init__(self):
                self.ordered_labels = ["left", "right"]

        result = self.namespace["normalize_ordered_labels"](PydanticLike(), response_spec)
        self.assertEqual(result, ("left", "right"))

    def test_extract_handles_text_attribute_wrapper(self) -> None:
        response_spec = {"format": "ordered_labels", "probe_count": 3, "label_vocab": ["left", "right"]}

        class TextWrapper:
            def __init__(self):
                self.text = '{"ordered_labels": ["left", "right", "left"]}'

        result = self.namespace["normalize_ordered_labels"](TextWrapper(), response_spec)
        self.assertEqual(result, ("left", "right", "left"))

    def test_extract_handles_text_key_wrapper_dict(self) -> None:
        response_spec = {"format": "ordered_labels", "probe_count": 3, "label_vocab": ["left", "right"]}
        result = self.namespace["normalize_ordered_labels"](
            {"text": '{"ordered_labels": ["left", "right", "left"]}'},
            response_spec,
        )
        self.assertEqual(result, ("left", "right", "left"))

    def test_extract_handles_bare_code_fences(self) -> None:
        response_spec = {"format": "ordered_labels", "probe_count": 2, "label_vocab": ["left", "right"]}
        fenced = '```\n["left", "right"]\n```'
        result = self.namespace["normalize_ordered_labels"](fenced, response_spec)
        self.assertEqual(result, ("left", "right"))

    def test_run_flexible_task_marks_wrong_label_count_as_format_failure(self) -> None:
        row = self.rows[0]
        llm = FakeLLM({"ordered_labels": ["only_one"]}, final_call_index=len(row["inference"]["turns"]))
        result = self.namespace["run_flexible_task"](
            llm,
            row["inference"]["turns"],
            row["inference"]["response_spec"],
            tuple(row["scoring"]["final_probe_targets"]),
        )
        self.assertEqual(result["numerator"], 0)
        self.assertEqual(result["score_status"], "wrong_label_count")
        self.assertEqual(result["predictions"], [""] * len(row["scoring"]["final_probe_targets"]))

    def test_run_flexible_task_marks_unparseable_responses_as_schema_failures(self) -> None:
        row = self.rows[0]
        llm = FakeLLM({"unexpected": "shape"}, final_call_index=len(row["inference"]["turns"]))
        result = self.namespace["run_flexible_task"](
            llm,
            row["inference"]["turns"],
            row["inference"]["response_spec"],
            tuple(row["scoring"]["final_probe_targets"]),
        )
        self.assertEqual(result["numerator"], 0)
        self.assertEqual(result["score_status"], "schema_format_failure")
        self.assertEqual(result["predictions"], [""] * len(row["scoring"]["final_probe_targets"]))

    def test_run_flexible_task_keeps_cognitive_mismatch_distinct_from_format_failure(self) -> None:
        row = self.rows[0]
        targets = tuple(row["scoring"]["final_probe_targets"])
        alternate_label = next(
            label
            for label in row["inference"]["response_spec"]["label_vocab"]
            if label != targets[0]
        )
        llm = FakeLLM(
            {"ordered_labels": [alternate_label, *targets[1:]]},
            final_call_index=len(row["inference"]["turns"]),
        )
        result = self.namespace["run_flexible_task"](
            llm,
            row["inference"]["turns"],
            row["inference"]["response_spec"],
            targets,
        )
        self.assertEqual(result["score_status"], "cognitive_mismatch")
        self.assertEqual(result["numerator"], len(targets) - 1)

    def _make_suite_summary_fixture(self, namespace):
        rows = [
            {"analysis": {"suite_task_id": "explicit_rule_update", "structure_family_id": "two_step_focus", "difficulty_bin": "hard"}},
            {"analysis": {"suite_task_id": "latent_rule_update", "structure_family_id": "three_step_bridge", "difficulty_bin": "hard"}},
            {"analysis": {"suite_task_id": "context_binding", "structure_family_id": "two_step_focus", "difficulty_bin": "medium"}},
            {"analysis": {"suite_task_id": "trial_cued_switch", "structure_family_id": "three_step_bridge", "difficulty_bin": "medium"}},
        ]
        runs = FakeRuns(
            [
                {"numerator": 5, "denominator": 5, "predictions": ["accept"] * 5, "incongruent_numerator": 3, "incongruent_denominator": 3, "congruent_numerator": 2, "congruent_denominator": 2, "first_probe_numerator": 1, "first_probe_denominator": 1, "obsolete_rule_error_numerator": 0, "obsolete_rule_error_denominator": 5},
                {"numerator": 3, "denominator": 6, "predictions": ["north"] * 6, "incongruent_numerator": 1, "incongruent_denominator": 4, "congruent_numerator": 2, "congruent_denominator": 2, "first_probe_numerator": 0, "first_probe_denominator": 1, "obsolete_rule_error_numerator": 2, "obsolete_rule_error_denominator": 6},
                {"numerator": 6, "denominator": 6, "predictions": ["amber"] * 6, "incongruent_numerator": 4, "incongruent_denominator": 4, "congruent_numerator": 2, "congruent_denominator": 2, "first_probe_numerator": 1, "first_probe_denominator": 1, "obsolete_rule_error_numerator": 0, "obsolete_rule_error_denominator": 6},
                {"numerator": 0, "denominator": 5, "predictions": ["accept"] * 5, "incongruent_numerator": 0, "incongruent_denominator": 3, "congruent_numerator": 0, "congruent_denominator": 2, "first_probe_numerator": 0, "first_probe_denominator": 1, "obsolete_rule_error_numerator": 4, "obsolete_rule_error_denominator": 5},
            ]
        )
        return runs, rows

    def test_default_summary_contains_exactly_compact_keys(self) -> None:
        code_cells = _load_code_cells()
        namespace = dict(self.namespace)
        exec(code_cells["cell-task"], namespace)
        runs, rows = self._make_suite_summary_fixture(namespace)
        summary = namespace["summarize_suite_benchmark"](runs, rows)
        expected_keys = {"score", "protocol_valid_rate", "scorable_episodes", "episodes", "macro_accuracy", "incongruent_accuracy", "first_probe_accuracy", "obsolete_rule_error_rate"}
        self.assertEqual(set(summary.keys()), expected_keys)

    def test_default_summary_excludes_debug_keys(self) -> None:
        code_cells = _load_code_cells()
        namespace = dict(self.namespace)
        exec(code_cells["cell-task"], namespace)
        runs, rows = self._make_suite_summary_fixture(namespace)
        summary = namespace["summarize_suite_benchmark"](runs, rows)
        forbidden_keys = {"score_formula", "micro_accuracy", "congruent_accuracy", "requires_switch_accuracy", "switch_cost", "numerator", "denominator", "incongruent_numerator", "incongruent_denominator", "congruent_numerator", "congruent_denominator", "first_probe_numerator", "first_probe_denominator", "obsolete_rule_error_numerator", "obsolete_rule_error_denominator", "requires_switch_numerator", "requires_switch_denominator", "per_task_accuracy", "per_task_metrics", "structure_family_accuracy", "difficulty_bin_accuracy"}
        self.assertEqual(set(summary.keys()) & forbidden_keys, set())

    def test_debug_summary_exposes_detailed_fields(self) -> None:
        code_cells = _load_code_cells()
        namespace = dict(self.namespace)
        exec(code_cells["cell-task"], namespace)
        runs, rows = self._make_suite_summary_fixture(namespace)
        summary = namespace["summarize_suite_benchmark"](runs, rows, include_debug=True)
        self.assertAlmostEqual(summary["micro_accuracy"], 14 / 22)
        self.assertEqual(set(summary["structure_family_accuracy"]), {"three_step_bridge", "two_step_focus"})
        self.assertAlmostEqual(summary["congruent_accuracy"], 6 / 8)
        self.assertAlmostEqual(summary["switch_cost"], 6 / 8 - 8 / 14)
        self.assertIn("per_task_metrics", summary)
        self.assertAlmostEqual(summary["per_task_metrics"]["explicit_rule_update"]["first_probe_accuracy"], 1.0)
        self.assertIn("score_formula", summary)

    def test_suite_summary_uses_macro_average_and_structure_breakdown(self) -> None:
        code_cells = _load_code_cells()
        namespace = dict(self.namespace)
        exec(code_cells["cell-task"], namespace)
        runs, rows = self._make_suite_summary_fixture(namespace)
        summary = namespace["summarize_suite_benchmark"](runs, rows)
        self.assertAlmostEqual(summary["macro_accuracy"], (1.0 + 0.5 + 1.0 + 0.0) / 4)
        self.assertAlmostEqual(summary["incongruent_accuracy"], 8 / 14)
        self.assertAlmostEqual(summary["first_probe_accuracy"], 2 / 4)
        self.assertAlmostEqual(summary["obsolete_rule_error_rate"], 6 / 22)
        self.assertAlmostEqual(
            summary["score"],
            (
                summary["macro_accuracy"]
                + summary["incongruent_accuracy"]
                + summary["first_probe_accuracy"]
                + (1.0 - summary["obsolete_rule_error_rate"])
            )
            / 4.0,
        )
        self.assertEqual(summary["scorable_episodes"], 4)
        self.assertAlmostEqual(summary["protocol_valid_rate"], 1.0)

    def test_score_episode_sets_scorable_true_for_correct(self) -> None:
        result = self.namespace["score_episode"](("orbit",), ("orbit",))
        self.assertTrue(result["scorable"])

    def test_score_episode_sets_scorable_true_for_cognitive_mismatch(self) -> None:
        result = self.namespace["score_episode"](("orbit",), ("anchor",))
        self.assertTrue(result["scorable"])

    def test_score_episode_sets_scorable_false_for_protocol_failures(self) -> None:
        for status in ("prompt_failure", "schema_format_failure", "wrong_label_count", "invalid_label_vocab"):
            result = self.namespace["score_episode"](None, ("orbit",), score_status=status)
            self.assertFalse(result["scorable"], msg=f"expected scorable=False for {status}")

    def test_score_episode_rejects_non_boolean_requires_switch(self) -> None:
        with self.assertRaisesRegex(ValueError, "probe_metadata\\.requires_switch must be a boolean"):
            self.namespace["score_episode"](
                ("orbit",),
                ("orbit",),
                probe_metadata=(
                    {
                        "probe_index": 1,
                        "target_label": "orbit",
                        "obsolete_rule_label": None,
                        "congruency": "congruent",
                        "requires_switch": "false",
                    },
                ),
            )

    def test_all_protocol_failures_produce_zero_score_not_025(self) -> None:
        """Regression: fully invalid runs must not receive 0.25 via the obsolete-rule term."""
        code_cells = _load_code_cells()
        namespace = dict(self.namespace)
        exec(code_cells["cell-task"], namespace)
        rows = [
            {"analysis": {"suite_task_id": "explicit_rule_update", "structure_family_id": "two_step_focus", "difficulty_bin": "hard"}},
            {"analysis": {"suite_task_id": "latent_rule_update", "structure_family_id": "three_step_bridge", "difficulty_bin": "hard"}},
        ]
        runs = FakeRuns(
            [
                {"numerator": 0, "denominator": 5, "predictions": [""] * 5, "scorable": False, "incongruent_numerator": 0, "incongruent_denominator": 3, "congruent_numerator": 0, "congruent_denominator": 2, "first_probe_numerator": 0, "first_probe_denominator": 1, "obsolete_rule_error_numerator": 0, "obsolete_rule_error_denominator": 5},
                {"numerator": 0, "denominator": 6, "predictions": [""] * 6, "scorable": False, "incongruent_numerator": 0, "incongruent_denominator": 4, "congruent_numerator": 0, "congruent_denominator": 2, "first_probe_numerator": 0, "first_probe_denominator": 1, "obsolete_rule_error_numerator": 0, "obsolete_rule_error_denominator": 6},
            ]
        )
        summary = namespace["summarize_suite_benchmark"](runs, rows)
        self.assertAlmostEqual(summary["score"], 0.0)
        self.assertEqual(summary["scorable_episodes"], 0)
        self.assertAlmostEqual(summary["protocol_valid_rate"], 0.0)

    def test_mixed_valid_invalid_episodes_get_partial_credit(self) -> None:
        """Mixed batches: only valid episodes contribute to the obsolete-rule term."""
        code_cells = _load_code_cells()
        namespace = dict(self.namespace)
        exec(code_cells["cell-task"], namespace)
        rows = [
            {"analysis": {"suite_task_id": "explicit_rule_update", "structure_family_id": "two_step_focus", "difficulty_bin": "hard"}},
            {"analysis": {"suite_task_id": "latent_rule_update", "structure_family_id": "three_step_bridge", "difficulty_bin": "hard"}},
        ]
        runs = FakeRuns(
            [
                {"numerator": 5, "denominator": 5, "predictions": ["accept"] * 5, "scorable": True, "incongruent_numerator": 3, "incongruent_denominator": 3, "congruent_numerator": 2, "congruent_denominator": 2, "first_probe_numerator": 1, "first_probe_denominator": 1, "obsolete_rule_error_numerator": 0, "obsolete_rule_error_denominator": 5},
                {"numerator": 0, "denominator": 6, "predictions": [""] * 6, "scorable": False, "incongruent_numerator": 0, "incongruent_denominator": 4, "congruent_numerator": 0, "congruent_denominator": 2, "first_probe_numerator": 0, "first_probe_denominator": 1, "obsolete_rule_error_numerator": 0, "obsolete_rule_error_denominator": 6},
            ]
        )
        summary = namespace["summarize_suite_benchmark"](runs, rows)
        self.assertEqual(summary["scorable_episodes"], 1)
        self.assertAlmostEqual(summary["protocol_valid_rate"], 0.5)
        expected_score = (
            summary["macro_accuracy"]
            + summary["incongruent_accuracy"]
            + summary["first_probe_accuracy"]
            + 0.5 * (1.0 - summary["obsolete_rule_error_rate"])
        ) / 4.0
        self.assertAlmostEqual(summary["score"], expected_score)
