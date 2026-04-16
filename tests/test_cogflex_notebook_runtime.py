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
    return namespace


class CogflexNotebookRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.namespace = load_notebook_namespace()

    def test_run_flexible_task_ignores_schema_metadata_in_response_spec(self) -> None:
        row = self.namespace["_load_rows"](PUBLIC_ROWS_PATH)[0]
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
