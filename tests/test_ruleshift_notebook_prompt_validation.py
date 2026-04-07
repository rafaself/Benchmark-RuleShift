import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_ruleshift_dataset import build_split, private_answer_key_payload, sanitize_private_rows


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "kaggle/notebook/ruleshift_notebook_task.ipynb"
PUBLIC_ROWS_PATH = ROOT / "kaggle/dataset/public/public_leaderboard_rows.json"


class _BenchStub:
    @staticmethod
    def task(*args, **kwargs):
        def decorator(fn):
            return fn

        return decorator


def load_notebook_namespace() -> dict[str, object]:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code_cells = {
        cell["id"]: "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    }
    namespace: dict[str, object] = {"Path": Path, "kbench": _BenchStub(), "pd": None}
    exec(code_cells["cell-runtime-types"], namespace)
    exec(code_cells["cell-runtime-normalize"], namespace)
    exec(code_cells["cell-runtime-score"], namespace)
    runtime_load_prefix = code_cells["cell-runtime-load"].split(
        "leaderboard_rows = load_selected_rows()",
        1,
    )[0]
    exec(runtime_load_prefix, namespace)
    namespace.update(
        {
            "EVAL_SPLIT": "public",
            "ROWS_PATH": PUBLIC_ROWS_PATH,
            "EXPECTED_PUBLIC_EPISODE_COUNT": 80,
            "EXPECTED_PRIVATE_EPISODE_COUNT": 400,
            "EXPECTED_EPISODES_PER_GROUP": {"public": 20, "private": 100},
            "PRIVATE_ANSWER_KEY_PATH_ENV_VAR": "RULESHIFT_PRIVATE_ANSWER_KEY_PATH",
            "PRIVATE_ANSWER_KEY_PATH": None,
        }
    )
    return namespace


class FakeLLM:
    def __init__(self, final_response: object) -> None:
        self.final_response = final_response
        self.calls: list[tuple[str, object | None]] = []

    def prompt(self, prompt: str, schema: object | None = None) -> object:
        self.calls.append((prompt, schema))
        if schema is None:
            return "ack"
        return self.final_response


class RuleshiftNotebookRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.namespace = load_notebook_namespace()
        cls.rows = json.loads(PUBLIC_ROWS_PATH.read_text(encoding="utf-8"))
        private_rows, private_answers = build_split(
            "private",
            variants_per_rule=5,
            variant_start=1,
            private_seed="notebook-test-private-seed",
        )
        cls.private_rows = sanitize_private_rows(private_rows)
        cls.private_answer_key = private_answer_key_payload(private_answers)

    def test_load_rows_accepts_the_public_split(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            loaded_rows = self.namespace["_load_rows"](PUBLIC_ROWS_PATH)
        self.assertEqual(len(loaded_rows), 80)
        self.assertEqual(len(loaded_rows[0]["inference"]["turns"]), 3)

    def test_load_rows_accepts_private_inference_only_split(self) -> None:
        self.namespace["EVAL_SPLIT"] = "private"
        with tempfile.TemporaryDirectory() as tmpdir, contextlib.redirect_stdout(io.StringIO()):
            private_rows_path = Path(tmpdir) / "private_rows.json"
            private_rows_path.write_text(json.dumps(self.private_rows), encoding="utf-8")
            loaded_rows = self.namespace["_load_rows"](private_rows_path)
        self.assertEqual(len(loaded_rows), 400)
        self.assertNotIn("scoring", loaded_rows[0])
        self.namespace["EVAL_SPLIT"] = "public"

    def test_attach_private_scoring_joins_by_episode_id(self) -> None:
        self.namespace["EVAL_SPLIT"] = "private"
        with tempfile.TemporaryDirectory() as tmpdir:
            answer_key_path = Path(tmpdir) / "private_answer_key.json"
            answer_key_path.write_text(json.dumps(self.private_answer_key), encoding="utf-8")
            self.namespace["PRIVATE_ANSWER_KEY_PATH"] = answer_key_path
            attached_rows = self.namespace["_attach_private_scoring"](self.private_rows)
        self.assertIn("scoring", attached_rows[0])
        self.assertEqual(len(attached_rows[0]["scoring"]["final_probe_targets"]), 4)
        self.namespace["PRIVATE_ANSWER_KEY_PATH"] = None
        self.namespace["EVAL_SPLIT"] = "public"

    def test_attach_private_scoring_requires_external_answer_key(self) -> None:
        self.namespace["EVAL_SPLIT"] = "private"
        self.namespace["PRIVATE_ANSWER_KEY_PATH"] = None
        with self.assertRaisesRegex(RuntimeError, "Private split requires an external answer key"):
            self.namespace["attach_selected_scoring"](self.private_rows)
        self.namespace["EVAL_SPLIT"] = "public"

    def test_validate_row_rejects_missing_turn(self) -> None:
        row = json.loads(json.dumps(self.rows[0]))
        row["inference"]["turns"] = row["inference"]["turns"][:2]
        with self.assertRaisesRegex(ValueError, "expected exactly 3 turns"):
            self.namespace["_validate_row"](row)

    def test_run_binary_task_sends_first_two_turns_before_scored_turn(self) -> None:
        row = self.rows[0]
        llm = FakeLLM({"probe_1": "type_b", "probe_2": "type_b", "probe_3": "type_a", "probe_4": "type_a"})
        result = self.namespace["run_binary_task"](llm, row["inference"]["turns"], tuple(row["scoring"]["final_probe_targets"]))
        self.assertEqual(result["denominator"], 4)
        self.assertEqual(len(llm.calls), 3)
        self.assertIsNone(llm.calls[0][1])
        self.assertIsNone(llm.calls[1][1])
        self.assertIs(self.namespace["BinaryResponse"], llm.calls[2][1])

    def test_normalize_binary_response_accepts_plain_text(self) -> None:
        normalized = self.namespace["normalize_binary_response"]("type_a, type_b\ntype_a, type_b")
        self.assertEqual(normalized, ("type_a", "type_b", "type_a", "type_b"))
