from __future__ import annotations

from dataclasses import fields

from pydantic import TypeAdapter
import pytest

from core.kaggle import KaggleExecutionError, load_leaderboard_dataframe, run_binary_task
from core.kaggle.runner import BinaryResponse, Label, normalize_binary_response


class _RaisingLLM:
    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def prompt(self, *_args, **_kwargs):
        raise self._exc


class _StringLLM:
    def prompt(self, *_args, **_kwargs):
        return "attract, repel, attract, repel"


class _MappingLLM:
    def prompt(self, *_args, **_kwargs):
        return {
            "probe_6": "attract",
            "probe_7": "repel",
            "probe_8": "attract",
            "probe_9": "repel",
        }


class _SchemaAwareLLM:
    def __init__(self) -> None:
        self.last_schema = None

    def prompt(self, *_args, **kwargs):
        self.last_schema = kwargs.get("schema")
        return BinaryResponse(
            probe_6=Label.attract,
            probe_7=Label.repel,
            probe_8=Label.attract,
            probe_9=Label.repel,
        )


class _StringFieldBinaryResponseLLM:
    def prompt(self, *_args, **_kwargs):
        return BinaryResponse(
            probe_6="attract",
            probe_7="repel",
            probe_8="attract",
            probe_9="repel",
        )


class _InvalidBinaryResponseLLM:
    def prompt(self, *_args, **_kwargs):
        return BinaryResponse(
            probe_6="invalid_label",
            probe_7="repel",
            probe_8="attract",
            probe_9="repel",
        )


@pytest.mark.parametrize(
    ("response", "expected_type"),
    [
        (None, "NoneType"),
        ("not,a,valid,binary,response", "str"),
        ({"unexpected": 1}, "dict"),
    ],
)
def test_run_binary_task_raises_for_unscoreable_outputs(response, expected_type):
    class _BadResponseLLM:
        def prompt(self, *_args, **_kwargs):
            return response

    targets = ("attract", "repel", "attract", "repel")

    with pytest.raises(
        KaggleExecutionError,
        match=rf"unscoreable response of type {expected_type}",
    ):
        run_binary_task(
            llm=_BadResponseLLM(),
            prompt_binary="prompt",
            probe_targets=targets,
        )


def test_run_binary_task_scores_minimal_valid_binary_response():
    targets = ("attract", "repel", "attract", "repel")
    llm = _SchemaAwareLLM()

    assert run_binary_task(
        llm=llm,
        prompt_binary="prompt",
        probe_targets=targets,
    ) == (4, 4)
    assert llm.last_schema is BinaryResponse


def test_binary_response_as_tuple_accepts_enum_fields():
    response = BinaryResponse(
        probe_6=Label.attract,
        probe_7=Label.repel,
        probe_8=Label.attract,
        probe_9=Label.repel,
    )

    assert response.as_tuple() == ("attract", "repel", "attract", "repel")


def test_binary_response_as_tuple_accepts_string_fields():
    response = BinaryResponse(
        probe_6="attract",
        probe_7="repel",
        probe_8="attract",
        probe_9="repel",
    )

    assert response.as_tuple() == ("attract", "repel", "attract", "repel")
    assert normalize_binary_response(response) == ("attract", "repel", "attract", "repel")


def test_binary_response_as_tuple_rejects_invalid_string_fields():
    response = BinaryResponse(
        probe_6="invalid_label",
        probe_7="repel",
        probe_8="attract",
        probe_9="repel",
    )

    with pytest.raises(ValueError, match=r"probe_6"):
        response.as_tuple()


def test_run_binary_task_scores_valid_string_and_mapping_responses():
    targets = ("attract", "repel", "attract", "repel")

    assert run_binary_task(
        llm=_StringLLM(),
        prompt_binary="prompt",
        probe_targets=targets,
    ) == (4, 4)
    assert run_binary_task(
        llm=_MappingLLM(),
        prompt_binary="prompt",
        probe_targets=targets,
    ) == (4, 4)


def test_run_binary_task_scores_binary_response_with_string_fields():
    targets = ("attract", "repel", "attract", "repel")

    assert run_binary_task(
        llm=_StringFieldBinaryResponseLLM(),
        prompt_binary="prompt",
        probe_targets=targets,
    ) == (4, 4)


def test_run_binary_task_raises_for_invalid_binary_response_fields():
    targets = ("attract", "repel", "attract", "repel")

    with pytest.raises(
        KaggleExecutionError,
        match=r"invalid binary response.*probe_6.*invalid_label",
    ):
        run_binary_task(
            llm=_InvalidBinaryResponseLLM(),
            prompt_binary="prompt",
            probe_targets=targets,
        )


def test_binary_response_matches_structured_output_schema_contract():
    schema = TypeAdapter(BinaryResponse).json_schema()

    assert getattr(BinaryResponse, "__slots__", None) is None
    assert [field.type for field in fields(BinaryResponse)] == [Label, Label, Label, Label]
    assert schema["type"] == "object"
    assert schema["title"] == "BinaryResponse"
    assert schema["required"] == ["probe_6", "probe_7", "probe_8", "probe_9"]
    assert list(schema["properties"]) == ["probe_6", "probe_7", "probe_8", "probe_9"]
    assert schema["$defs"]["Label"]["enum"] == ["attract", "repel"]


def test_run_binary_task_surfaces_provider_exception():
    targets = ("attract", "repel", "attract", "repel")

    with pytest.raises(KaggleExecutionError, match="llm.prompt failed"):
        run_binary_task(
            llm=_RaisingLLM(RuntimeError("provider unavailable")),
            prompt_binary="prompt",
            probe_targets=targets,
        )


def test_run_binary_task_surfaces_timeout_like_failure():
    targets = ("attract", "repel", "attract", "repel")

    with pytest.raises(KaggleExecutionError, match="llm.prompt failed"):
        run_binary_task(
            llm=_RaisingLLM(TimeoutError("provider timeout")),
            prompt_binary="prompt",
            probe_targets=targets,
        )


def test_load_leaderboard_dataframe_preserves_public_private_behavior(monkeypatch):
    private_root, frozen_splits, leaderboard_df = load_leaderboard_dataframe()

    assert private_root is not None
    assert set(frozen_splits) == {"public_leaderboard", "private_leaderboard"}
    assert set(leaderboard_df["split"]) == {"public_leaderboard", "private_leaderboard"}

    monkeypatch.delenv("RULESHIFT_PRIVATE_DATASET_ROOT", raising=False)
    private_root, frozen_splits, leaderboard_df = load_leaderboard_dataframe()

    assert private_root is None
    assert set(frozen_splits) == {"public_leaderboard"}
    assert set(leaderboard_df["split"]) == {"public_leaderboard"}
