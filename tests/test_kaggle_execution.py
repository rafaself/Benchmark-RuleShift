from __future__ import annotations

from dataclasses import fields

from pydantic import TypeAdapter
import pytest

from core.kaggle import KaggleExecutionError, load_leaderboard_dataframe, run_binary_task
from core.kaggle.runner import BinaryResponse, Label, normalize_binary_response

_EXPECTED_PUBLIC_EPISODES = 54
_EXPECTED_PRIVATE_EPISODES = 270
_EXPECTED_ATTACHED_EPISODES = _EXPECTED_PUBLIC_EPISODES + _EXPECTED_PRIVATE_EPISODES


class _RaisingLLM:
    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def prompt(self, *_args, **_kwargs):
        raise self._exc


class _StringLLM:
    def prompt(self, *_args, **_kwargs):
        return "type_a, type_b, type_a, type_b"


class _MappingLLM:
    def prompt(self, *_args, **_kwargs):
        return {
            "probe_6": "type_a",
            "probe_7": "type_b",
            "probe_8": "type_a",
            "probe_9": "type_b",
        }


class _SchemaAwareLLM:
    def __init__(self) -> None:
        self.last_schema = None

    def prompt(self, *_args, **kwargs):
        self.last_schema = kwargs.get("schema")
        return BinaryResponse(
            probe_6=Label.type_a,
            probe_7=Label.type_b,
            probe_8=Label.type_a,
            probe_9=Label.type_b,
        )


class _StringFieldBinaryResponseLLM:
    def prompt(self, *_args, **_kwargs):
        return BinaryResponse(
            probe_6="type_a",
            probe_7="type_b",
            probe_8="type_a",
            probe_9="type_b",
        )


class _InvalidBinaryResponseLLM:
    def prompt(self, *_args, **_kwargs):
        return BinaryResponse(
            probe_6="invalid_label",
            probe_7="type_b",
            probe_8="type_a",
            probe_9="type_b",
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

    targets = ("type_a", "type_b", "type_a", "type_b")

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
    targets = ("type_a", "type_b", "type_a", "type_b")
    llm = _SchemaAwareLLM()

    assert run_binary_task(
        llm=llm,
        prompt_binary="prompt",
        probe_targets=targets,
    ) == (4, 4)
    assert llm.last_schema is BinaryResponse


def test_binary_response_as_tuple_accepts_enum_fields():
    response = BinaryResponse(
        probe_6=Label.type_a,
        probe_7=Label.type_b,
        probe_8=Label.type_a,
        probe_9=Label.type_b,
    )

    assert response.as_tuple() == ("type_a", "type_b", "type_a", "type_b")


def test_binary_response_as_tuple_accepts_string_fields():
    response = BinaryResponse(
        probe_6="type_a",
        probe_7="type_b",
        probe_8="type_a",
        probe_9="type_b",
    )

    assert response.as_tuple() == ("type_a", "type_b", "type_a", "type_b")
    assert normalize_binary_response(response) == ("type_a", "type_b", "type_a", "type_b")


def test_binary_response_as_tuple_rejects_invalid_string_fields():
    response = BinaryResponse(
        probe_6="invalid_label",
        probe_7="type_b",
        probe_8="type_a",
        probe_9="type_b",
    )

    with pytest.raises(ValueError, match=r"probe_6"):
        response.as_tuple()


def test_normalize_binary_response_tolerates_stale_as_tuple(monkeypatch: pytest.MonkeyPatch):
    def _stale_as_tuple(self) -> tuple[str, str, str, str]:
        return (
            self.probe_6.value,
            self.probe_7.value,
            self.probe_8.value,
            self.probe_9.value,
        )

    monkeypatch.setattr(BinaryResponse, "as_tuple", _stale_as_tuple)

    response = BinaryResponse(
        probe_6="type_a",
        probe_7="type_b",
        probe_8="type_a",
        probe_9="type_b",
    )

    assert normalize_binary_response(response) == ("type_a", "type_b", "type_a", "type_b")


def test_run_binary_task_scores_valid_string_and_mapping_responses():
    targets = ("type_a", "type_b", "type_a", "type_b")

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
    targets = ("type_a", "type_b", "type_a", "type_b")

    assert run_binary_task(
        llm=_StringFieldBinaryResponseLLM(),
        prompt_binary="prompt",
        probe_targets=targets,
    ) == (4, 4)


def test_run_binary_task_raises_for_invalid_binary_response_fields():
    targets = ("type_a", "type_b", "type_a", "type_b")

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
    assert schema["$defs"]["Label"]["enum"] == ["type_a", "type_b"]


def test_run_binary_task_surfaces_provider_exception():
    targets = ("type_a", "type_b", "type_a", "type_b")

    with pytest.raises(KaggleExecutionError, match="llm.prompt failed"):
        run_binary_task(
            llm=_RaisingLLM(RuntimeError("provider unavailable")),
            prompt_binary="prompt",
            probe_targets=targets,
        )


def test_run_binary_task_surfaces_timeout_like_failure():
    targets = ("type_a", "type_b", "type_a", "type_b")

    with pytest.raises(KaggleExecutionError, match="llm.prompt failed"):
        run_binary_task(
            llm=_RaisingLLM(TimeoutError("provider timeout")),
            prompt_binary="prompt",
            probe_targets=targets,
        )


@pytest.mark.parametrize("legacy_output", ["attract, repel, attract, repel", "zark, blim, zark, blim"])
def test_normalize_binary_response_rejects_legacy_public_output_strings(legacy_output: str):
    assert normalize_binary_response(legacy_output) is None


@pytest.mark.parametrize("legacy_value", ["attract", "repel", "zark", "blim"])
def test_binary_response_as_tuple_rejects_legacy_public_output_fields(legacy_value: str):
    response = BinaryResponse(
        probe_6=legacy_value,
        probe_7="type_b",
        probe_8="type_a",
        probe_9="type_b",
    )

    with pytest.raises(ValueError, match=r"probe_6"):
        response.as_tuple()


def test_load_leaderboard_dataframe_preserves_public_private_behavior(monkeypatch):
    private_root, frozen_splits, leaderboard_df = load_leaderboard_dataframe()

    assert private_root is not None
    assert set(frozen_splits) == {"public_leaderboard", "private_leaderboard"}
    assert set(leaderboard_df["split"]) == {"public_leaderboard", "private_leaderboard"}
    assert len(frozen_splits["public_leaderboard"]) == _EXPECTED_PUBLIC_EPISODES
    assert len(frozen_splits["private_leaderboard"]) == _EXPECTED_PRIVATE_EPISODES
    assert len(leaderboard_df) == _EXPECTED_ATTACHED_EPISODES

    monkeypatch.delenv("RULESHIFT_PRIVATE_DATASET_ROOT", raising=False)
    private_root, frozen_splits, leaderboard_df = load_leaderboard_dataframe()

    assert private_root is None
    assert set(frozen_splits) == {"public_leaderboard"}
    assert set(leaderboard_df["split"]) == {"public_leaderboard"}
    assert len(frozen_splits["public_leaderboard"]) == _EXPECTED_PUBLIC_EPISODES
    assert len(leaderboard_df) == _EXPECTED_PUBLIC_EPISODES
