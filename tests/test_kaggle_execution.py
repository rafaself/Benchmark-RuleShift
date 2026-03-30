from __future__ import annotations

from dataclasses import dataclass

from core.kaggle import load_leaderboard_dataframe, run_binary_task


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


def test_run_binary_task_scores_string_and_mapping_responses():
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


def test_run_binary_task_treats_invalid_and_provider_failures_as_zero_score():
    @dataclass(frozen=True)
    class _BadShapeLLM:
        def prompt(self, *_args, **_kwargs):
            return {"unexpected": 1}

    targets = ("attract", "repel", "attract", "repel")

    assert run_binary_task(
        llm=_BadShapeLLM(),
        prompt_binary="prompt",
        probe_targets=targets,
    ) == (0, 4)
    assert run_binary_task(
        llm=_RaisingLLM(RuntimeError("provider timeout")),
        prompt_binary="prompt",
        probe_targets=targets,
    ) == (0, 4)


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
