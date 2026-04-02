from __future__ import annotations

from pathlib import Path

from tasks.ruleshift_benchmark.runner import run_binary_task
from tasks.ruleshift_benchmark.splits import build_benchmark_bundle, build_leaderboard_rows

_EXPECTED_PUBLIC_EPISODES = 54
_EXPECTED_PRIVATE_EPISODES = 270


class _LLMStub:
    def __init__(self) -> None:
        self.call_count = 0

    def prompt(self, *_args, **_kwargs) -> dict[str, str]:
        self.call_count += 1
        return {
            "probe_6": "type_a",
            "probe_7": "type_b",
            "probe_8": "type_a",
            "probe_9": "type_b",
        }


def test_pipeline_runs_end_to_end_from_bundle_to_scoring(
    mounted_private_dataset_root: Path,
) -> None:
    bundle = build_benchmark_bundle(private_dataset_root=mounted_private_dataset_root)
    rows = build_leaderboard_rows(bundle)
    llm = _LLMStub()

    scores = [
        run_binary_task(
            llm=llm,
            prompt_binary=row["prompt_binary"],
            probe_targets=row["probe_targets"],
        )
        for row in rows
    ]

    numerator = sum(num_correct for num_correct, _total in scores)
    denominator = sum(total for _num_correct, total in scores)

    assert [partition["partition"] for partition in bundle["partitions"]] == [
        "public_leaderboard",
        "private_leaderboard",
    ]
    assert [partition["episode_count"] for partition in bundle["partitions"]] == [
        _EXPECTED_PUBLIC_EPISODES,
        _EXPECTED_PRIVATE_EPISODES,
    ]
    assert len(rows) == _EXPECTED_PUBLIC_EPISODES + _EXPECTED_PRIVATE_EPISODES
    assert {row["split"] for row in rows} == {
        "public_leaderboard",
        "private_leaderboard",
    }
    assert llm.call_count == len(rows)
    assert denominator == len(rows) * 4
    assert 0 <= numerator <= denominator
