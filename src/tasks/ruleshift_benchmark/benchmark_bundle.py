from __future__ import annotations

from pathlib import Path
from typing import Final

from tasks.ruleshift_benchmark.protocol import format_public_label
from tasks.ruleshift_benchmark.render import render_binary_prompt
from tasks.ruleshift_benchmark.splits import (
    MANIFEST_VERSION,
    discover_private_dataset_root,
    load_frozen_split,
    load_split_manifest,
    resolve_private_dataset_root,
)

__all__ = [
    "TASK_NAME",
    "build_benchmark_bundle",
    "build_leaderboard_rows",
]

TASK_NAME: Final[str] = "ruleshift_benchmark"


def build_benchmark_bundle(
    *,
    include_private: bool = True,
    private_dataset_root: Path | str | None = None,
) -> dict[str, object]:
    partitions = [_build_partition_bundle("public_leaderboard")]

    if include_private:
        resolved_private_root = _resolve_optional_private_dataset_root(private_dataset_root)
        if resolved_private_root is not None:
            partitions.append(
                _build_partition_bundle(
                    "private_leaderboard",
                    private_dataset_root=resolved_private_root,
                )
            )

    return {
        "task": TASK_NAME,
        "benchmark_version": MANIFEST_VERSION,
        "partitions": partitions,
    }


def _build_partition_bundle(
    partition: str,
    *,
    private_dataset_root: Path | str | None = None,
) -> dict[str, object]:
    manifest = load_split_manifest(partition, private_dataset_root=private_dataset_root)
    records = load_frozen_split(partition, private_dataset_root=private_dataset_root)

    return {
        "partition": manifest.partition,
        "episode_split": manifest.episode_split.value,
        "manifest_version": manifest.manifest_version,
        "seed_bank_version": manifest.seed_bank_version,
        "episode_count": len(records),
        "episodes": [
            {
                "seed": record.seed,
                "episode_id": record.episode.episode_id,
                "prompt_binary": render_binary_prompt(record.episode),
                "probe_targets": [
                    format_public_label(label) for label in record.episode.probe_targets
                ],
            }
            for record in records
        ],
    }


def build_leaderboard_rows(bundle: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for partition in bundle["partitions"]:
        split_name = partition["partition"]
        for episode in partition["episodes"]:
            rows.append(
                {
                    "episode_id": episode["episode_id"],
                    "split": split_name,
                    "prompt_binary": episode["prompt_binary"],
                    "probe_targets": tuple(episode["probe_targets"]),
                }
            )
    return rows


def _resolve_optional_private_dataset_root(
    private_dataset_root: Path | str | None,
) -> Path | None:
    if private_dataset_root is not None:
        return resolve_private_dataset_root(private_dataset_root)
    return discover_private_dataset_root()
