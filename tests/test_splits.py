from __future__ import annotations

from dataclasses import replace
import json

import pytest

from schema import DIFFICULTY_VERSION, GENERATOR_VERSION, SPEC_VERSION, TEMPLATE_SET_VERSION
from splits import (
    PARTITIONS,
    FrozenSplitEpisode,
    assert_no_partition_overlap,
    audit_frozen_splits,
    load_all_frozen_splits,
    load_frozen_split,
    load_split_manifest,
)
from validate import normalize_episode_payload


def _payload_fingerprint(record: FrozenSplitEpisode) -> str:
    payload = normalize_episode_payload(record.episode)
    payload = {
        key: value
        for key, value in payload.items()
        if key not in {"episode_id", "split"}
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


@pytest.mark.parametrize("partition", PARTITIONS)
def test_load_split_manifest_is_deterministic(partition: str):
    assert load_split_manifest(partition) == load_split_manifest(partition)


@pytest.mark.parametrize("partition", PARTITIONS)
def test_generate_frozen_split_is_deterministic(partition: str):
    assert load_frozen_split(partition) == load_frozen_split(partition)


@pytest.mark.parametrize("partition", PARTITIONS)
def test_frozen_split_records_preserve_manifest_metadata(partition: str):
    manifest = load_split_manifest(partition)
    split_records = load_frozen_split(partition)

    assert manifest.spec_version == SPEC_VERSION
    assert manifest.generator_version == GENERATOR_VERSION
    assert manifest.template_set_version == TEMPLATE_SET_VERSION
    assert manifest.difficulty_version == DIFFICULTY_VERSION

    assert tuple(record.seed for record in split_records) == manifest.seeds
    assert all(record.partition == partition for record in split_records)
    assert all(record.manifest_version == manifest.manifest_version for record in split_records)
    assert all(record.seed_bank_version == manifest.seed_bank_version for record in split_records)
    assert all(record.episode.split is manifest.episode_split for record in split_records)


def test_partition_names_remain_external_to_episode_schema():
    all_splits = load_all_frozen_splits()

    assert {record.episode.split.value for record in all_splits["dev"]} == {"dev"}
    assert {
        record.episode.split.value for record in all_splits["public_leaderboard"]
    } == {"public"}
    assert {
        record.episode.split.value for record in all_splits["private_leaderboard"]
    } == {"private"}


def test_frozen_partitions_have_no_seed_episode_id_or_payload_overlap():
    all_splits = load_all_frozen_splits()
    all_records = tuple(
        record
        for partition in PARTITIONS
        for record in all_splits[partition]
    )

    seeds = tuple(record.seed for record in all_records)
    episode_ids = tuple(record.episode.episode_id for record in all_records)
    payloads = tuple(_payload_fingerprint(record) for record in all_records)

    assert len(seeds) == len(set(seeds))
    assert len(episode_ids) == len(set(episode_ids))
    assert len(payloads) == len(set(payloads))
    assert_no_partition_overlap(all_splits) is None


def test_audit_summaries_are_stable_across_repeated_runs():
    audit_result = audit_frozen_splits()

    assert audit_result == audit_frozen_splits()
    assert audit_result.issues == ()
    assert tuple(partition for partition, _ in audit_result.per_partition) == PARTITIONS


def test_overlap_seed_fails_with_clear_reason():
    all_splits = load_all_frozen_splits()
    dev_record = all_splits["dev"][0]
    public_record = all_splits["public_leaderboard"][0]
    duplicated_public_record = replace(public_record, seed=dev_record.seed)
    mutated_splits = dict(all_splits)
    mutated_splits["public_leaderboard"] = (
        duplicated_public_record,
        *all_splits["public_leaderboard"][1:],
    )

    with pytest.raises(ValueError, match="overlap_seed"):
        assert_no_partition_overlap(mutated_splits)


def test_overlap_episode_id_fails_with_clear_reason():
    all_splits = load_all_frozen_splits()
    dev_record = all_splits["dev"][0]
    public_record = all_splits["public_leaderboard"][0]
    duplicated_episode = replace(
        public_record.episode,
        episode_id=dev_record.episode.episode_id,
    )
    duplicated_public_record = replace(public_record, episode=duplicated_episode)
    mutated_splits = dict(all_splits)
    mutated_splits["public_leaderboard"] = (
        duplicated_public_record,
        *all_splits["public_leaderboard"][1:],
    )

    with pytest.raises(ValueError, match="overlap_episode_id"):
        assert_no_partition_overlap(mutated_splits)


def test_overlap_payload_fails_with_clear_reason():
    all_splits = load_all_frozen_splits()
    dev_record = all_splits["dev"][0]
    public_record = all_splits["public_leaderboard"][0]
    duplicated_episode = replace(
        dev_record.episode,
        episode_id="ife-r3-payload-copy",
        split=public_record.episode.split,
    )
    duplicated_public_record = replace(public_record, episode=duplicated_episode)
    mutated_splits = dict(all_splits)
    mutated_splits["public_leaderboard"] = (
        duplicated_public_record,
        *all_splits["public_leaderboard"][1:],
    )

    with pytest.raises(ValueError, match="overlap_payload"):
        assert_no_partition_overlap(mutated_splits)
