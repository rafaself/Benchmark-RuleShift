"""Tests for the private evaluation split loader (Release 2).

Validates that:
- load_private_split() reads from materialized episodes, not private_leaderboard.json seeds
- The loaded episodes match what would be generated from private_leaderboard.json (round-trip)
- Structural invariants hold (episode count, split label, manifest metadata)
- Error paths work correctly (missing file, wrong partition, wrong manifest version)
- The default repo-local path resolves to packaging/kaggle/private/private_episodes.json
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from private_split import load_private_split, PRIVATE_EPISODES_FILENAME
from splits import load_split_manifest, generate_frozen_split, MANIFEST_VERSION
from validate import normalize_episode_payload


_REPO_ROOT = Path(__file__).resolve().parents[1]
_PRIVATE_DATASET_ROOT = _REPO_ROOT / "packaging" / "kaggle" / "private"
_PRIVATE_EPISODES_PATH = _PRIVATE_DATASET_ROOT / PRIVATE_EPISODES_FILENAME


# ---------------------------------------------------------------------------
# Basic loading
# ---------------------------------------------------------------------------

def test_load_private_split_returns_episodes():
    records = load_private_split(_PRIVATE_DATASET_ROOT)
    assert len(records) == 16


def test_load_private_split_default_path():
    """load_private_split() with no argument resolves the repo-local default."""
    records = load_private_split()
    assert len(records) == 16


def test_load_private_split_partition_label():
    records = load_private_split(_PRIVATE_DATASET_ROOT)
    for record in records:
        assert record.partition == "private_leaderboard"


def test_load_private_split_episode_split_label():
    records = load_private_split(_PRIVATE_DATASET_ROOT)
    from tasks.ruleshift_benchmark.protocol import Split
    for record in records:
        assert record.episode.split is Split.PRIVATE


def test_load_private_split_manifest_metadata():
    manifest = load_split_manifest("private_leaderboard")
    records = load_private_split(_PRIVATE_DATASET_ROOT)
    for record in records:
        assert record.manifest_version == manifest.manifest_version
        assert record.seed_bank_version == manifest.seed_bank_version


def test_load_private_split_episode_ids_match_seeds():
    records = load_private_split(_PRIVATE_DATASET_ROOT)
    for record in records:
        expected_id = f"ife-r12-{record.seed}"
        assert record.episode.episode_id == expected_id


# ---------------------------------------------------------------------------
# Round-trip: materialized == generated
# ---------------------------------------------------------------------------

def test_load_private_split_matches_generated_episodes():
    """Materialized private episodes must exactly match generator output."""
    manifest = load_split_manifest("private_leaderboard")
    generated = generate_frozen_split(manifest)
    loaded = load_private_split(_PRIVATE_DATASET_ROOT)

    assert len(loaded) == len(generated)
    for gen_record, mat_record in zip(generated, loaded):
        assert gen_record.seed == mat_record.seed
        assert gen_record.partition == mat_record.partition
        assert gen_record.manifest_version == mat_record.manifest_version
        assert gen_record.seed_bank_version == mat_record.seed_bank_version
        assert normalize_episode_payload(gen_record.episode) == normalize_episode_payload(mat_record.episode)


def test_load_private_split_probe_targets_are_valid():
    from tasks.ruleshift_benchmark.protocol import InteractionLabel
    records = load_private_split(_PRIVATE_DATASET_ROOT)
    for record in records:
        targets = record.episode.probe_targets
        assert len(targets) == 4
        for t in targets:
            assert isinstance(t, InteractionLabel)


# ---------------------------------------------------------------------------
# private_episodes.json file integrity
# ---------------------------------------------------------------------------

def test_private_episodes_file_exists():
    assert _PRIVATE_EPISODES_PATH.is_file()


def test_private_episodes_file_is_valid_json():
    data = json.loads(_PRIVATE_EPISODES_PATH.read_text(encoding="utf-8"))
    assert data["partition"] == "private_leaderboard"
    assert data["episode_split"] == "private"
    assert data["manifest_version"] == MANIFEST_VERSION
    assert data["episode_count"] == 16
    assert len(data["episodes"]) == 16


def test_private_episodes_independent_of_private_leaderboard_json():
    """load_private_split() must not read from private_leaderboard.json."""
    import importlib
    import core.private_split as module
    import inspect
    source = inspect.getsource(module)
    assert "private_leaderboard.json" not in source
    assert "load_split_manifest" not in source


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

def test_load_private_split_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_private_split(Path("/nonexistent/path/that/does/not/exist"))


def test_load_private_split_wrong_partition_raises():
    data = json.loads(_PRIVATE_EPISODES_PATH.read_text(encoding="utf-8"))
    data["partition"] = "dev"
    with tempfile.TemporaryDirectory() as tmpdir:
        bad_path = Path(tmpdir) / PRIVATE_EPISODES_FILENAME
        bad_path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match="partition"):
            load_private_split(Path(tmpdir))


def test_load_private_split_wrong_manifest_version_raises():
    data = json.loads(_PRIVATE_EPISODES_PATH.read_text(encoding="utf-8"))
    data["manifest_version"] = "R99"
    with tempfile.TemporaryDirectory() as tmpdir:
        bad_path = Path(tmpdir) / PRIVATE_EPISODES_FILENAME
        bad_path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match="manifest_version"):
            load_private_split(Path(tmpdir))


def test_load_private_split_empty_episodes_raises():
    data = json.loads(_PRIVATE_EPISODES_PATH.read_text(encoding="utf-8"))
    data["episodes"] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        bad_path = Path(tmpdir) / PRIVATE_EPISODES_FILENAME
        bad_path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match="episodes"):
            load_private_split(Path(tmpdir))


# ---------------------------------------------------------------------------
# Release 3: split rotation
# ---------------------------------------------------------------------------

def test_active_private_split_is_r14_private_3():
    """Active private split must be R14-private-3 (rotated from R14-private-2)."""
    data = json.loads(_PRIVATE_EPISODES_PATH.read_text(encoding="utf-8"))
    assert data["seed_bank_version"] == "R14-private-3"


def test_active_private_split_seeds_are_9108_to_9123():
    """Active private split must use the new seed range 9108–9123."""
    records = load_private_split(_PRIVATE_DATASET_ROOT)
    seeds = tuple(r.seed for r in records)
    assert seeds == tuple(range(9108, 9124))


def test_retired_split_r14_private_2_not_in_active_seeds():
    """Old R14-private-2 seeds 9060–9075 must not appear in the active split."""
    records = load_private_split(_PRIVATE_DATASET_ROOT)
    active_seeds = {r.seed for r in records}
    old_seeds = set(range(9060, 9076))  # R14-private-2 seeds
    assert active_seeds.isdisjoint(old_seeds), (
        f"Active split must not contain retired seeds; overlap: {active_seeds & old_seeds}"
    )


def test_private_episodes_retired_versions_field_lists_r14_private_2():
    """private_episodes.json must carry retired_seed_bank_versions to mark R14-private-2."""
    data = json.loads(_PRIVATE_EPISODES_PATH.read_text(encoding="utf-8"))
    assert "retired_seed_bank_versions" in data
    assert "R14-private-2" in data["retired_seed_bank_versions"]


def test_load_private_split_rejects_file_with_retired_version_as_active():
    """Loader must reject a private_episodes.json whose active version is in its retired list."""
    data = json.loads(_PRIVATE_EPISODES_PATH.read_text(encoding="utf-8"))
    # Simulate a stale file where the active version was not rotated
    data["seed_bank_version"] = "R14-private-2"
    data["retired_seed_bank_versions"] = ["R14-private-2"]
    with tempfile.TemporaryDirectory() as tmpdir:
        bad_path = Path(tmpdir) / PRIVATE_EPISODES_FILENAME
        bad_path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match="retired"):
            load_private_split(Path(tmpdir))


def test_private_split_no_overlap_with_dev_or_public():
    """New private seeds must not overlap with dev or public seeds."""
    from splits import load_split_manifest
    dev_seeds = set(load_split_manifest("dev").seeds)
    pub_seeds = set(load_split_manifest("public_leaderboard").seeds)
    records = load_private_split(_PRIVATE_DATASET_ROOT)
    priv_seeds = {r.seed for r in records}
    assert priv_seeds.isdisjoint(dev_seeds), "Private seeds overlap with dev"
    assert priv_seeds.isdisjoint(pub_seeds), "Private seeds overlap with public"
