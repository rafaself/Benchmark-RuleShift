from __future__ import annotations

import json
import tempfile
from pathlib import Path
import subprocess
import sys

import pytest

from core.private_split import (
    PRIVATE_DATASET_ROOT_ENV_VAR,
    PRIVATE_EPISODES_FILENAME,
    PRIVATE_SPLIT_ARTIFACT_SCHEMA_VERSION,
    build_private_split_artifact,
    load_private_split,
    load_private_split_manifest_info,
)
from core.splits import generate_frozen_split, load_frozen_split, load_split_manifest
from tasks.ruleshift_benchmark.protocol import InteractionLabel, Split


def test_load_private_split_returns_episodes(
    mounted_private_dataset_root: Path,
):
    records = load_private_split(mounted_private_dataset_root)
    assert len(records) == 16


def test_load_private_split_uses_mounted_environment_dataset():
    records = load_private_split()
    assert len(records) == 16


def test_load_private_split_partition_label(
    mounted_private_dataset_root: Path,
):
    records = load_private_split(mounted_private_dataset_root)
    for record in records:
        assert record.partition == "private_leaderboard"


def test_load_private_split_episode_split_label(
    mounted_private_dataset_root: Path,
):
    records = load_private_split(mounted_private_dataset_root)
    for record in records:
        assert record.episode.split is Split.PRIVATE


def test_load_private_split_manifest_metadata_matches_manifest_interface():
    manifest = load_split_manifest("private_leaderboard")
    records = load_private_split()

    for record in records:
        assert record.manifest_version == manifest.manifest_version
        assert record.seed_bank_version == manifest.seed_bank_version


def test_private_split_cannot_be_regenerated_from_manifest():
    manifest = load_split_manifest("private_leaderboard")
    with pytest.raises(ValueError, match="runtime regeneration is disabled"):
        generate_frozen_split(manifest)


def test_load_frozen_split_uses_private_loader_for_private_partition():
    direct = load_private_split()
    via_partition = load_frozen_split("private_leaderboard")
    assert via_partition == direct


def test_private_split_probe_targets_are_valid():
    records = load_private_split()
    for record in records:
        assert len(record.episode.probe_targets) == 4
        for target in record.episode.probe_targets:
            assert isinstance(target, InteractionLabel)


def test_private_manifest_info_comes_from_private_episodes_payload():
    payload = load_private_split_manifest_info()

    assert payload["benchmark_version"] == "R14"
    assert payload["schema_version"] == PRIVATE_SPLIT_ARTIFACT_SCHEMA_VERSION
    assert isinstance(payload["artifact_checksum"], str)
    assert payload["artifact_checksum"]
    assert payload["episode_split"] == "private"
    assert len(payload["seeds"]) == 16


def test_load_private_split_missing_explicit_root_raises():
    with pytest.raises(FileNotFoundError, match="private_episodes.json not found"):
        load_private_split(Path("/nonexistent/path/that/does/not/exist"))


def test_load_private_split_missing_mount_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv(PRIVATE_DATASET_ROOT_ENV_VAR, raising=False)
    with pytest.raises(
        FileNotFoundError,
        match="Private evaluation dataset is not attached",
    ):
        load_private_split()


def test_load_private_split_wrong_partition_raises(
    mounted_private_dataset_root: Path,
):
    data = json.loads(
        (mounted_private_dataset_root / PRIVATE_EPISODES_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    data["partition"] = "dev"
    with tempfile.TemporaryDirectory() as tmpdir:
        bad_path = Path(tmpdir) / PRIVATE_EPISODES_FILENAME
        bad_path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match="partition"):
            load_private_split(Path(tmpdir))


def test_load_private_split_wrong_benchmark_version_raises(
    mounted_private_dataset_root: Path,
):
    data = json.loads(
        (mounted_private_dataset_root / PRIVATE_EPISODES_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    data["benchmark_version"] = "R99"
    with tempfile.TemporaryDirectory() as tmpdir:
        bad_path = Path(tmpdir) / PRIVATE_EPISODES_FILENAME
        bad_path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match="benchmark_version"):
            load_private_split(Path(tmpdir))


def test_load_private_split_empty_episodes_raises(
    mounted_private_dataset_root: Path,
):
    data = json.loads(
        (mounted_private_dataset_root / PRIVATE_EPISODES_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    data["episodes"] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        bad_path = Path(tmpdir) / PRIVATE_EPISODES_FILENAME
        bad_path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match="episodes"):
            load_private_split(Path(tmpdir))


def test_load_private_split_rejects_checksum_mismatch(
    mounted_private_dataset_root: Path,
):
    data = json.loads(
        (mounted_private_dataset_root / PRIVATE_EPISODES_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    data["artifact_checksum"] = "bad-checksum"
    with tempfile.TemporaryDirectory() as tmpdir:
        bad_path = Path(tmpdir) / PRIVATE_EPISODES_FILENAME
        bad_path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match="artifact_checksum"):
            load_private_split(Path(tmpdir))


def test_offline_generation_script_writes_private_artifact(tmp_path: Path):
    seeds_path = tmp_path / "private_seeds.json"
    output_path = tmp_path / "mounted-private" / PRIVATE_EPISODES_FILENAME
    seeds = [31000, 31001, 31002, 31003]
    seeds_path.write_text(json.dumps(seeds), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/generate_private_split_artifact.py",
            "--benchmark-version",
            "R14",
            "--seeds-file",
            str(seeds_path),
            "--output",
            str(output_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert Path(completed.stdout.strip()) == output_path

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["benchmark_version"] == "R14"
    assert payload["schema_version"] == PRIVATE_SPLIT_ARTIFACT_SCHEMA_VERSION
    assert payload["artifact_checksum"]
    assert tuple(row["seed"] for row in payload["episodes"]) == tuple(seeds)
    assert len(load_private_split(output_path.parent)) == len(seeds)


def test_offline_generation_script_rejects_repo_local_output(tmp_path: Path):
    seeds_path = tmp_path / "private_seeds.json"
    seeds_path.write_text(json.dumps([32000, 32001]), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/generate_private_split_artifact.py",
            "--benchmark-version",
            "R14",
            "--seeds-file",
            str(seeds_path),
            "--output",
            "packaging/kaggle/private_episodes.json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode != 0
    assert "outside the public repository tree" in (completed.stderr or completed.stdout)


def test_build_private_split_artifact_requires_unique_seeds():
    with pytest.raises(ValueError, match="unique"):
        build_private_split_artifact(
            benchmark_version="R14",
            seeds=(1, 1),
        )
