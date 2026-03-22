from __future__ import annotations

import json
from pathlib import Path

from kaggle import load_kaggle_staging_manifest, validate_kaggle_staging_manifest
from schema import DIFFICULTY_VERSION, GENERATOR_VERSION, SPEC_VERSION, TEMPLATE_SET_VERSION
from splits import MANIFEST_VERSION, PARTITIONS, load_split_manifest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_KAGGLE_DIR = _REPO_ROOT / "packaging" / "kaggle"
_NOTEBOOK_PATH = _KAGGLE_DIR / "iron_find_electric_v1_kaggle_staging.ipynb"
_CARD_PATH = _KAGGLE_DIR / "BENCHMARK_CARD.md"
_USAGE_PATH = _KAGGLE_DIR / "README.md"
_PACKAGING_NOTE_PATH = _KAGGLE_DIR / "PACKAGING_NOTE.md"


def _read_notebook_sources() -> str:
    notebook = json.loads(_NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", ()))
        for cell in notebook["cells"]
    )


def test_kaggle_staging_manifest_resolves_current_frozen_artifacts():
    manifest = load_kaggle_staging_manifest()

    validate_kaggle_staging_manifest()

    assert manifest["bundle_version"] == "R16"
    assert manifest["task_id"] == "iron_find_electric_v1"
    assert manifest["task_name"] == "Iron Find Electric v1"

    benchmark_versions = manifest["benchmark_versions"]
    assert benchmark_versions == {
        "manifest_version": MANIFEST_VERSION,
        "spec_version": SPEC_VERSION,
        "generator_version": GENERATOR_VERSION,
        "template_set_version": TEMPLATE_SET_VERSION,
        "difficulty_version": DIFFICULTY_VERSION,
    }

    assert tuple(manifest["frozen_split_manifests"]) == PARTITIONS
    assert manifest["current_emitted_difficulty_labels"] == ["easy", "medium"]
    assert manifest["reserved_difficulty_labels"] == ["hard"]

    for partition in PARTITIONS:
        artifact = manifest["frozen_split_manifests"][partition]
        manifest_path = _REPO_ROOT / artifact["path"]
        split_manifest = load_split_manifest(partition)

        assert manifest_path.is_file()
        assert manifest_path.name == f"{partition}.json"
        assert artifact["manifest_version"] == split_manifest.manifest_version
        assert artifact["seed_bank_version"] == split_manifest.seed_bank_version
        assert artifact["episode_split"] == split_manifest.episode_split.value


def test_kaggle_staging_notebook_points_at_frozen_bundle_artifacts():
    sources = _read_notebook_sources()

    assert "frozen_artifacts_manifest.json" in sources
    assert "benchmark_card" in sources
    assert "anti_shortcut_validity_gate_r13" in sources
    assert "empirical_reaudit_r15" in sources
    assert "validate_kaggle_staging_manifest" in sources
    assert "load_split_manifest" in sources


def test_benchmark_card_matches_current_implementation_state():
    text = _CARD_PATH.read_text(encoding="utf-8")

    assert "Iron Find Electric v1" in text
    assert "cognitive flexibility under hidden rule shift" in text
    assert "Binary" in text
    assert "leaderboard-primary" in text
    assert "Narrative" in text
    assert "robustness companion" in text
    assert "Post-shift Probe Accuracy" in text
    assert "`random`" in text
    assert "`never_update`" in text
    assert "`last_evidence`" in text
    assert "`physics_prior`" in text
    assert "`template_position`" in text
    assert "easy" in text
    assert "medium" in text
    assert "reserved and not emitted" in text
    assert "recency shortcut was materially reduced" in text
    assert "R13 anti-shortcut validity gate" in text
    assert "R15 empirical re-audit" in text


def test_packaging_docs_do_not_claim_unsupported_features():
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (_CARD_PATH, _USAGE_PATH, _PACKAGING_NOTE_PATH)
    )
    text += "\n" + _read_notebook_sources()
    lowered = text.lower()

    assert "does **not** claim" in text

    explicit_non_claims = (
        "full executive-function decomposition",
        "online detection latency",
        "switch cost",
    )
    for phrase in explicit_non_claims:
        assert phrase in lowered

    forbidden_positive_claims = (
        "hard leaderboard slice",
        "emitted hard slice",
        "hard episodes are included",
        "submission score claim",
        "measures switch cost",
        "measures online detection latency",
        "full executive-function decomposition benchmark",
    )

    for phrase in forbidden_positive_claims:
        assert phrase not in lowered
