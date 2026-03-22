from __future__ import annotations

import json
from pathlib import Path
import tomllib

from kaggle import load_kaggle_staging_manifest, validate_kaggle_staging_manifest
from schema import DIFFICULTY_VERSION, GENERATOR_VERSION, SPEC_VERSION, TEMPLATE_SET_VERSION
from splits import MANIFEST_VERSION, PARTITIONS, load_split_manifest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_KAGGLE_DIR = _REPO_ROOT / "packaging" / "kaggle"
_PYPROJECT_PATH = _REPO_ROOT / "pyproject.toml"
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


def _load_pyproject() -> dict[str, object]:
    return tomllib.loads(_PYPROJECT_PATH.read_text(encoding="utf-8"))


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


def test_pyproject_exposes_local_console_entrypoints():
    pyproject = _load_pyproject()

    assert pyproject["project"]["name"] == "iron-find-electric"
    assert pyproject["project"]["scripts"] == {
        "ife": "core.cli:entrypoint",
        "ife-test": "core.cli:test_entrypoint",
        "ife-validity": "core.cli:validity_entrypoint",
        "ife-reaudit": "core.cli:reaudit_entrypoint",
        "ife-integrity": "core.cli:integrity_entrypoint",
        "ife-evidence-pass": "core.cli:evidence_pass_entrypoint",
    }


def test_kaggle_staging_notebook_points_at_frozen_bundle_artifacts():
    sources = _read_notebook_sources()

    assert "frozen_artifacts_manifest.json" in sources
    assert "benchmark_card" in sources
    assert "anti_shortcut_validity_gate_r13" in sources
    assert "empirical_reaudit_r15" in sources
    assert "validate_kaggle_staging_manifest" in sources
    assert "load_split_manifest" in sources
    assert "load_frozen_split" in sources
    assert "run_model_benchmark" in sources
    assert "ModelMode.BINARY" in sources
    assert "ModelMode.NARRATIVE" in sources
    assert "same_probe_targets" in sources
    assert "Binary-only headline metric" in sources
    assert "Narrative does not change the headline score" in sources


def test_benchmark_card_matches_current_implementation_state():
    text = _CARD_PATH.read_text(encoding="utf-8")

    assert "Iron Find Electric v1" in text
    assert "narrow Executive Functions benchmark for cognitive flexibility" in text
    assert "controlled substrate" in text
    assert "frozen episodes" in text
    assert "Binary" in text
    assert "only leaderboard-primary" in text
    assert "Narrative" in text
    assert "same frozen episodes and probe targets as Binary" in text
    assert "only the final four labels are scored" in text
    assert "Post-shift Probe Accuracy" in text
    assert "Binary task" in text
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
        "physics skill",
        "full executive-function decomposition",
        "online detection latency",
        "switch cost",
        "recovery length",
        "immediate post-shift drop",
        "broad agi capability",
    )
    for phrase in explicit_non_claims:
        assert phrase in lowered

    assert "Binary-only Post-shift Probe Accuracy".lower() in lowered
    assert "Narrative does not change the headline score".lower() in lowered

    forbidden_positive_claims = (
        "hard leaderboard slice",
        "emitted hard slice",
        "hard episodes are included",
        "submission score claim",
        "measures switch cost",
        "measures online detection latency",
        "full executive-function decomposition benchmark",
        "broad adaptation benchmark",
        "physics benchmark",
    )

    for phrase in forbidden_positive_claims:
        assert phrase not in lowered
