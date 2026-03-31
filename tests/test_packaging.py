from __future__ import annotations

import json
from pathlib import Path
import tomllib

from core.kaggle.manifest import load_kaggle_staging_manifest, validate_kaggle_staging_manifest
from core.splits import MANIFEST_VERSION, PUBLIC_PARTITIONS, load_split_manifest
from tasks.ruleshift_benchmark.schema import (
    DIFFICULTY_VERSION,
    GENERATOR_VERSION,
    SPEC_VERSION,
    TEMPLATE_SET_VERSION,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_KAGGLE_DIR = _REPO_ROOT / "packaging" / "kaggle"
_NOTEBOOK_PATH = _KAGGLE_DIR / "ruleshift_notebook_task.ipynb"
_KERNEL_METADATA_PATH = _KAGGLE_DIR / "kernel-metadata.json"
_PYPROJECT_PATH = _REPO_ROOT / "pyproject.toml"


def _load_pyproject() -> dict[str, object]:
    return tomllib.loads(_PYPROJECT_PATH.read_text(encoding="utf-8"))


def _read_notebook_sources() -> str:
    notebook = json.loads(_NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", ())) for cell in notebook["cells"])


def test_public_kaggle_manifest_matches_runtime_contract():
    manifest = load_kaggle_staging_manifest()

    validate_kaggle_staging_manifest()

    assert set(manifest) == {"benchmark_versions", "entry_points", "frozen_split_manifests"}
    assert manifest["benchmark_versions"] == {
        "manifest_version": MANIFEST_VERSION,
        "spec_version": SPEC_VERSION,
        "generator_version": GENERATOR_VERSION,
        "template_set_version": TEMPLATE_SET_VERSION,
        "difficulty_version": DIFFICULTY_VERSION,
    }
    assert tuple(manifest["entry_points"]) == ("kbench_notebook", "kernel_metadata")
    assert tuple(manifest["frozen_split_manifests"]) == PUBLIC_PARTITIONS

    for partition in PUBLIC_PARTITIONS:
        artifact = manifest["frozen_split_manifests"][partition]
        split_manifest = load_split_manifest(partition)
        assert artifact["manifest_version"] == split_manifest.manifest_version
        assert artifact["seed_bank_version"] == split_manifest.seed_bank_version
        assert artifact["episode_split"] == split_manifest.episode_split.value


def test_packaging_directory_contains_only_runtime_files():
    top_level_files = sorted(path.name for path in _KAGGLE_DIR.iterdir() if path.is_file())

    assert top_level_files == [
        "dataset-metadata.json",
        "frozen_artifacts_manifest.json",
        "kernel-metadata.json",
        "ruleshift_notebook_task.ipynb",
    ]


def test_pyproject_keeps_default_install_surface_runtime_only():
    pyproject = _load_pyproject()

    assert pyproject["project"]["name"] == "ruleshift-benchmark"
    assert pyproject["project"]["dependencies"] == []
    assert "scripts" not in pyproject["project"]
    assert "optional-dependencies" not in pyproject["project"]
    assert "ruff" not in pyproject.get("tool", {})
    assert "mypy" not in pyproject.get("tool", {})
    assert pyproject["tool"]["setuptools"]["packages"]["find"]["include"] == ["core*", "tasks*"]


def test_readme_describes_runtime_surface_without_maintainer_or_benchmark_card_weight():
    readme_text = _REPO_ROOT.joinpath("README.md").read_text(encoding="utf-8")

    assert "scripts/build_runtime_dataset_package.py" in readme_text
    assert "scripts/build_kernel_package.py" in readme_text
    assert "src/frozen_splits/dev.json" in readme_text
    assert "src/frozen_splits/public_leaderboard.json" in readme_text
    assert "packaging/kaggle/BENCHMARK_CARD.md" not in readme_text
    assert "tools/maintainer" not in readme_text
    assert "contract-audit" not in readme_text
    assert "`score`" in readme_text
    assert "`manifest_version`" in readme_text
    assert "`narrative_result`" not in readme_text
    assert "`comparison`" not in readme_text
    assert "`slices`" not in readme_text


def test_notebook_imports_runtime_helpers_only():
    sources = _read_notebook_sources()

    assert "from core.kaggle import (" in sources
    assert "load_leaderboard_dataframe" in sources
    assert "run_binary_task" in sources
    assert "build_kaggle_payload" in sources
    assert "pd.DataFrame(" in sources
    assert ".style.hide(axis=\"index\")" in sources
    assert ".set_caption(" in sources
    assert ".to_string(index=False)" not in sources


def test_public_kernel_metadata_references_runtime_dataset_only():
    kernel_metadata = json.loads(_KERNEL_METADATA_PATH.read_text(encoding="utf-8"))

    assert kernel_metadata["dataset_sources"] == ["raptorengineer/ruleshift-runtime"]
    assert kernel_metadata["code_file"] == "ruleshift_notebook_task.ipynb"
