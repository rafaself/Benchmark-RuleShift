from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CANONICAL_KERNEL_METADATA = json.loads(
    (_REPO_ROOT / "packaging" / "kaggle" / "kernel-metadata.json").read_text(encoding="utf-8")
)
_CANONICAL_DATASET_METADATA = json.loads(
    (_REPO_ROOT / "packaging" / "kaggle" / "dataset-metadata.json").read_text(encoding="utf-8")
)
_FORBIDDEN_FILENAMES = ("private_leaderboard.json", "private_episodes.json")
_WORKFLOW_EXPECTED_SCRIPTS = {
    ".github/workflows/deploy-kaggle-dataset.yml": "scripts/build_runtime_dataset_package.py",
    ".github/workflows/deploy-kaggle-notebook.yml": "scripts/build_kernel_package.py",
}
_WORKFLOW_FILES = tuple(
    sorted(path.relative_to(_REPO_ROOT).as_posix() for path in (_REPO_ROOT / ".github" / "workflows").glob("*.yml"))
)
_PINNED_ACTIONS = {
    "actions/checkout": "08eba0b27e820071cde6df949e0beb9ba4906955",
    "actions/setup-python": "a26af69be951a213d495a4c3e4e4022e16d87065",
}
_EXPECTED_RUNTIME_SOURCE_FILES = {
    "src/core/__init__.py",
    "src/core/kaggle/__init__.py",
    "src/core/kaggle/manifest.py",
    "src/core/kaggle/payload.py",
    "src/core/kaggle/runner.py",
    "src/core/private_split.py",
    "src/core/splits.py",
    "src/frozen_splits/dev.json",
    "src/frozen_splits/public_leaderboard.json",
    "src/tasks/__init__.py",
    "src/tasks/ruleshift_benchmark/__init__.py",
    "src/tasks/ruleshift_benchmark/generator.py",
    "src/tasks/ruleshift_benchmark/protocol.py",
    "src/tasks/ruleshift_benchmark/render.py",
    "src/tasks/ruleshift_benchmark/rules.py",
    "src/tasks/ruleshift_benchmark/schema.py",
}


def _run(script: str, extra_args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, script, *extra_args],
        cwd=_REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )


def _files_in(directory: Path) -> set[str]:
    return {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}


def test_canonical_metadata_is_runtime_ready():
    for name, metadata in [
        ("dataset-metadata.json", _CANONICAL_DATASET_METADATA),
        ("kernel-metadata.json", _CANONICAL_KERNEL_METADATA),
    ]:
        assert "KAGGLE_USERNAME" not in json.dumps(metadata)
        owner, slug = metadata["id"].split("/")
        assert owner and slug, f"{name} id must be owner/slug"

    assert _CANONICAL_DATASET_METADATA["id"] in _CANONICAL_KERNEL_METADATA["dataset_sources"]


def test_deploy_workflows_reference_canonical_build_scripts():
    for workflow_path, expected_script in _WORKFLOW_EXPECTED_SCRIPTS.items():
        workflow_text = (_REPO_ROOT / workflow_path).read_text(encoding="utf-8")
        assert expected_script in workflow_text
        assert "scripts/cd/" not in workflow_text


def test_only_lean_deploy_workflows_remain_and_are_hardened():
    assert _WORKFLOW_FILES == (
        ".github/workflows/deploy-kaggle-dataset.yml",
        ".github/workflows/deploy-kaggle-notebook.yml",
    )

    for workflow_path in _WORKFLOW_FILES:
        workflow_text = (_REPO_ROOT / workflow_path).read_text(encoding="utf-8")

        assert "workflow_call:" not in workflow_text
        assert "permissions:\n  contents: read" in workflow_text
        assert "concurrency:" in workflow_text
        assert "cancel-in-progress: false" in workflow_text
        assert 'python-version: "3.11"' in workflow_text
        assert "cache: pip" in workflow_text
        assert "cache-dependency-path: requirements-dev.txt" in workflow_text
        assert "pip install -r requirements-dev.txt" in workflow_text
        assert "pip install -e ." in workflow_text
        assert "pip install \"kaggle==${KAGGLE_CLI_VERSION}\"" in workflow_text
        assert "actions/upload-artifact" not in workflow_text

        for action_name, sha in _PINNED_ACTIONS.items():
            assert f"uses: {action_name}@{sha}" in workflow_text

        unpinned_actions = re.findall(r"uses:\s+([^@\s]+)@([^\s]+)", workflow_text)
        assert all(len(ref) == 40 for _, ref in unpinned_actions)


@pytest.fixture(scope="module")
def runtime_package(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("runtime-package")
    result = _run("scripts/build_runtime_dataset_package.py", ["--output-dir", str(out)])
    assert result.returncode == 0, result.stderr or result.stdout
    assert Path(result.stdout.strip()) == out
    return out


def test_runtime_dataset_build_output_is_exact_and_public(runtime_package: Path):
    assert {path.name for path in runtime_package.iterdir()} == {"dataset-metadata.json", "packaging", "src"}

    all_files = _files_in(runtime_package)
    assert {
        path for path in all_files if path.startswith("src/")
    } == _EXPECTED_RUNTIME_SOURCE_FILES
    assert (runtime_package / "packaging" / "kaggle" / "frozen_artifacts_manifest.json").is_file()
    assert {p.name for p in (runtime_package / "src" / "frozen_splits").iterdir()} == {
        "dev.json",
        "public_leaderboard.json",
    }
    assert "__pycache__" not in {part for path in runtime_package.rglob("*") for part in path.parts}
    assert not list(runtime_package.rglob("*.pyc"))

    for forbidden in _FORBIDDEN_FILENAMES:
        assert all(Path(path).name != forbidden for path in all_files)


def test_runtime_dataset_metadata_preserves_canonical_fields_and_exact_resources(runtime_package: Path):
    packaged = json.loads((runtime_package / "dataset-metadata.json").read_text(encoding="utf-8"))

    for key, value in _CANONICAL_DATASET_METADATA.items():
        assert packaged[key] == value

    resource_paths = [entry["path"] for entry in packaged["resources"]]
    assert resource_paths == sorted(resource_paths)
    assert len(resource_paths) == len(set(resource_paths))
    assert "dataset-metadata.json" not in resource_paths
    assert set(resource_paths) == _files_in(runtime_package) - {"dataset-metadata.json"}


@pytest.fixture(scope="module")
def kernel_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("kernel-bundle")
    result = _run("scripts/build_kernel_package.py", ["--output-dir", str(out)])
    assert result.returncode == 0, result.stderr or result.stdout
    assert Path(result.stdout.strip()) == out
    return out


def test_kernel_bundle_is_canonical(kernel_bundle: Path):
    assert {path.name for path in kernel_bundle.iterdir()} == {
        "kernel-metadata.json",
        "ruleshift_notebook_task.ipynb",
    }

    notebook_src = _REPO_ROOT / "packaging" / "kaggle" / "ruleshift_notebook_task.ipynb"
    notebook_dest = kernel_bundle / "ruleshift_notebook_task.ipynb"
    assert hashlib.sha256(notebook_src.read_bytes()).hexdigest() == hashlib.sha256(
        notebook_dest.read_bytes()
    ).hexdigest()

    kernel_metadata = json.loads((kernel_bundle / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert kernel_metadata == _CANONICAL_KERNEL_METADATA
    assert "KAGGLE_USERNAME" not in json.dumps(kernel_metadata)

    notebook = json.loads(notebook_dest.read_text(encoding="utf-8"))
    last_code = next(cell for cell in reversed(notebook["cells"]) if cell.get("cell_type") == "code")
    magic_lines = [line.strip() for line in "".join(last_code.get("source", ())).splitlines() if line.strip().startswith("%")]
    assert magic_lines == ["%choose ruleshift_benchmark_v1_binary"]
