"""Deployment build validation — gates required before any Kaggle upload.

Invokes the two CD scripts as subprocesses (same way CI would) and inspects
the produced artifacts.  These tests are intentionally coarser than the
packaging unit tests: they validate the *built output*, not the source files.

Checks:
  Runtime dataset package (build_runtime_dataset_package.py)
    - script exits 0
    - exact top-level file set (dataset-metadata.json + src/ + packaging/)
    - dataset-metadata.json has correct id, title, license
    - dataset-metadata.json resources list is non-empty and consistent
    - src/frozen_splits/ contains dev.json and public_leaderboard.json only
    - no private artifact appears anywhere in the output tree
    - packaging/kaggle/frozen_artifacts_manifest.json is present

  Kernel bundle (build_kernel_package.py)
    - script exits 0
    - exact file set (kernel-metadata.json + notebook)
    - notebook is byte-identical to the source
    - kernel-metadata.json fields: id, code_file, language, kernel_type,
      is_private, enable_gpu, enable_tpu, enable_internet
    - dataset_sources contains the slug passed via --runtime-dataset-slug
    - competition_sources and kernel_sources are empty lists
    - last code cell in copied notebook contains %choose ruleshift_benchmark_v1_binary
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RUNTIME_DATASET_SLUG = "raptorengineer/ruleshift-runtime"
_FORBIDDEN_FILENAMES = ("private_leaderboard.json", "private_episodes.json")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _run(script: str, extra_args: list[str], cwd: Path = _REPO_ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, script, *extra_args],
        cwd=cwd,
        check=False,
        text=True,
        capture_output=True,
    )


def _files_in(directory: Path) -> set[str]:
    """Return all file paths relative to directory as a set of POSIX strings."""
    return {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}


def _top_level_names(directory: Path) -> set[str]:
    return {p.name for p in directory.iterdir()}


# ---------------------------------------------------------------------------
# Runtime dataset package
# ---------------------------------------------------------------------------


class TestRuntimeDatasetPackage:
    @pytest.fixture(scope="class")
    def package_dir(self, tmp_path_factory: pytest.TempPathFactory) -> Path:
        out = tmp_path_factory.mktemp("runtime-package")
        result = _run(
            "scripts/cd/build_runtime_dataset_package.py",
            ["--output-dir", str(out)],
        )
        assert result.returncode == 0, result.stderr or result.stdout
        assert Path(result.stdout.strip()) == out
        return out

    def test_script_produces_output_directory(self, package_dir: Path):
        assert package_dir.is_dir()

    def test_top_level_structure(self, package_dir: Path):
        top = _top_level_names(package_dir)
        assert "dataset-metadata.json" in top
        assert "src" in top
        assert "packaging" in top
        # nothing else at the top level
        assert top == {"dataset-metadata.json", "src", "packaging"}

    def test_dataset_metadata_id(self, package_dir: Path):
        meta = json.loads((package_dir / "dataset-metadata.json").read_text(encoding="utf-8"))
        assert meta["id"] == _RUNTIME_DATASET_SLUG

    def test_dataset_metadata_title(self, package_dir: Path):
        meta = json.loads((package_dir / "dataset-metadata.json").read_text(encoding="utf-8"))
        assert isinstance(meta["title"], str) and meta["title"]

    def test_dataset_metadata_license(self, package_dir: Path):
        meta = json.loads((package_dir / "dataset-metadata.json").read_text(encoding="utf-8"))
        assert meta["licenses"] == [{"name": "CC0-1.0"}]

    def test_dataset_metadata_resources_non_empty(self, package_dir: Path):
        meta = json.loads((package_dir / "dataset-metadata.json").read_text(encoding="utf-8"))
        assert isinstance(meta["resources"], list) and len(meta["resources"]) > 0

    def test_dataset_metadata_resources_list_all_real_files(self, package_dir: Path):
        meta = json.loads((package_dir / "dataset-metadata.json").read_text(encoding="utf-8"))
        for entry in meta["resources"]:
            assert (package_dir / entry["path"]).is_file(), (
                f"resource {entry['path']!r} listed in dataset-metadata.json but not found"
            )

    def test_dataset_metadata_self_not_in_resources(self, package_dir: Path):
        meta = json.loads((package_dir / "dataset-metadata.json").read_text(encoding="utf-8"))
        resource_paths = {e["path"] for e in meta["resources"]}
        assert "dataset-metadata.json" not in resource_paths

    def test_frozen_splits_contains_only_public_manifests(self, package_dir: Path):
        splits_dir = package_dir / "src" / "frozen_splits"
        assert splits_dir.is_dir()
        present = {p.name for p in splits_dir.iterdir() if p.is_file()}
        assert present == {"dev.json", "public_leaderboard.json"}

    def test_frozen_artifacts_manifest_is_present(self, package_dir: Path):
        assert (package_dir / "packaging" / "kaggle" / "frozen_artifacts_manifest.json").is_file()

    def test_no_private_artifacts_in_output(self, package_dir: Path):
        all_files = _files_in(package_dir)
        for forbidden in _FORBIDDEN_FILENAMES:
            matches = [f for f in all_files if Path(f).name == forbidden]
            assert not matches, (
                f"Private artifact {forbidden!r} found in runtime package output: {matches}"
            )

    def test_no_pycache_in_output(self, package_dir: Path):
        all_parts = {part for p in package_dir.rglob("*") for part in p.parts}
        assert "__pycache__" not in all_parts
        pyc_files = [p for p in package_dir.rglob("*.pyc")]
        assert not pyc_files, f"Compiled bytecode found in output: {pyc_files}"


# ---------------------------------------------------------------------------
# Kernel bundle
# ---------------------------------------------------------------------------


class TestKernelBundle:
    @pytest.fixture(scope="class")
    def bundle_dir(self, tmp_path_factory: pytest.TempPathFactory) -> Path:
        out = tmp_path_factory.mktemp("kernel-bundle")
        result = _run(
            "scripts/cd/build_kernel_package.py",
            [
                "--runtime-dataset-slug", _RUNTIME_DATASET_SLUG,
                "--output-dir", str(out),
            ],
        )
        assert result.returncode == 0, result.stderr or result.stdout
        assert Path(result.stdout.strip()) == out
        return out

    def test_script_produces_output_directory(self, bundle_dir: Path):
        assert bundle_dir.is_dir()

    def test_exact_file_set(self, bundle_dir: Path):
        files = _top_level_names(bundle_dir)
        assert files == {"kernel-metadata.json", "ruleshift_notebook_task.ipynb"}

    def test_notebook_is_byte_identical_to_source(self, bundle_dir: Path):
        import hashlib

        source = _REPO_ROOT / "packaging" / "kaggle" / "ruleshift_notebook_task.ipynb"
        dest = bundle_dir / "ruleshift_notebook_task.ipynb"
        assert hashlib.sha256(source.read_bytes()).hexdigest() == \
               hashlib.sha256(dest.read_bytes()).hexdigest()

    def test_kernel_metadata_id(self, bundle_dir: Path):
        meta = json.loads((bundle_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
        assert meta["id"] == "raptorengineer/ruleshift-notebook-task"

    def test_kernel_metadata_code_file_matches_notebook(self, bundle_dir: Path):
        meta = json.loads((bundle_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
        assert meta["code_file"] == "ruleshift_notebook_task.ipynb"
        assert (bundle_dir / meta["code_file"]).is_file()

    def test_kernel_metadata_language(self, bundle_dir: Path):
        meta = json.loads((bundle_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
        assert meta["language"] == "python"

    def test_kernel_metadata_kernel_type(self, bundle_dir: Path):
        meta = json.loads((bundle_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
        assert meta["kernel_type"] == "notebook"

    def test_kernel_metadata_is_private(self, bundle_dir: Path):
        meta = json.loads((bundle_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
        assert meta["is_private"] is True

    def test_kernel_metadata_compute_flags(self, bundle_dir: Path):
        meta = json.loads((bundle_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
        assert meta["enable_gpu"] is False
        assert meta["enable_tpu"] is False
        assert meta["enable_internet"] is False

    def test_kernel_metadata_dataset_sources_contains_slug(self, bundle_dir: Path):
        meta = json.loads((bundle_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
        assert _RUNTIME_DATASET_SLUG in meta["dataset_sources"]

    def test_kernel_metadata_dataset_sources_single_entry(self, bundle_dir: Path):
        meta = json.loads((bundle_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
        assert meta["dataset_sources"] == [_RUNTIME_DATASET_SLUG]

    def test_kernel_metadata_competition_and_kernel_sources_empty(self, bundle_dir: Path):
        meta = json.loads((bundle_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
        assert meta["competition_sources"] == []
        assert meta["kernel_sources"] == []

    def test_last_code_cell_contains_choose(self, bundle_dir: Path):
        notebook = json.loads(
            (bundle_dir / "ruleshift_notebook_task.ipynb").read_text(encoding="utf-8")
        )
        last_code = next(
            c for c in reversed(notebook["cells"]) if c.get("cell_type") == "code"
        )
        source = "".join(last_code.get("source", ()))
        magic_lines = [
            line.strip() for line in source.splitlines() if line.strip().startswith("%")
        ]
        assert magic_lines == ["%choose ruleshift_benchmark_v1_binary"], (
            f"Last code cell in bundled notebook must contain exactly "
            f"'%choose ruleshift_benchmark_v1_binary', got: {magic_lines!r}"
        )

    def test_kernel_package_slug_injection(self, tmp_path: Path):
        """Passing a different slug produces a kernel-metadata.json with that slug."""
        custom_slug = "owner/custom-runtime-slug"
        result = _run(
            "scripts/cd/build_kernel_package.py",
            ["--runtime-dataset-slug", custom_slug, "--output-dir", str(tmp_path)],
        )
        assert result.returncode == 0, result.stderr or result.stdout
        meta = json.loads((tmp_path / "kernel-metadata.json").read_text(encoding="utf-8"))
        assert meta["dataset_sources"] == [custom_slug]
