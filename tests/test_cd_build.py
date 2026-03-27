"""Deployment build validation — gates required before any Kaggle upload.

Invokes the two CD scripts as subprocesses (same way CI would) and inspects
the produced artifacts.  These tests are intentionally coarser than the
packaging unit tests: they validate the *built output*, not the source files.

Checks:
  Canonical metadata (source-of-truth regression)
    - no placeholder values (KAGGLE_USERNAME must not appear)
    - id fields are owner/slug format
    - kernel dataset_sources references the canonical dataset id
    - removed identity arguments are rejected by the build scripts

  Runtime dataset package (build_runtime_dataset_package.py)
    - script exits 0
    - exact top-level file set (dataset-metadata.json + src/ + packaging/)
    - dataset-metadata.json id/title/licenses match canonical dataset-metadata.json
    - packaged dataset metadata keeps canonical fields unchanged and adds only resources
    - dataset-metadata.json resources list is non-empty and consistent
    - dataset-metadata.json resources list is sorted, unique, and exactly matches the built file set
    - packaged metadata contains no placeholder values
    - src/frozen_splits/ contains dev.json and public_leaderboard.json only
    - no private artifact appears anywhere in the output tree
    - packaging/kaggle/frozen_artifacts_manifest.json is present
    - optional audit manifest captures metadata checksums and output file inventory

  Kernel bundle (build_kernel_package.py)
    - script exits 0
    - exact file set (kernel-metadata.json + notebook)
    - notebook is byte-identical to the source
    - kernel-metadata.json is content-identical to canonical kernel-metadata.json
    - packaged metadata contains no placeholder values
    - last code cell in copied notebook contains %choose ruleshift_benchmark_v1_binary
    - optional audit manifest captures metadata checksums and output file inventory
"""

from __future__ import annotations

import json
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


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Canonical metadata — source-of-truth regression tests
# ---------------------------------------------------------------------------


def test_canonical_metadata_contains_no_placeholders():
    """Fail if KAGGLE_USERNAME is reintroduced as a value in either metadata file."""
    for name, meta_dict in [
        ("kernel-metadata.json", _CANONICAL_KERNEL_METADATA),
        ("dataset-metadata.json", _CANONICAL_DATASET_METADATA),
    ]:
        assert "KAGGLE_USERNAME" not in json.dumps(meta_dict), (
            f"{name} contains the placeholder string 'KAGGLE_USERNAME'"
        )


def test_canonical_metadata_ids_are_owner_slug():
    """Both id fields must be owner/slug — no template tokens, no missing owner."""
    for name, meta_dict in [
        ("kernel-metadata.json", _CANONICAL_KERNEL_METADATA),
        ("dataset-metadata.json", _CANONICAL_DATASET_METADATA),
    ]:
        id_value = meta_dict.get("id", "")
        parts = id_value.split("/")
        assert len(parts) == 2 and all(parts), (
            f"{name} id must be owner/slug format, got {id_value!r}"
        )


def test_kernel_dataset_sources_reference_canonical_dataset():
    """Canonical cross-reference: kernel dataset_sources must include the dataset id."""
    assert _CANONICAL_DATASET_METADATA["id"] in _CANONICAL_KERNEL_METADATA["dataset_sources"], (
        "kernel-metadata.json dataset_sources and dataset-metadata.json id are out of sync"
    )


def test_build_script_rejects_removed_runtime_slug_argument(tmp_path: Path):
    """Regression: --runtime-dataset-slug must no longer be accepted by the kernel build script."""
    result = _run(
        "scripts/cd/build_kernel_package.py",
        ["--runtime-dataset-slug", "owner/slug", "--output-dir", str(tmp_path)],
    )
    assert result.returncode != 0, (
        "build_kernel_package.py accepted --runtime-dataset-slug, which must have been removed"
    )


# ---------------------------------------------------------------------------
# Runtime dataset package
# ---------------------------------------------------------------------------


class TestRuntimeDatasetPackage:
    @pytest.fixture(scope="class")
    def build_output(self, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
        out = tmp_path_factory.mktemp("runtime-package")
        audit_manifest = tmp_path_factory.mktemp("runtime-package-audit") / "audit.json"
        result = _run(
            "scripts/cd/build_runtime_dataset_package.py",
            [
                "--output-dir",
                str(out),
                "--audit-manifest-path",
                str(audit_manifest),
            ],
        )
        assert result.returncode == 0, result.stderr or result.stdout
        assert Path(result.stdout.strip()) == out
        return out, audit_manifest

    @pytest.fixture(scope="class")
    def package_dir(self, build_output: tuple[Path, Path]) -> Path:
        return build_output[0]

    @pytest.fixture(scope="class")
    def audit_manifest_path(self, build_output: tuple[Path, Path]) -> Path:
        return build_output[1]

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
        assert meta["id"] == _CANONICAL_DATASET_METADATA["id"]

    def test_dataset_metadata_title(self, package_dir: Path):
        meta = json.loads((package_dir / "dataset-metadata.json").read_text(encoding="utf-8"))
        assert meta["title"] == _CANONICAL_DATASET_METADATA["title"]

    def test_dataset_metadata_license(self, package_dir: Path):
        meta = json.loads((package_dir / "dataset-metadata.json").read_text(encoding="utf-8"))
        assert meta["licenses"] == _CANONICAL_DATASET_METADATA["licenses"]

    def test_dataset_metadata_resources_non_empty(self, package_dir: Path):
        meta = json.loads((package_dir / "dataset-metadata.json").read_text(encoding="utf-8"))
        assert isinstance(meta["resources"], list) and len(meta["resources"]) > 0

    def test_dataset_metadata_resources_list_all_real_files(self, package_dir: Path):
        meta = json.loads((package_dir / "dataset-metadata.json").read_text(encoding="utf-8"))
        for entry in meta["resources"]:
            assert (package_dir / entry["path"]).is_file(), (
                f"resource {entry['path']!r} listed in dataset-metadata.json but not found"
            )

    def test_dataset_metadata_preserves_all_canonical_fields(self, package_dir: Path):
        meta = json.loads((package_dir / "dataset-metadata.json").read_text(encoding="utf-8"))
        for key, value in _CANONICAL_DATASET_METADATA.items():
            assert meta[key] == value
        assert set(meta) == set(_CANONICAL_DATASET_METADATA) | {"resources"}

    def test_dataset_metadata_self_not_in_resources(self, package_dir: Path):
        meta = json.loads((package_dir / "dataset-metadata.json").read_text(encoding="utf-8"))
        resource_paths = {e["path"] for e in meta["resources"]}
        assert "dataset-metadata.json" not in resource_paths

    def test_dataset_metadata_resources_are_sorted_and_unique(self, package_dir: Path):
        meta = json.loads((package_dir / "dataset-metadata.json").read_text(encoding="utf-8"))
        resource_paths = [e["path"] for e in meta["resources"]]
        assert resource_paths == sorted(resource_paths)
        assert len(resource_paths) == len(set(resource_paths))

    def test_dataset_metadata_resources_exactly_match_built_files(self, package_dir: Path):
        meta = json.loads((package_dir / "dataset-metadata.json").read_text(encoding="utf-8"))
        resource_paths = {e["path"] for e in meta["resources"]}
        built_files = _files_in(package_dir) - {"dataset-metadata.json"}
        assert resource_paths == built_files

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

    def test_package_dataset_metadata_contains_no_placeholders(self, package_dir: Path):
        """KAGGLE_USERNAME must not appear in the packaged dataset-metadata.json."""
        text = (package_dir / "dataset-metadata.json").read_text(encoding="utf-8")
        assert "KAGGLE_USERNAME" not in text

    def test_audit_manifest_captures_runtime_package(self, audit_manifest_path: Path, package_dir: Path):
        manifest = json.loads(audit_manifest_path.read_text(encoding="utf-8"))
        file_inventory_paths = [entry["path"] for entry in manifest["files"]]

        assert manifest["artifact_kind"] == "runtime_dataset_package"
        assert manifest["metadata_contract"] == "canonical_fields_plus_resources"
        assert manifest["canonical_metadata_sha256"] == _sha256(
            _REPO_ROOT / "packaging" / "kaggle" / "dataset-metadata.json"
        )
        assert manifest["packaged_metadata_sha256"] == _sha256(package_dir / "dataset-metadata.json")
        assert file_inventory_paths == sorted(file_inventory_paths)
        assert set(file_inventory_paths) == _files_in(package_dir)


# ---------------------------------------------------------------------------
# Kernel bundle
# ---------------------------------------------------------------------------


class TestKernelBundle:
    @pytest.fixture(scope="class")
    def build_output(self, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
        out = tmp_path_factory.mktemp("kernel-bundle")
        audit_manifest = tmp_path_factory.mktemp("kernel-bundle-audit") / "audit.json"
        result = _run(
            "scripts/cd/build_kernel_package.py",
            [
                "--output-dir",
                str(out),
                "--audit-manifest-path",
                str(audit_manifest),
            ],
        )
        assert result.returncode == 0, result.stderr or result.stdout
        assert Path(result.stdout.strip()) == out
        return out, audit_manifest

    @pytest.fixture(scope="class")
    def bundle_dir(self, build_output: tuple[Path, Path]) -> Path:
        return build_output[0]

    @pytest.fixture(scope="class")
    def audit_manifest_path(self, build_output: tuple[Path, Path]) -> Path:
        return build_output[1]

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

    def test_kernel_metadata_matches_canonical(self, bundle_dir: Path):
        built = json.loads((bundle_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
        assert built == _CANONICAL_KERNEL_METADATA

    def test_kernel_bundle_metadata_contains_no_placeholders(self, bundle_dir: Path):
        """KAGGLE_USERNAME must not appear in the packaged kernel-metadata.json."""
        text = (bundle_dir / "kernel-metadata.json").read_text(encoding="utf-8")
        assert "KAGGLE_USERNAME" not in text

    def test_kernel_metadata_id(self, bundle_dir: Path):
        meta = json.loads((bundle_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
        assert meta["id"] == _CANONICAL_KERNEL_METADATA["id"]

    def test_kernel_metadata_title_matches_canonical_metadata(self, bundle_dir: Path):
        meta = json.loads((bundle_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
        assert meta["title"] == _CANONICAL_KERNEL_METADATA["title"]

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

    def test_kernel_metadata_dataset_sources(self, bundle_dir: Path):
        meta = json.loads((bundle_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
        assert meta["dataset_sources"] == _CANONICAL_KERNEL_METADATA["dataset_sources"]

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

    def test_audit_manifest_captures_kernel_bundle(self, audit_manifest_path: Path, bundle_dir: Path):
        manifest = json.loads(audit_manifest_path.read_text(encoding="utf-8"))
        file_inventory_paths = [entry["path"] for entry in manifest["files"]]

        assert manifest["artifact_kind"] == "kernel_bundle"
        assert manifest["metadata_contract"] == "byte_identical_metadata_copy"
        assert manifest["canonical_metadata_sha256"] == _sha256(
            _REPO_ROOT / "packaging" / "kaggle" / "kernel-metadata.json"
        )
        assert manifest["packaged_metadata_sha256"] == _sha256(bundle_dir / "kernel-metadata.json")
        assert file_inventory_paths == sorted(file_inventory_paths)
        assert set(file_inventory_paths) == _files_in(bundle_dir)
