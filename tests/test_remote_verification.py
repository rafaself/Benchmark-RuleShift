import json
import hashlib
from pathlib import Path
import pytest

from core.kaggle import verify_remote_hashes, ArtifactResult

def get_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()

@pytest.fixture
def mock_env(tmp_path):
    # Setup mock kernel and dataset dirs
    kernel_dir = tmp_path / "kernel"
    dataset_dir = tmp_path / "dataset"
    kernel_dir.mkdir()
    dataset_dir.mkdir()
    
    notebook_content = b"notebook content"
    notebook_hash = get_hash(notebook_content)
    
    meta_content = b'{"id": "test/kernel"}'
    meta_hash = get_hash(meta_content)
    
    split_content = b'{"episodes": []}'
    split_hash = get_hash(split_content)
    
    manifest = {
        "entry_points": {
            "kbench_notebook": {
                "path": "packaging/kaggle/ruleshift_notebook_task.ipynb",
                "sha256": notebook_hash
            },
            "kernel_metadata": {
                "path": "packaging/kaggle/kernel-metadata.json",
                "sha256": meta_hash
            }
        },
        "frozen_split_manifests": {
            "dev": {
                "path": "src/frozen_splits/dev.json",
                "sha256": split_hash
            }
        }
    }
    
    return {
        "kernel_dir": kernel_dir,
        "dataset_dir": dataset_dir,
        "manifest": manifest,
        "notebook_content": notebook_content,
        "meta_content": meta_content,
        "split_content": split_content,
        "notebook_hash": notebook_hash,
        "meta_hash": meta_hash,
        "split_hash": split_hash,
    }

def setup_remote_files(env, notebook_content=None, meta_content=None, split_content=None):
    if notebook_content is not None:
        (env["kernel_dir"] / "ruleshift_notebook_task.ipynb").write_bytes(notebook_content)
    if meta_content is not None:
        (env["kernel_dir"] / "kernel-metadata.json").write_bytes(meta_content)
    if split_content is not None:
        s_path = env["dataset_dir"] / "src" / "frozen_splits" / "dev.json"
        s_path.parent.mkdir(parents=True, exist_ok=True)
        s_path.write_bytes(split_content)

def test_verify_remote_hashes_success(mock_env):
    setup_remote_files(
        mock_env,
        mock_env["notebook_content"],
        mock_env["meta_content"],
        mock_env["split_content"]
    )
    
    results = verify_remote_hashes(
        mock_env["manifest"], mock_env["kernel_dir"], mock_env["dataset_dir"]
    )
    
    for res in results:
        assert res.status == "MATCH"

def test_verify_remote_hashes_notebook_mismatch(mock_env):
    setup_remote_files(
        mock_env,
        b"WRONG notebook content",
        mock_env["meta_content"],
        mock_env["split_content"]
    )
    
    results = verify_remote_hashes(
        mock_env["manifest"], mock_env["kernel_dir"], mock_env["dataset_dir"]
    )
    
    res_notebook = next(r for r in results if r.name == "entry_points.kbench_notebook")
    assert res_notebook.status == "MISMATCH"
    assert res_notebook.remote_hash == get_hash(b"WRONG notebook content")

def test_verify_remote_hashes_dataset_file_mismatch(mock_env):
    setup_remote_files(
        mock_env,
        mock_env["notebook_content"],
        mock_env["meta_content"],
        b'{"episodes": ["corrupted"]}'
    )
    
    results = verify_remote_hashes(
        mock_env["manifest"], mock_env["kernel_dir"], mock_env["dataset_dir"]
    )
    
    res_split = next(r for r in results if r.name == "frozen_split_manifests.dev")
    assert res_split.status == "MISMATCH"

def test_verify_remote_hashes_missing_artifact(mock_env):
    setup_remote_files(
        mock_env,
        mock_env["notebook_content"],
        mock_env["meta_content"],
        None # Missing split
    )
    
    results = verify_remote_hashes(
        mock_env["manifest"], mock_env["kernel_dir"], mock_env["dataset_dir"]
    )
    
    res_split = next(r for r in results if r.name == "frozen_split_manifests.dev")
    assert res_split.status == "MISSING"

def test_verify_remote_hashes_metadata_only_agreement_not_sufficient(mock_env):
    # Regression test proving metadata-only agreement does not count as success when content differs.
    setup_remote_files(
        mock_env,
        b"WRONG notebook content",
        mock_env["meta_content"], # This matches
        mock_env["split_content"]
    )
    
    results = verify_remote_hashes(
        mock_env["manifest"], mock_env["kernel_dir"], mock_env["dataset_dir"]
    )
    
    res_notebook = next(r for r in results if r.name == "entry_points.kbench_notebook")
    res_meta = next(r for r in results if r.name == "entry_points.kernel_metadata")
    
    assert res_meta.status == "MATCH"
    assert res_notebook.status == "MISMATCH"
