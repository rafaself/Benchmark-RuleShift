#!/usr/bin/env python3
"""Validate Kaggle metadata for deploy artifacts.

Usage:
    validate_kaggle_metadata.py preflight   — full pre-publish validation + capture outputs
    validate_kaggle_metadata.py dataset     — verify dataset payload before publish
    validate_kaggle_metadata.py notebook    — verify notebook payload before publish
"""
import json
import os
import sys
from pathlib import Path


def preflight():
    errors = []
    placeholders = ("YOUR_USERNAME", "<username>", "owner-slug")

    runtime_dir = Path(os.environ["DEPLOY_RUNTIME_DIR"])
    notebook_dir = Path(os.environ["DEPLOY_NOTEBOOK_DIR"])
    expected_notebook = os.environ["EXPECTED_NOTEBOOK_FILE"]

    dm = json.loads((runtime_dir / "dataset-metadata.json").read_text())
    km = json.loads((notebook_dir / "kernel-metadata.json").read_text())

    dataset_id = dm.get("id", "")
    kernel_id = km.get("id", "")
    dataset_sources = km.get("dataset_sources", [])

    if "/" not in dataset_id:
        errors.append(f"dataset id is not fully qualified: {dataset_id!r}")
    if "/" not in kernel_id:
        errors.append(f"kernel id is not fully qualified: {kernel_id!r}")

    for marker in placeholders:
        if marker in dataset_id:
            errors.append(f"dataset id still contains placeholder: {dataset_id!r}")
        if marker in kernel_id:
            errors.append(f"kernel id still contains placeholder: {kernel_id!r}")

    for field in ("title", "id", "licenses"):
        if not dm.get(field):
            errors.append(f"dataset-metadata.json missing {field!r}")

    if km.get("kernel_type") != "notebook":
        errors.append(f"kernel_type={km.get('kernel_type')!r}, expected 'notebook'")
    if km.get("code_file") != expected_notebook:
        errors.append(f"code_file={km.get('code_file')!r}, expected {expected_notebook!r}")
    if not km.get("code_file"):
        errors.append("kernel-metadata.json missing 'code_file'")
    if not dataset_sources:
        errors.append("dataset_sources is empty")
    if dataset_sources and dataset_id not in dataset_sources:
        errors.append(
            f"kernel dataset_sources {dataset_sources!r} does not reference dataset id {dataset_id!r}"
        )

    notebook_path = notebook_dir / expected_notebook
    if not notebook_path.exists():
        errors.append(f"notebook file missing: {notebook_path}")

    if errors:
        print("Metadata validation FAILED:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        sys.exit(1)

    print("Metadata validation passed.")
    print(f"dataset id: {dataset_id}")
    print(f"kernel id: {kernel_id}")
    print(f"code_file: {km['code_file']}")
    print(f"dataset_sources: {dataset_sources}")

    # Capture outputs for GitHub Actions
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            print(f"dataset_id={dataset_id}", file=f)
            print(f"kernel_id={kernel_id}", file=f)
            print(f"notebook_file={km['code_file']}", file=f)


def dataset():
    runtime_dir = Path(os.environ["DEPLOY_RUNTIME_DIR"])
    dm = json.loads((runtime_dir / "dataset-metadata.json").read_text())

    assert "/" in dm["id"], dm
    assert dm.get("title"), dm
    assert dm.get("licenses"), dm
    assert (runtime_dir / "src").is_dir(), f"missing: {runtime_dir / 'src'}"
    assert (runtime_dir / "packaging" / "kaggle" / "frozen_artifacts_manifest.json").exists()

    print("Dataset payload verified.")
    print(f"Dataset id: {dm['id']}")


def notebook():
    notebook_dir = Path(os.environ["DEPLOY_NOTEBOOK_DIR"])
    km = json.loads((notebook_dir / "kernel-metadata.json").read_text())
    notebook_path = notebook_dir / km["code_file"]

    assert km["kernel_type"] == "notebook", km
    assert "/" in km["id"], km
    assert notebook_path.exists(), f"missing: {notebook_path}"
    assert km.get("dataset_sources"), km

    print("Notebook payload verified.")
    print(f"Kernel id: {km['id']}")
    print(f"Code file: {km['code_file']}")
    print(f"Dataset sources: {km['dataset_sources']}")


MODES = {"preflight": preflight, "dataset": dataset, "notebook": notebook}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in MODES:
        print(f"Usage: {sys.argv[0]} {{{','.join(MODES)}}}", file=sys.stderr)
        sys.exit(2)
    MODES[sys.argv[1]]()
