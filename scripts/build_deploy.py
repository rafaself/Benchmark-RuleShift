#!/usr/bin/env python3
"""Build Kaggle deploy artifacts from the canonical repo state.

Generates:
  deploy/kaggle-notebook/   -- official submission notebook copy
  deploy/kaggle-runtime/    -- packaged dataset upload folder

This script is the single authoritative source of truth for what goes
into deploy/.  Never edit deploy/ by hand.

Usage:
    python scripts/build_deploy.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from check_public_private_isolation import _collect_public_location_errors

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
KAGGLE_DIR = REPO_ROOT / "packaging" / "kaggle"
DEPLOY_DIR = REPO_ROOT / "deploy"

NOTEBOOK_SRC = KAGGLE_DIR / "ruleshift_notebook_task.ipynb"
KERNEL_METADATA_SRC = KAGGLE_DIR / "kernel-metadata.json"
FROZEN_ARTIFACTS_MANIFEST_SRC = KAGGLE_DIR / "frozen_artifacts_manifest.json"

DEPLOY_NOTEBOOK_DIR = DEPLOY_DIR / "kaggle-notebook"
DEPLOY_RUNTIME_DIR = DEPLOY_DIR / "kaggle-runtime"


def _verify_sources() -> None:
    missing = []
    for path in (NOTEBOOK_SRC, KERNEL_METADATA_SRC, FROZEN_ARTIFACTS_MANIFEST_SRC, SRC_DIR):
        if not path.exists():
            missing.append(str(path))
    if missing:
        sys.exit(f"ERROR: missing source paths:\n" + "\n".join(f"  {p}" for p in missing))


def _verify_kernel_metadata() -> None:
    metadata = json.loads(KERNEL_METADATA_SRC.read_text(encoding="utf-8"))
    expected_code_file = NOTEBOOK_SRC.name
    if metadata.get("code_file") != expected_code_file:
        sys.exit(
            f"ERROR: kernel-metadata.json code_file={metadata.get('code_file')!r} "
            f"does not match notebook filename {expected_code_file!r}"
        )


_PRIVATE_ONLY_FILENAMES: tuple[str, ...] = (
    "private_leaderboard.json",
    "private_episodes.json",
)
_PUBLIC_RUNTIME_SPLITS: tuple[str, ...] = (
    "dev",
    "public_leaderboard",
)


def _verify_no_private_in_runtime() -> None:
    """Fail the build if any private-only asset appears in deploy/kaggle-runtime/."""
    violations = []
    for name in _PRIVATE_ONLY_FILENAMES:
        hits = list(DEPLOY_RUNTIME_DIR.rglob(name))
        violations.extend(hits)
    if violations:
        sys.exit(
            "ERROR: private-only asset(s) found in deploy/kaggle-runtime/ — "
            "private data must not enter the public runtime package:\n"
            + "\n".join(f"  {p.relative_to(REPO_ROOT)}" for p in violations)
        )


def _verify_public_runtime_split_whitelist() -> None:
    frozen_splits_dir = DEPLOY_RUNTIME_DIR / "src" / "frozen_splits"
    if not frozen_splits_dir.exists():
        sys.exit(
            "ERROR: deploy/kaggle-runtime/src/frozen_splits/ is missing from the public runtime package"
        )

    shipped = {
        path.name
        for path in frozen_splits_dir.iterdir()
        if path.is_file()
    }
    allowed = {f"{split_name}.json" for split_name in _PUBLIC_RUNTIME_SPLITS}
    if shipped != allowed:
        sys.exit(
            "ERROR: deploy/kaggle-runtime/src/frozen_splits/ must contain only the public split manifests:\n"
            f"  expected: {sorted(allowed)}\n"
            f"  observed: {sorted(shipped)}"
        )


def _verify_public_repo_isolation() -> None:
    violations = _collect_public_location_errors()
    if violations:
        sys.exit(
            "ERROR: public repo still contains private-only artifact references:\n"
            + "\n".join(f"  {violation}" for violation in violations)
        )


def _verify_dataset_metadata() -> None:
    meta_path = DEPLOY_RUNTIME_DIR / "dataset-metadata.json"
    if not meta_path.exists():
        sys.exit(f"ERROR: dataset-metadata.json not found at {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not meta.get("title"):
        sys.exit("ERROR: dataset-metadata.json missing 'title'")
    if not meta.get("id"):
        sys.exit("ERROR: dataset-metadata.json missing 'id'")
    if not meta.get("licenses"):
        sys.exit("ERROR: dataset-metadata.json missing 'licenses'")


def build_kaggle_notebook() -> None:
    """Copy the official notebook and kernel metadata into deploy/kaggle-notebook/."""
    if DEPLOY_NOTEBOOK_DIR.exists():
        shutil.rmtree(DEPLOY_NOTEBOOK_DIR)
    DEPLOY_NOTEBOOK_DIR.mkdir(parents=True)
    shutil.copy2(NOTEBOOK_SRC, DEPLOY_NOTEBOOK_DIR / NOTEBOOK_SRC.name)
    print(f"  notebook -> {DEPLOY_NOTEBOOK_DIR / NOTEBOOK_SRC.name}")
    shutil.copy2(KERNEL_METADATA_SRC, DEPLOY_NOTEBOOK_DIR / KERNEL_METADATA_SRC.name)
    print(f"  kernel-metadata.json -> {DEPLOY_NOTEBOOK_DIR / KERNEL_METADATA_SRC.name}")


def build_kaggle_runtime() -> None:
    """Assemble the runtime dataset folder in deploy/kaggle-runtime/.

    The runtime folder mirrors the subset the notebook expects to find
    under /kaggle/input/ruleshift-runtime/ on Kaggle:

      src/                               -- all top-level .py compatibility wrappers
      src/core/                          -- core infrastructure
      src/tasks/ruleshift_benchmark/     -- task-specific logic
      src/frozen_splits/                 -- frozen episode manifests
      packaging/kaggle/frozen_artifacts_manifest.json

    dataset-metadata.json is written from the canonical template on every
    build.
    """
    if DEPLOY_RUNTIME_DIR.exists():
        shutil.rmtree(DEPLOY_RUNTIME_DIR)
    DEPLOY_RUNTIME_DIR.mkdir(parents=True)

    # Write dataset-metadata.json from the canonical template.
    meta_path = DEPLOY_RUNTIME_DIR / "dataset-metadata.json"
    template = {
        "title": "RuleShift Runtime",
        "id": "raptorengineer/ruleshift-runtime",
        "licenses": [{"name": "CC0-1.0"}],
    }
    meta_path.write_text(json.dumps(template, indent=2) + "\n", encoding="utf-8")
    print(f"  dataset-metadata.json")

    # Copy src/ (subset needed by the Kaggle notebook).
    runtime_src = DEPLOY_RUNTIME_DIR / "src"
    runtime_src.mkdir()

    # Top-level compatibility wrappers (*.py in src/).
    for py in sorted(SRC_DIR.glob("*.py")):
        shutil.copy2(py, runtime_src / py.name)
        print(f"  src/{py.name}")

    # src/core/
    shutil.copytree(SRC_DIR / "core", runtime_src / "core", ignore=shutil.ignore_patterns("__pycache__"))
    print(f"  src/core/")

    # src/tasks/
    tasks_dst = runtime_src / "tasks"
    tasks_dst.mkdir()
    (tasks_dst / "__init__.py").write_text("", encoding="utf-8")
    shutil.copytree(
        SRC_DIR / "tasks" / "ruleshift_benchmark",
        tasks_dst / "ruleshift_benchmark",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    print(f"  src/tasks/ruleshift_benchmark/")

    # src/frozen_splits/ -- dev and public_leaderboard only; private split is not shipped
    frozen_splits_dst = runtime_src / "frozen_splits"
    frozen_splits_dst.mkdir()
    for split_name in _PUBLIC_RUNTIME_SPLITS:
        src_file = SRC_DIR / "frozen_splits" / f"{split_name}.json"
        shutil.copy2(src_file, frozen_splits_dst / src_file.name)
        print(f"  src/frozen_splits/{src_file.name}")
    print(f"  src/frozen_splits/")

    # packaging/kaggle/frozen_artifacts_manifest.json
    pkg_kaggle_dst = DEPLOY_RUNTIME_DIR / "packaging" / "kaggle"
    pkg_kaggle_dst.mkdir(parents=True)
    shutil.copy2(FROZEN_ARTIFACTS_MANIFEST_SRC, pkg_kaggle_dst / FROZEN_ARTIFACTS_MANIFEST_SRC.name)
    print(f"  packaging/kaggle/frozen_artifacts_manifest.json")

    # CONTENTS.md
    contents = (
        "# RuleShift Runtime Dataset Contents\n\n"
        "Expected Kaggle dataset root:\n"
        "`/kaggle/input/ruleshift-runtime/`\n\n"
        "Copied items:\n\n"
        "- `src/`\n"
        "  Required because the official notebook inserts "
        "`/kaggle/input/ruleshift-runtime/src` into `sys.path` and imports the "
        "runtime modules from there.\n"
        "- `src/core/`\n"
        "  Required because `src/kaggle.py`, `src/splits.py`, `src/parser.py`, "
        "and `src/metrics.py` re-export implementations from `core.*`.\n"
        "- `src/tasks/ruleshift_benchmark/`\n"
        "  Required because the runtime scoring, rendering, schema, protocol, "
        "baseline, and generator logic import task modules from this package.\n"
        "- `src/frozen_splits/`\n"
        "  Required because the notebook loads the frozen split manifests from "
        "`src/frozen_splits/*.json`.\n"
        "- `packaging/kaggle/frozen_artifacts_manifest.json`\n"
        "  Required because the notebook locates the attached dataset root by "
        "finding this file under `packaging/kaggle/`.\n\n"
        "Packaging notes:\n\n"
        "- This deploy folder intentionally contains only copied runtime files "
        "plus Kaggle dataset metadata.\n"
        "- The official notebook is expected to run separately with this dataset "
        "attached at `/kaggle/input/ruleshift-runtime/`.\n"
        "- `validate_kaggle_staging_manifest()` is not expected to pass against "
        "this dataset-only package because the manifest still references the "
        "official notebook and `kernel-metadata.json`, which are intentionally "
        "not copied here.\n"
    )
    (DEPLOY_RUNTIME_DIR / "CONTENTS.md").write_text(contents, encoding="utf-8")
    print(f"  CONTENTS.md")


def main() -> None:
    print("Verifying sources...")
    _verify_sources()
    _verify_kernel_metadata()
    _verify_public_repo_isolation()

    print("\nBuilding deploy/kaggle-notebook/...")
    build_kaggle_notebook()

    print("\nBuilding deploy/kaggle-runtime/...")
    build_kaggle_runtime()

    print("\nVerifying deploy/kaggle-runtime/dataset-metadata.json...")
    _verify_dataset_metadata()

    print("\nVerifying no private-only assets in deploy/kaggle-runtime/...")
    _verify_no_private_in_runtime()
    _verify_public_runtime_split_whitelist()

    print("\nDone.")
    print(f"  {DEPLOY_NOTEBOOK_DIR}")
    print(f"  {DEPLOY_RUNTIME_DIR}")


if __name__ == "__main__":
    main()
