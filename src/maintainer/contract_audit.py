"""P0 contract audit — validates public Kaggle artifact contract end-to-end.

Checks for drift between:
  1. Implemented benchmark state (code and frozen assets)
  2. Official Kaggle notebook + metadata
  3. Serialized task definition (materialized from notebook)
  4. Serialized run artifact (artifact.json from canonical benchmark runs)
  5. Canonical manifest / artifact hashes
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Final

from core.kaggle import load_kaggle_staging_manifest, validate_kaggle_staging_manifest
from core.splits import PARTITIONS

__all__ = [
    "CANONICAL_BINARY_TASK_NAME",
    "CANONICAL_NARRATIVE_TASK_NAME",
    "CANONICAL_NOTEBOOK_FILENAME",
    "CANONICAL_NOTEBOOK_RELPATH",
    "CANONICAL_TASK_ID",
    "CANONICAL_TASK_NAME",
    "EXPECTED_DATASET_SOURCES",
    "EXPECTED_EPISODES_PER_SPLIT",
    "EXPECTED_PROBE_COUNT",
    "EXPECTED_SPLITS",
    "check_manifest_hashes",
    "check_notebook_metadata",
    "check_run_artifact",
    "check_split_episode_counts",
    "check_task_artifact",
    "find_latest_run_artifact",
    "is_known_bad_run_shape",
    "materialize_task_definition",
    "run_contract_audit",
]

# ── canonical contract values ────────────────────────────────────

CANONICAL_NOTEBOOK_RELPATH: Final[str] = "packaging/kaggle/ruleshift_notebook_task.ipynb"
CANONICAL_NOTEBOOK_FILENAME: Final[str] = "ruleshift_notebook_task.ipynb"
CANONICAL_KERNEL_METADATA_RELPATH: Final[str] = "packaging/kaggle/kernel-metadata.json"
CANONICAL_TASK_ID: Final[str] = "ruleshift_benchmark_v1"
CANONICAL_TASK_NAME: Final[str] = "RuleShift Benchmark v1"
CANONICAL_BINARY_TASK_NAME: Final[str] = "ruleshift_benchmark_v1_binary"
CANONICAL_NARRATIVE_TASK_NAME: Final[str] = "ruleshift_benchmark_v1_narrative"
_KERNEL_METADATA_PATH: Final[Path] = (
    Path(__file__).resolve().parents[2] / "packaging" / "kaggle" / "kernel-metadata.json"
)
EXPECTED_DATASET_SOURCES: Final[tuple[str, ...]] = tuple(
    json.loads(_KERNEL_METADATA_PATH.read_text(encoding="utf-8"))["dataset_sources"]
)
EXPECTED_SPLITS: Final[tuple[str, ...]] = PARTITIONS
EXPECTED_EPISODES_PER_SPLIT: Final[int] = 18
EXPECTED_PROBE_COUNT: Final[int] = 4


def _repo_root(repo_root: Path | str | None = None) -> Path:
    if repo_root is None:
        return Path(__file__).resolve().parents[2]
    return Path(repo_root).resolve()


# ── 1. Notebook / metadata ──────────────────────────────────────


def check_notebook_metadata(repo_root: Path | str | None = None) -> list[str]:
    """Validate canonical notebook path and kernel-metadata.json contract.

    Returns a list of error strings; empty means the check passed.
    """
    root = _repo_root(repo_root)
    errors: list[str] = []

    notebook_path = root / CANONICAL_NOTEBOOK_RELPATH
    if not notebook_path.is_file():
        errors.append(f"canonical notebook missing: {CANONICAL_NOTEBOOK_RELPATH}")

    metadata_path = root / CANONICAL_KERNEL_METADATA_RELPATH
    if not metadata_path.is_file():
        errors.append(f"kernel-metadata.json missing: {CANONICAL_KERNEL_METADATA_RELPATH}")
        return errors

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    code_file = metadata.get("code_file")
    if code_file != CANONICAL_NOTEBOOK_FILENAME:
        errors.append(
            f"kernel-metadata.json code_file must be {CANONICAL_NOTEBOOK_FILENAME!r}, "
            f"got {code_file!r}"
        )

    dataset_sources = metadata.get("dataset_sources", [])
    for expected in EXPECTED_DATASET_SOURCES:
        if expected not in dataset_sources:
            errors.append(f"kernel-metadata.json dataset_sources missing {expected!r}")

    return errors


# ── 2. Task artifact ────────────────────────────────────────────

_TASK_PATTERN = re.compile(
    r'@kbench\.task\(\s*name="([^"]+)",\s*description=\((.*?)\),?\s*\)',
    re.DOTALL,
)
_CHOOSE_PATTERN = re.compile(r'^%choose\s+(\S+)', re.MULTILINE)
# Detect the Narrative companion as a plain function (not a @kbench.task).
_NARRATIVE_FN_PATTERN = re.compile(
    r'def\s+ruleshift_benchmark_v1_narrative\s*\('
)


def materialize_task_definition(repo_root: Path | str | None = None) -> dict[str, Any]:
    """Extract and serialize the task definition from the official notebook.

    Reads the notebook source, extracts ``@kbench.task`` registrations and
    the ``%choose`` directive, then enriches with manifest metadata to
    produce the canonical task.json shape.
    """
    root = _repo_root(repo_root)
    notebook_path = root / CANONICAL_NOTEBOOK_RELPATH
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    sources = "\n".join(
        "".join(cell.get("source", ())) for cell in notebook["cells"]
    )

    tasks: list[dict[str, str]] = []
    for match in _TASK_PATTERN.finditer(sources):
        name = match.group(1)
        desc_raw = match.group(2)
        desc_parts = re.findall(r'"([^"]*)"', desc_raw)
        description = "".join(desc_parts)
        role = "leaderboard_primary" if "binary" in name else "companion"
        tasks.append({"name": name, "description": description, "role": role})

    # Narrative is intentionally a plain function (not @kbench.task) but is
    # still a contract companion.  Detect it via its function definition.
    registered_names = {t["name"] for t in tasks}
    if (
        CANONICAL_NARRATIVE_TASK_NAME not in registered_names
        and _NARRATIVE_FN_PATTERN.search(sources)
    ):
        tasks.append({
            "name": CANONICAL_NARRATIVE_TASK_NAME,
            "description": (
                "Narrative companion for RuleShift Benchmark v1. Same "
                "episodes as Binary, natural-language prompt format. "
                "Robustness evidence only — not leaderboard-scored."
            ),
            "role": "companion",
        })

    choose_match = _CHOOSE_PATTERN.search(sources)
    chosen_task = choose_match.group(1) if choose_match else None

    manifest = load_kaggle_staging_manifest(root)

    return {
        "task_id": manifest.get("task_id", ""),
        "task_name": manifest.get("task_name", ""),
        "tasks": tasks,
        "chosen_task": chosen_task,
        "splits": list(EXPECTED_SPLITS),
        "episodes_per_split": EXPECTED_EPISODES_PER_SPLIT,
        "total_episodes": EXPECTED_EPISODES_PER_SPLIT * len(EXPECTED_SPLITS),
        "probe_count": EXPECTED_PROBE_COUNT,
    }


def check_task_artifact(task_json: dict[str, Any]) -> list[str]:
    """Validate a materialized task definition against the benchmark contract."""
    errors: list[str] = []

    if task_json.get("task_id") != CANONICAL_TASK_ID:
        errors.append(
            f"task_id must be {CANONICAL_TASK_ID!r}, "
            f"got {task_json.get('task_id')!r}"
        )

    if task_json.get("task_name") != CANONICAL_TASK_NAME:
        errors.append(
            f"task_name must be {CANONICAL_TASK_NAME!r}, "
            f"got {task_json.get('task_name')!r}"
        )

    task_names = {t["name"] for t in task_json.get("tasks", [])}
    if CANONICAL_BINARY_TASK_NAME not in task_names:
        errors.append(f"missing leaderboard-primary task: {CANONICAL_BINARY_TASK_NAME}")
    if CANONICAL_NARRATIVE_TASK_NAME not in task_names:
        errors.append(f"missing companion task: {CANONICAL_NARRATIVE_TASK_NAME}")

    for t in task_json.get("tasks", []):
        if not t.get("description"):
            errors.append(f"task {t['name']!r} has empty description")

    if task_json.get("chosen_task") != CANONICAL_BINARY_TASK_NAME:
        errors.append(
            f"chosen_task must be {CANONICAL_BINARY_TASK_NAME!r}, "
            f"got {task_json.get('chosen_task')!r}"
        )

    if tuple(task_json.get("splits", ())) != EXPECTED_SPLITS:
        errors.append(f"splits must be {list(EXPECTED_SPLITS)}")

    if task_json.get("probe_count") != EXPECTED_PROBE_COUNT:
        errors.append(f"probe_count must be {EXPECTED_PROBE_COUNT}")

    if task_json.get("episodes_per_split") != EXPECTED_EPISODES_PER_SPLIT:
        errors.append(
            f"episodes_per_split must be {EXPECTED_EPISODES_PER_SPLIT}, "
            f"got {task_json.get('episodes_per_split')!r}"
        )

    return errors


# ── 3. Run artifact ─────────────────────────────────────────────


def is_known_bad_run_shape(run_json: dict[str, Any]) -> bool:
    """Detect the exact known bad run shape:

    - one conversation
    - conversations[].metrics == {}
    - aggregated results[].numericResult contains only confidenceInterval
    """
    conversations = run_json.get("conversations")
    results = run_json.get("results")

    if not isinstance(conversations, list) or not isinstance(results, list):
        return False

    if len(conversations) != 1:
        return False
    conv = conversations[0]
    if not isinstance(conv, dict) or conv.get("metrics") != {}:
        return False

    if not results:
        return False
    for result in results:
        if not isinstance(result, dict):
            return False
        nr = result.get("numericResult")
        if not isinstance(nr, dict):
            return False
        if set(nr.keys()) - {"confidenceInterval"}:
            return False

    return True


def check_run_artifact(run_json: dict[str, Any]) -> list[str]:
    """Validate a serialized run artifact against the benchmark contract.

    Handles both the canonical benchmark artifact format (``splits``/``rows``)
    and a kbench-style format (``conversations``/``results``).
    """
    errors: list[str] = []

    # ── known bad shape (kbench format) ──────────────────────────
    if is_known_bad_run_shape(run_json):
        errors.append(
            "known bad run shape: one conversation, empty metrics, "
            "only confidenceInterval in numericResult"
        )

    # ── local artifact format checks ─────────────────────────────
    if "splits" in run_json:
        splits = run_json["splits"]
        if isinstance(splits, list):
            total_episodes = sum(
                s.get("episode_count", 0) for s in splits if isinstance(s, dict)
            )
            if total_episodes <= 1:
                errors.append(
                    "run has at most 1 episode across all splits — insufficient evidence"
                )

    # ── Narrative companion evidence ─────────────────────────────
    prompt_modes = run_json.get("prompt_modes", [])
    if isinstance(prompt_modes, list) and "narrative" not in prompt_modes:
        errors.append(
            "missing Narrative companion evidence: prompt_modes does not include 'narrative'"
        )

    # ── aggregated result evidence ───────────────────────────────
    has_aggregated = False
    if run_json.get("diagnostic_summary"):
        has_aggregated = True
    if run_json.get("execution_summary"):
        has_aggregated = True
    if run_json.get("results"):
        results = run_json["results"]
        if isinstance(results, list) and results:
            for r in results:
                if isinstance(r, dict):
                    nr = r.get("numericResult", {})
                    if isinstance(nr, dict) and set(nr.keys()) - {"confidenceInterval"}:
                        has_aggregated = True
                        break

    if not has_aggregated:
        errors.append(
            "missing required aggregated result evidence: "
            "no diagnostic_summary, execution_summary, or non-trivial results"
        )

    return errors


# ── 4. Manifest / hashes ────────────────────────────────────────


def check_manifest_hashes(repo_root: Path | str | None = None) -> list[str]:
    """Validate canonical artifact references and SHA-256 hashes."""
    root = _repo_root(repo_root)
    errors: list[str] = []
    try:
        validate_kaggle_staging_manifest(root)
    except (ValueError, TypeError, FileNotFoundError) as exc:
        errors.append(f"manifest validation failed: {exc}")
    return errors


def check_split_episode_counts(repo_root: Path | str | None = None) -> list[str]:
    """Validate that EXPECTED_EPISODES_PER_SPLIT matches actual frozen manifests.

    This cross-check prevents contract drift where the constant is updated
    without updating the frozen split seed banks, or vice versa.
    """
    root = _repo_root(repo_root)
    from core.splits import PUBLIC_PARTITIONS, load_split_manifest

    errors: list[str] = []
    for partition in PUBLIC_PARTITIONS:
        try:
            manifest = load_split_manifest(partition, repo_root=root)
        except (FileNotFoundError, OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            errors.append(f"{partition} manifest could not be loaded: {exc}")
            continue
        actual = len(manifest.seeds)
        if actual != EXPECTED_EPISODES_PER_SPLIT:
            errors.append(
                f"{partition} manifest has {actual} seeds but "
                f"EXPECTED_EPISODES_PER_SPLIT is {EXPECTED_EPISODES_PER_SPLIT}"
            )
    return errors


# ── artifact finder ──────────────────────────────────────────────


def find_latest_run_artifact(repo_root: Path | str | None = None) -> Path | None:
    """Find the most recently modified run ``latest/artifact.json`` under reports/.

    Filters out comparison artifacts and other non-run schemas by requiring
    the ``artifact_schema_version`` key.  Prefers artifacts that include
    Narrative companion evidence (the contract requires it), falling back
    to binary-only if necessary.
    """
    root = _repo_root(repo_root)
    reports_dir = root / "reports"
    if not reports_dir.is_dir():
        return None
    candidates = sorted(
        reports_dir.rglob("latest/artifact.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    with_narrative: list[Path] = []
    without_narrative: list[Path] = []
    for candidate in candidates:
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
            if "artifact_schema_version" not in data:
                continue
            if "narrative" in data.get("prompt_modes", []):
                with_narrative.append(candidate)
            else:
                without_narrative.append(candidate)
        except (json.JSONDecodeError, OSError):
            continue
    if with_narrative:
        return with_narrative[0]
    if without_narrative:
        return without_narrative[0]
    return None


# ── orchestrator ─────────────────────────────────────────────────


def run_contract_audit(
    repo_root: Path | str | None = None,
    *,
    run_artifact_path: Path | str | None = None,
) -> dict[str, Any]:
    """Run the full P0 contract audit.

    Returns a structured report with per-section pass/fail and error details.
    """
    root = _repo_root(repo_root)

    # 1. Notebook / metadata
    notebook_errors = check_notebook_metadata(root)

    # 2. Task artifact
    task_errors: list[str] = []
    task_json: dict[str, Any] | None = None
    try:
        task_json = materialize_task_definition(root)
        task_errors = check_task_artifact(task_json)
    except Exception as exc:
        task_errors.append(f"task materialization failed: {exc}")

    # 3. Run artifact
    run_errors: list[str] = []
    resolved_run_path: str | None = None

    run_path_to_try: Path | None = None
    if run_artifact_path is not None:
        run_path_to_try = Path(run_artifact_path)
    else:
        run_path_to_try = find_latest_run_artifact(root)

    if run_path_to_try is None:
        pass
    elif not run_path_to_try.is_file():
        run_errors.append(f"run artifact not found: {run_path_to_try}")
    else:
        resolved_run_path = str(run_path_to_try)
        run_json_data = json.loads(run_path_to_try.read_text(encoding="utf-8"))
        run_errors = check_run_artifact(run_json_data)

    # 4. Manifest / hashes
    manifest_errors = check_manifest_hashes(root)

    # 5. Split episode counts vs constant
    split_count_errors = check_split_episode_counts(root)

    all_errors = notebook_errors + task_errors + run_errors + manifest_errors + split_count_errors

    return {
        "audit": "p0_contract",
        "passed": len(all_errors) == 0,
        "checks": {
            "notebook_metadata": {
                "passed": len(notebook_errors) == 0,
                "errors": notebook_errors,
            },
            "task_artifact": {
                "passed": len(task_errors) == 0,
                "task_json": task_json,
                "errors": task_errors,
            },
            "run_artifact": {
                "passed": len(run_errors) == 0,
                "run_artifact_path": resolved_run_path,
                "errors": run_errors,
            },
            "manifest_hashes": {
                "passed": len(manifest_errors) == 0,
                "errors": manifest_errors,
            },
            "split_episode_counts": {
                "passed": len(split_count_errors) == 0,
                "errors": split_count_errors,
            },
        },
        "error_count": len(all_errors),
        "errors": all_errors,
    }
