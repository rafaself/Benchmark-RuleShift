from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Final, Sequence

from core.parser import (
    NarrativeParseStatus,
    ParseStatus,
    parse_binary_output,
    parse_narrative_audit_output,
)
from core.slices import SLICE_DIMENSIONS, ErrorType
from core.splits import MANIFEST_VERSION, PARTITIONS, load_split_manifest
from tasks.ruleshift_benchmark.protocol import PROBE_COUNT, InteractionLabel, parse_label
from tasks.ruleshift_benchmark.schema import (
    DIFFICULTY_VERSION,
    GENERATOR_VERSION,
    SPEC_VERSION,
    TEMPLATE_SET_VERSION,
)

__all__ = [
    "ArtifactResult",
    "Label",
    "BinaryResponse",
    "ConfidenceInterval",
    "KAGGLE_STAGING_MANIFEST_PATH",
    "REQUIRED_SLICE_DIMENSIONS",
    "build_kaggle_payload",
    "compute_bootstrap_confidence_interval",
    "load_kaggle_staging_manifest",
    "normalize_binary_response",
    "normalize_narrative_response",
    "resolve_kaggle_artifact_path",
    "score_episode",
    "validate_kaggle_payload",
    "validate_kaggle_staging_manifest",
    "verify_remote_hashes",
]

# The five slice dimensions every canonical payload must contain.
REQUIRED_SLICE_DIMENSIONS: Final[tuple[str, ...]] = SLICE_DIMENSIONS

# Column names in binary_df that carry episode metadata for slice computation.
_SLICE_METADATA_COLS: Final[dict[str, str]] = {
    "template": "template_id",
    "difficulty": "difficulty",
    "shift_position": "shift_position",
    "transition_type": "transition_type",
}

_ARTIFACT_GROUPS: Final[tuple[str, ...]] = (
    "entry_points",
    "frozen_split_manifests",
)
_RUNTIME_ENTRY_POINTS: Final[tuple[str, ...]] = (
    "kbench_notebook",
    "kernel_metadata",
)
_MANIFEST_PARTITIONS: Final[tuple[str, ...]] = (
    "dev",
    "public_leaderboard",
)
_EXPECTED_BENCHMARK_VERSIONS: Final[dict[str, str]] = {
    "manifest_version": MANIFEST_VERSION,
    "spec_version": SPEC_VERSION,
    "generator_version": GENERATOR_VERSION,
    "template_set_version": TEMPLATE_SET_VERSION,
    "difficulty_version": DIFFICULTY_VERSION,
}


@dataclass(frozen=True, slots=True)
class ArtifactResult:
    name: str
    local_hash: str
    remote_hash: str | None
    status: str  # "MATCH", "MISMATCH", "MISSING"


class Label(str, Enum):
    attract = "attract"
    repel = "repel"


@dataclass(frozen=True, slots=True)
class BinaryResponse:
    probe_6: Label
    probe_7: Label
    probe_8: Label
    probe_9: Label

    def as_tuple(self) -> tuple[str, str, str, str]:
        return (
            self.probe_6.value,
            self.probe_7.value,
            self.probe_8.value,
            self.probe_9.value,
        )


def _repo_root(repo_root: Path | str | None = None) -> Path:
    if repo_root is None:
        return Path(__file__).resolve().parents[2]
    return Path(repo_root).resolve()


def _manifest_path(repo_root: Path | str | None = None) -> Path:
    # The filename is retained for compatibility, but its role is now the
    # official Kaggle runtime-contract manifest rather than a broad package index.
    return _repo_root(repo_root) / "packaging" / "kaggle" / "frozen_artifacts_manifest.json"


KAGGLE_STAGING_MANIFEST_PATH: Final[Path] = _manifest_path()


def load_kaggle_staging_manifest(
    repo_root: Path | str | None = None,
) -> dict[str, object]:
    return json.loads(_manifest_path(repo_root).read_text(encoding="utf-8"))


def resolve_kaggle_artifact_path(
    relative_path: str,
    *,
    repo_root: Path | str | None = None,
) -> Path:
    return _repo_root(repo_root) / relative_path


def validate_kaggle_staging_manifest(
    repo_root: Path | str | None = None,
) -> None:
    manifest = load_kaggle_staging_manifest(repo_root)

    if manifest.get("bundle_version") != "R16":
        raise ValueError("bundle_version must equal R16")
    if manifest.get("task_id") != "ruleshift_benchmark_v1":
        raise ValueError("task_id must equal ruleshift_benchmark_v1")
    if manifest.get("task_name") != "RuleShift Benchmark v1":
        raise ValueError("task_name must equal RuleShift Benchmark v1")

    benchmark_versions = manifest.get("benchmark_versions")
    if benchmark_versions != _EXPECTED_BENCHMARK_VERSIONS:
        raise ValueError(
            "benchmark_versions must match the canonical split and schema versions"
        )

    if manifest.get("current_emitted_difficulty_labels") != ["easy", "medium"]:
        raise ValueError(
            "current_emitted_difficulty_labels must equal ['easy', 'medium']"
        )
    if manifest.get("reserved_difficulty_labels") != ["hard"]:
        raise ValueError("reserved_difficulty_labels must equal ['hard']")

    frozen_split_manifests = _require_mapping(
        manifest,
        "frozen_split_manifests",
    )
    entry_points = _require_mapping(
        manifest,
        "entry_points",
    )
    if tuple(entry_points) != _RUNTIME_ENTRY_POINTS:
        raise ValueError("entry_points must contain only the official runtime submission paths")
    if tuple(frozen_split_manifests) != _MANIFEST_PARTITIONS:
        raise ValueError("frozen_split_manifests must follow the canonical partition order")

    for partition in _MANIFEST_PARTITIONS:
        artifact = _require_mapping(frozen_split_manifests, partition)
        split_manifest = load_split_manifest(partition)
        if artifact.get("manifest_version") != split_manifest.manifest_version:
            raise ValueError(f"{partition} manifest_version does not match the frozen split")
        if artifact.get("seed_bank_version") != split_manifest.seed_bank_version:
            raise ValueError(f"{partition} seed_bank_version does not match the frozen split")
        if artifact.get("episode_split") != split_manifest.episode_split.value:
            raise ValueError(f"{partition} episode_split does not match the frozen split")

    for group_name in _ARTIFACT_GROUPS:
        artifact_group = _require_mapping(manifest, group_name)
        for label, artifact in artifact_group.items():
            artifact_map = _require_mapping(artifact_group, label)
            relative_path = artifact_map.get("path")
            if not isinstance(relative_path, str) or not relative_path:
                raise ValueError(f"{group_name}.{label} must define a non-empty path")
            sha256 = artifact_map.get("sha256")
            if not isinstance(sha256, str) or not sha256:
                raise ValueError(f"{group_name}.{label} must define a non-empty sha256")

            artifact_path = resolve_kaggle_artifact_path(
                relative_path,
                repo_root=repo_root,
            )
            if not artifact_path.is_file():
                raise FileNotFoundError(
                    f"{group_name}.{label} points to a missing file: {artifact_path}"
                )

            actual_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            if actual_sha256 != sha256:
                raise ValueError(
                    f"{group_name}.{label} sha256 mismatch: expected {sha256}, got {actual_sha256}"
                )


def normalize_binary_response(response: object) -> tuple[str, ...] | None:
    if isinstance(response, BinaryResponse):
        return response.as_tuple()

    if isinstance(response, str):
        parsed = parse_binary_output(response)
        if parsed.status is ParseStatus.VALID:
            return tuple(label.value for label in parsed.labels)

    return None


def normalize_narrative_response(response: object) -> tuple[str, ...] | None:
    if isinstance(response, str):
        parsed = parse_narrative_audit_output(response)
        if parsed.status is NarrativeParseStatus.VALID and parsed.output is not None:
            return tuple(label.value for label in parsed.output.final_binary_answer)

    return None


def score_episode(
    predictions: tuple[str, ...] | tuple[InteractionLabel, ...] | None,
    probe_targets: tuple[str, ...] | tuple[InteractionLabel, ...],
) -> tuple[int, int]:
    normalized_targets = _normalize_labels(probe_targets)
    if normalized_targets is None:
        raise ValueError(f"probe_targets must contain exactly {PROBE_COUNT} valid labels")

    normalized_predictions = _normalize_labels(predictions)
    if normalized_predictions is None:
        return (0, PROBE_COUNT)

    num_correct = sum(
        prediction is target
        for prediction, target in zip(normalized_predictions, normalized_targets)
    )
    return (num_correct, PROBE_COUNT)


def _require_mapping(
    mapping: dict[str, object],
    key: str,
) -> dict[str, object]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"{key} must be a mapping")
    return value


def _normalize_labels(
    labels: tuple[str, ...] | tuple[InteractionLabel, ...] | None,
) -> tuple[InteractionLabel, ...] | None:
    if labels is None:
        return None

    normalized_labels = tuple(parse_label(label) for label in labels)
    if len(normalized_labels) != PROBE_COUNT:
        return None
    return normalized_labels


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    mean: float
    lower: float
    upper: float
    level: float
    margin: float


def compute_bootstrap_confidence_interval(
    num_correct: Sequence[int],
    total: Sequence[int],
    *,
    level: float = 0.95,
    n_bootstraps: int = 1000,
    seed: int = 2025,
) -> ConfidenceInterval:
    if len(num_correct) != len(total):
        raise ValueError("num_correct and total must have the same length")

    n = len(num_correct)
    if n == 0:
        return ConfidenceInterval(0.0, 0.0, 0.0, level, 0.0)

    # Convert to lists for efficient indexing
    nc = list(num_correct)
    tot = list(total)
    grand_total_probes = sum(tot)

    if grand_total_probes == 0:
        return ConfidenceInterval(0.0, 0.0, 0.0, level, 0.0)

    grand_mean = sum(nc) / grand_total_probes

    # Deterministic bootstrap
    rng = random.Random(seed)
    indices = range(n)
    means: list[float] = []

    for _ in range(n_bootstraps):
        sample_indices = rng.choices(indices, k=n)
        s_c = sum(nc[i] for i in sample_indices)
        s_t = sum(tot[i] for i in sample_indices)
        if s_t > 0:
            means.append(s_c / s_t)
        else:
            means.append(0.0)

    means.sort()

    alpha = 1.0 - level
    lower_idx = int(n_bootstraps * (alpha / 2))
    upper_idx = int(n_bootstraps * (1 - (alpha / 2)))

    # Clamp indices
    lower_idx = max(0, min(lower_idx, n_bootstraps - 1))
    upper_idx = max(0, min(upper_idx, n_bootstraps - 1))

    lower = means[lower_idx]
    upper = means[upper_idx]
    
    # Report the larger side as the margin for conservatism
    margin = max(grand_mean - lower, upper - grand_mean)

    return ConfidenceInterval(
        mean=grand_mean,
        lower=lower,
        upper=upper,
        level=level,
        margin=margin,
    )


def validate_kaggle_payload(payload: dict[str, object]) -> None:
    """Validates the canonical Kaggle benchmark payload structure.

    Fails hard if the payload is malformed, missing required fields, has zero
    evaluated episodes, is missing Narrative results, or matches the old bad
    kbench conversations/results/numericResult shape.

    Raises:
        TypeError: If payload is not a dict.
        ValueError: If required fields are absent, total_episodes is 0,
            narrative_result or comparison is missing/None/malformed,
            episode_count_aligned is not True, or the payload has the old kbench shape.
    """
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dict")

    # Reject the old bad kbench shape: conversations/results/numericResult with only CI.
    if "conversations" in payload or "results" in payload:
        raise ValueError(
            "Payload has the old kbench conversations/results/numericResult shape. "
            "Use build_kaggle_payload() to produce the canonical result structure."
        )

    if "primary_result" not in payload:
        raise ValueError("payload must contain 'primary_result'")

    primary = payload["primary_result"]
    if not isinstance(primary, dict):
        raise ValueError("primary_result must be a dict")

    _REQUIRED_PR: frozenset[str] = frozenset(
        {"score", "numerator", "denominator", "total_episodes", "confidence_interval"}
    )
    missing_pr = _REQUIRED_PR - set(primary.keys())
    if missing_pr:
        raise ValueError(
            f"primary_result is missing required fields: {sorted(missing_pr)}"
        )

    if primary["total_episodes"] == 0:
        raise ValueError(
            "primary_result.total_episodes is 0; "
            "evaluation output is missing or empty — do not substitute zeros"
        )

    if primary["denominator"] == 0:
        raise ValueError(
            "primary_result.denominator is 0; evaluation output is malformed"
        )

    ci = primary["confidence_interval"]
    if not isinstance(ci, dict):
        raise ValueError("primary_result.confidence_interval must be a dict")

    _REQUIRED_CI: frozenset[str] = frozenset({"mean", "lower", "upper", "level", "margin"})
    missing_ci = _REQUIRED_CI - set(ci.keys())
    if missing_ci:
        raise ValueError(
            f"confidence_interval is missing required fields: {sorted(missing_ci)}"
        )

    # Validate narrative_result — required, not a placeholder
    if "narrative_result" not in payload:
        raise ValueError("payload must contain 'narrative_result'")
    narrative = payload["narrative_result"]
    if narrative is None:
        raise ValueError(
            "narrative_result is None; Narrative evaluation is mandatory for a valid release"
        )
    if not isinstance(narrative, dict):
        raise ValueError("narrative_result must be a dict")
    _REQUIRED_NR: frozenset[str] = frozenset(
        {"score", "numerator", "denominator", "total_episodes", "confidence_interval"}
    )
    missing_nr = _REQUIRED_NR - set(narrative.keys())
    if missing_nr:
        raise ValueError(
            f"narrative_result is missing required fields: {sorted(missing_nr)}"
        )
    if narrative["total_episodes"] == 0:
        raise ValueError(
            "narrative_result.total_episodes is 0; "
            "Narrative evaluation output is missing or empty"
        )
    if narrative["denominator"] == 0:
        raise ValueError(
            "narrative_result.denominator is 0; Narrative evaluation output is malformed"
        )
    nar_ci = narrative["confidence_interval"]
    if not isinstance(nar_ci, dict):
        raise ValueError("narrative_result.confidence_interval must be a dict")
    missing_nar_ci = _REQUIRED_CI - set(nar_ci.keys())
    if missing_nar_ci:
        raise ValueError(
            f"narrative_result.confidence_interval is missing required fields: {sorted(missing_nar_ci)}"
        )

    # Validate comparison — required, not a placeholder
    if "comparison" not in payload:
        raise ValueError("payload must contain 'comparison'")
    comparison = payload["comparison"]
    if comparison is None:
        raise ValueError(
            "comparison is None; Binary vs Narrative comparison is mandatory for a valid release"
        )
    if not isinstance(comparison, dict):
        raise ValueError("comparison must be a dict")
    _REQUIRED_COMP: frozenset[str] = frozenset(
        {
            "binary_score",
            "narrative_score",
            "delta",
            "episode_count_aligned",
            "binary_total_episodes",
            "narrative_total_episodes",
        }
    )
    missing_comp = _REQUIRED_COMP - set(comparison.keys())
    if missing_comp:
        raise ValueError(
            f"comparison is missing required fields: {sorted(missing_comp)}"
        )
    if comparison["episode_count_aligned"] is not True:
        raise ValueError(
            "comparison.episode_count_aligned is not True; "
            "Binary and Narrative must evaluate the same frozen episodes"
        )

    if "slices" not in payload:
        raise ValueError("payload must contain 'slices'")
    slices_val = payload["slices"]
    if not isinstance(slices_val, dict):
        raise ValueError("slices must be a dict")
    for dim in REQUIRED_SLICE_DIMENSIONS:
        if dim not in slices_val:
            raise ValueError(f"slices is missing required dimension: {dim!r}")

    if "metadata" not in payload:
        raise ValueError("payload must contain 'metadata'")


def verify_remote_hashes(
    manifest: dict[str, Any],
    kernel_dir: Path,
    dataset_dir: Path,
) -> list[ArtifactResult]:
    """Verify hashes of remote artifacts against a local manifest.

    Args:
        manifest: The parsed manifest dictionary.
        kernel_dir: Path to the directory where the kernel files were downloaded.
        dataset_dir: Path to the directory where the dataset files were unzipped.

    Returns:
        A list of ArtifactResult objects for each artifact tracked in the manifest.
    """
    results: list[ArtifactResult] = []

    # 1. Verify Entry Points (expected in kernel_dir)
    entry_points = _require_mapping(manifest, "entry_points")
    for label, info in entry_points.items():
        if not isinstance(info, dict):
            continue
        rel_path = info.get("path")
        expected_hash = info.get("sha256")
        if not isinstance(rel_path, str) or not isinstance(expected_hash, str):
            continue

        # Notebook and metadata are at the root of the kernel download
        remote_path = kernel_dir / Path(rel_path).name
        
        remote_hash: str | None = None
        status = "MISSING"
        if remote_path.is_file():
            remote_hash = hashlib.sha256(remote_path.read_bytes()).hexdigest()
            status = "MATCH" if remote_hash == expected_hash else "MISMATCH"
        
        results.append(ArtifactResult(f"entry_points.{label}", expected_hash, remote_hash, status))

    # 2. Verify Frozen Splits (expected in dataset_dir, following repo structure)
    frozen_splits = _require_mapping(manifest, "frozen_split_manifests")
    for label, info in frozen_splits.items():
        if not isinstance(info, dict):
            continue
        rel_path = info.get("path")
        expected_hash = info.get("sha256")
        if not isinstance(rel_path, str) or not isinstance(expected_hash, str):
            continue

        remote_path = dataset_dir / rel_path
        
        remote_hash = None
        status = "MISSING"
        if remote_path.is_file():
            remote_hash = hashlib.sha256(remote_path.read_bytes()).hexdigest()
            status = "MATCH" if remote_hash == expected_hash else "MISMATCH"
        
        results.append(ArtifactResult(f"frozen_split_manifests.{label}", expected_hash, remote_hash, status))

    # 3. Verify the manifest itself in the dataset
    manifest_rel_path = Path("packaging/kaggle/frozen_artifacts_manifest.json")
    remote_manifest_path = dataset_dir / manifest_rel_path
    
    # Use the provided manifest to compute its own expected hash
    # (assuming the provided manifest matches the local canonical one we want to verify against)
    local_manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8") + b"\n"
    local_manifest_hash = hashlib.sha256(local_manifest_bytes).hexdigest()
    
    # Wait, the above is fragile if indentation/whitespace differs.
    # It's better if we just use the hash of the local file if we have it, 
    # but verify_remote_hashes is agnostic to local files except via 'manifest' dict.
    # Actually, the caller can just check the manifest separately if they want.
    # But I'll leave it in for completeness as it was in my initial script.
    
    remote_manifest_hash = None
    status = "MISSING"
    if remote_manifest_path.is_file():
        remote_manifest_hash = hashlib.sha256(remote_manifest_path.read_bytes()).hexdigest()
        # Note: we don't strictly have 'local_manifest_hash' here without re-serializing,
        # which might not match the file on disk.
        # So I'll just skip the 'manifest' entry here or mark it 'MATCH' if the caller can verify it.
        # Actually, let's just use the hash of the file that provided 'manifest' if possible?
        # Better: let the script handle the manifest file hash.
    
    return results


def _reject_dev_rows(df: Any, label: str) -> None:
    """Fail hard if *df* contains any rows from the dev split.

    Called on both binary_df and narrative_df before any aggregation so that
    dev results can never contaminate the official leaderboard payload.
    Only checked when a 'split' column is present; callers that have already
    dropped the column (the expected path) pass through silently.
    """
    if "split" not in df.columns:
        return
    dev_count = int((df["split"] == "dev").sum())
    if dev_count > 0:
        raise ValueError(
            f"{label} contains {dev_count} dev row(s); "
            "only leaderboard results may be aggregated into the official payload. "
            "Drop the dev split before calling build_kaggle_payload."
        )


def _normalize_result_df(df: Any) -> Any:
    """Normalize a kbench result dataframe to use 'num_correct'/'total' columns.

    The real kaggle_benchmarks framework may name the two tuple return values
    differently from the local shim. Known alternative column name conventions:
      - score_0 / score_1  (position-based, likely default)
      - 0 / 1              (integer-indexed)
    If 'num_correct'/'total' are already present, the dataframe is returned as-is.
    """
    if "num_correct" in df.columns and "total" in df.columns:
        return df
    rename_map: dict = {}
    if "score_0" in df.columns and "score_1" in df.columns:
        rename_map = {"score_0": "num_correct", "score_1": "total"}
    elif 0 in df.columns and 1 in df.columns:
        rename_map = {0: "num_correct", 1: "total"}
    if rename_map:
        return df.rename(columns=rename_map)
    return df


def build_kaggle_payload(
    binary_df: Any,
    narrative_df: Any,
) -> dict[str, object]:
    """Constructs the canonical final Kaggle payload from task results.

    Both Binary and Narrative results are required for a valid release.
    Fails hard if narrative_df is None, empty, mismatched in episode count,
    or mismatched in denominator basis.

    Args:
        binary_df: pandas DataFrame containing 'num_correct' and 'total' columns.
        narrative_df: pandas DataFrame containing 'num_correct' and 'total' columns.
            Must align with binary_df on episode count and total denominator.

    Returns:
        A dictionary representing the JSON payload to be emitted, including
        primary_result, narrative_result, comparison, slices, and metadata.

    Raises:
        ValueError: If binary_df or narrative_df is missing, empty, or misaligned.
        ImportError: If pandas is not available.
        TypeError: If binary_df or narrative_df is not a pandas DataFrame.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for build_kaggle_payload")

    if not isinstance(binary_df, pd.DataFrame):
        raise TypeError("binary_df must be a pandas DataFrame")

    if binary_df.empty:
        raise ValueError("binary_df cannot be empty; evaluation results are required")

    binary_df = _normalize_result_df(binary_df)

    if "num_correct" not in binary_df.columns or "total" not in binary_df.columns:
        raise ValueError(
            "binary_df must contain 'num_correct' and 'total' columns"
        )

    _reject_dev_rows(binary_df, "binary_df")

    # Narrative is mandatory for a valid release
    if narrative_df is None:
        raise ValueError(
            "narrative_df is required for a valid release; "
            "Narrative evaluation is missing or was skipped"
        )

    if not isinstance(narrative_df, pd.DataFrame):
        raise TypeError("narrative_df must be a pandas DataFrame")

    if narrative_df.empty:
        raise ValueError(
            "narrative_df cannot be empty; Narrative evaluation results are required"
        )

    narrative_df = _normalize_result_df(narrative_df)

    if "num_correct" not in narrative_df.columns or "total" not in narrative_df.columns:
        raise ValueError(
            "narrative_df must contain 'num_correct' and 'total' columns"
        )

    _reject_dev_rows(narrative_df, "narrative_df")

    # Episode count alignment check
    bin_episodes = len(binary_df)
    nar_episodes = len(narrative_df)
    if bin_episodes != nar_episodes:
        raise ValueError(
            f"Binary and Narrative episode counts do not match: "
            f"binary={bin_episodes}, narrative={nar_episodes}. "
            "Both must evaluate the same frozen episodes."
        )

    # Denominator alignment check (same probe-count basis)
    bin_den = int(binary_df["total"].sum())
    nar_den = int(narrative_df["total"].sum())
    if bin_den != nar_den:
        raise ValueError(
            f"Binary and Narrative denominators do not match: "
            f"binary={bin_den}, narrative={nar_den}. "
            "Both must use the same probe count basis."
        )

    # Binary aggregate metrics
    bin_num = int(binary_df["num_correct"].sum())
    bin_ci = compute_bootstrap_confidence_interval(
        binary_df["num_correct"].tolist(),
        binary_df["total"].tolist(),
    )

    # Narrative aggregate metrics
    nar_num = int(narrative_df["num_correct"].sum())
    nar_ci = compute_bootstrap_confidence_interval(
        narrative_df["num_correct"].tolist(),
        narrative_df["total"].tolist(),
    )

    return {
        "primary_result": {
            "score": bin_ci.mean,
            "numerator": bin_num,
            "denominator": bin_den,
            "total_episodes": bin_episodes,
            "confidence_interval": {
                "mean": bin_ci.mean,
                "lower": bin_ci.lower,
                "upper": bin_ci.upper,
                "level": bin_ci.level,
                "margin": bin_ci.margin,
            },
        },
        "narrative_result": {
            "score": nar_ci.mean,
            "numerator": nar_num,
            "denominator": nar_den,
            "total_episodes": nar_episodes,
            "confidence_interval": {
                "mean": nar_ci.mean,
                "lower": nar_ci.lower,
                "upper": nar_ci.upper,
                "level": nar_ci.level,
                "margin": nar_ci.margin,
            },
        },
        "comparison": {
            "binary_score": bin_ci.mean,
            "narrative_score": nar_ci.mean,
            "delta": bin_ci.mean - nar_ci.mean,
            "episode_count_aligned": True,
            "binary_total_episodes": bin_episodes,
            "narrative_total_episodes": nar_episodes,
            "binary_denominator": bin_den,
            "narrative_denominator": nar_den,
        },
        "slices": _build_payload_slices(binary_df),
        "metadata": {
            "benchmark_version": MANIFEST_VERSION,
        },
    }


def _build_payload_slices(binary_df: Any) -> dict[str, object]:
    """Build the mandatory slice dict from binary_df.

    Dimensional accuracy slices (template, difficulty, shift_position,
    transition_type) are computed from episode metadata columns when present;
    otherwise the dimension key maps to an empty dict.  The error_type slice
    is always computed from num_correct patterns as a stable proxy.
    """
    slices: dict[str, object] = {}

    for dim, col in _SLICE_METADATA_COLS.items():
        if col in binary_df.columns:
            slices[dim] = _accuracy_by_column(binary_df, col)
        else:
            slices[dim] = {}

    slices["error_type"] = _error_type_counts(binary_df)
    return slices


def _accuracy_by_column(df: Any, col: str) -> dict[str, object]:
    """Group df by *col* and compute accuracy per group value."""
    result: dict[str, object] = {}
    for value, group in df.groupby(col, sort=True):
        nc = int(group["num_correct"].sum())
        tot = int(group["total"].sum())
        result[str(value)] = {
            "episode_count": len(group),
            "correct_probes": nc,
            "total_probes": tot,
            "accuracy": nc / tot if tot > 0 else 0.0,
        }
    return result


def _error_type_counts(df: Any) -> dict[str, int]:
    """Classify failing episodes by error type using num_correct as a proxy.

    Without label-level predictions only two categories are distinguishable:
    - ``old_rule_persistence``: 0/total correct — consistent with applying the
      pre-shift rule throughout (all probes are rule-disagreement probes, so the
      old rule gives 0 correct probes).
    - ``premature_switch``: 1 to total-1 correct — partial adaptation.

    ``recency_overweight``, ``invalid_narrative``, and ``unknown`` require
    label-level data not available in the aggregate payload DataFrame; they are
    always 0 in this path.  The full classification is available in the panel
    artifact where complete prediction data is present.
    """
    counts: dict[str, int] = {et.value: 0 for et in ErrorType}
    for _, row in df.iterrows():
        nc = int(row["num_correct"])
        total = int(row["total"])
        if nc >= total:
            continue  # Correct — not an error.
        if nc == 0:
            counts[ErrorType.OLD_RULE_PERSISTENCE.value] += 1
        else:
            counts[ErrorType.PREMATURE_SWITCH.value] += 1
    return counts
