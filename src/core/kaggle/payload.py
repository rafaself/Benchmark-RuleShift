from __future__ import annotations

from dataclasses import dataclass
import random
from typing import TYPE_CHECKING, Any, Final, Sequence

from core.splits import MANIFEST_VERSION

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "REQUIRED_PAYLOAD_FIELDS",
    "build_kaggle_payload",
    "compute_bootstrap_confidence_interval",
    "normalize_count_result_df",
    "validate_kaggle_payload",
]

REQUIRED_PAYLOAD_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "score",
        "numerator",
        "denominator",
        "total_episodes",
        "benchmark_version",
        "split",
        "manifest_version",
    }
)


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    mean: float
    lower: float
    upper: float
    level: float
    margin: float


def validate_kaggle_payload(payload: dict[str, object]) -> None:
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dict")

    if "conversations" in payload or "results" in payload:
        raise ValueError(
            "Payload has the old kbench conversations/results/numericResult shape. "
            "Use build_kaggle_payload() to produce the canonical result structure."
        )

    if set(payload) != REQUIRED_PAYLOAD_FIELDS:
        raise ValueError(
            "payload must contain exactly these fields: "
            f"{sorted(REQUIRED_PAYLOAD_FIELDS)}"
        )

    if not isinstance(payload["score"], (int, float)):
        raise ValueError("score must be numeric")
    if not isinstance(payload["numerator"], int):
        raise ValueError("numerator must be an int")
    if not isinstance(payload["denominator"], int):
        raise ValueError("denominator must be an int")
    if not isinstance(payload["total_episodes"], int):
        raise ValueError("total_episodes must be an int")
    if payload["total_episodes"] == 0:
        raise ValueError(
            "total_episodes is 0; "
            "evaluation output is missing or empty — do not substitute zeros"
        )
    if payload["denominator"] == 0:
        raise ValueError("denominator is 0; evaluation output is malformed")
    if not isinstance(payload["benchmark_version"], str) or not payload["benchmark_version"]:
        raise ValueError("benchmark_version must be a non-empty string")
    if not isinstance(payload["split"], str) or not payload["split"]:
        raise ValueError("split must be a non-empty string")
    if not isinstance(payload["manifest_version"], str) or not payload["manifest_version"]:
        raise ValueError("manifest_version must be a non-empty string")


def build_kaggle_payload(binary_df: Any) -> dict[str, object]:
    binary_df = normalize_count_result_df(binary_df, dataframe_label="binary_df")
    if binary_df.empty:
        raise ValueError("binary_df cannot be empty; evaluation results are required")
    if "num_correct" not in binary_df.columns or "total" not in binary_df.columns:
        raise ValueError("binary_df must contain 'num_correct' and 'total' columns")

    _reject_dev_rows(binary_df, "binary_df")

    numerator = int(binary_df["num_correct"].sum())
    denominator = int(binary_df["total"].sum())
    total_episodes = len(binary_df)
    score = compute_bootstrap_confidence_interval(
        binary_df["num_correct"].tolist(),
        binary_df["total"].tolist(),
    ).mean

    return {
        "score": score,
        "numerator": numerator,
        "denominator": denominator,
        "total_episodes": total_episodes,
        "benchmark_version": MANIFEST_VERSION,
        "split": _resolve_payload_split(binary_df),
        "manifest_version": MANIFEST_VERSION,
    }


def normalize_count_result_df(
    df: "pd.DataFrame",
    *,
    dataframe_label: str = "result_df",
) -> "pd.DataFrame":
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for normalize_count_result_df")

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{dataframe_label} must be a pandas DataFrame")

    out = df.copy()

    if "num_correct" in out.columns and "total" in out.columns:
        return out

    if "result" in out.columns:
        invalid_mask = ~out["result"].map(
            lambda value: isinstance(value, (tuple, list)) and len(value) == 2
        )
        if invalid_mask.any():
            raise ValueError(
                f"{dataframe_label}.result must contain 2-item tuple/list values"
            )

        if out.empty:
            out = out.drop(columns=["result"])
            out["num_correct"] = pd.Series(index=out.index, dtype="object")
            out["total"] = pd.Series(index=out.index, dtype="object")
            return out

        expanded = pd.DataFrame(out["result"].tolist(), index=out.index)
        out[["num_correct", "total"]] = expanded
        return out.drop(columns=["result"])

    if "score_0" in out.columns and "score_1" in out.columns:
        return out.rename(columns={"score_0": "num_correct", "score_1": "total"})

    if 0 in out.columns and 1 in out.columns:
        return out.rename(columns={0: "num_correct", 1: "total"})

    return out


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

    nc = list(num_correct)
    tot = list(total)
    grand_total_probes = sum(tot)
    if grand_total_probes == 0:
        return ConfidenceInterval(0.0, 0.0, 0.0, level, 0.0)

    grand_mean = sum(nc) / grand_total_probes
    rng = random.Random(seed)
    indices = range(n)
    means: list[float] = []

    for _ in range(n_bootstraps):
        sample_indices = rng.choices(indices, k=n)
        sample_correct = sum(nc[i] for i in sample_indices)
        sample_total = sum(tot[i] for i in sample_indices)
        if sample_total > 0:
            means.append(sample_correct / sample_total)

    means.sort()
    alpha = 1.0 - level
    lower_idx = int((alpha / 2) * (len(means) - 1))
    upper_idx = int((1 - alpha / 2) * (len(means) - 1))
    lower = means[lower_idx]
    upper = means[upper_idx]
    return ConfidenceInterval(
        mean=grand_mean,
        lower=lower,
        upper=upper,
        level=level,
        margin=max(grand_mean - lower, upper - grand_mean),
    )


def _reject_dev_rows(df: Any, label: str) -> None:
    if "split" not in df.columns:
        return
    dev_count = int((df["split"] == "dev").sum())
    if dev_count > 0:
        raise ValueError(
            f"{label} contains {dev_count} dev row(s); "
            "only leaderboard results may be aggregated into the official payload. "
            "Drop the dev split before calling build_kaggle_payload."
        )


def _resolve_payload_split(binary_df: Any) -> str:
    if "split" not in binary_df.columns:
        return "frozen_leaderboard"

    splits = {str(value) for value in binary_df["split"].dropna().unique()}
    if len(splits) == 1:
        return next(iter(splits))
    if splits == {"public_leaderboard", "private_leaderboard"}:
        return "frozen_leaderboard"
    return "frozen_leaderboard"
