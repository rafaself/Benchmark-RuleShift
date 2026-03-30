from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.private_split import discover_private_dataset_root, load_private_split
from core.splits import load_frozen_split
from tasks.ruleshift_benchmark.protocol import PROBE_COUNT, InteractionLabel, parse_label
from tasks.ruleshift_benchmark.render import render_binary_prompt

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "find_repo_root",
    "load_leaderboard_dataframe",
    "run_binary_task",
]


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


def find_repo_root() -> Path:
    candidates: list[Path] = []
    seen: set[Path] = set()

    for origin in (Path.cwd().resolve(),):
        for candidate in (origin, *origin.parents):
            if candidate not in seen:
                seen.add(candidate)
                candidates.append(candidate)

    for search_root in (Path("/kaggle/input"), Path("/kaggle/working")):
        if not search_root.exists():
            continue
        for manifest_path in search_root.rglob("frozen_artifacts_manifest.json"):
            candidate = manifest_path.parents[2]
            if candidate not in seen:
                seen.add(candidate)
                candidates.append(candidate)

    for candidate in candidates:
        if (candidate / "src").is_dir() and (
            candidate / "packaging" / "kaggle" / "frozen_artifacts_manifest.json"
        ).is_file():
            return candidate

    raise FileNotFoundError(
        "Could not locate repo root. Expected src/ and "
        "packaging/kaggle/frozen_artifacts_manifest.json."
    )


def load_leaderboard_dataframe(
    *,
    repo_root: Path | str | None = None,
) -> tuple[Path | None, dict[str, object], "pd.DataFrame"]:
    del repo_root
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for load_leaderboard_dataframe")

    frozen_splits: dict[str, object] = {
        "public_leaderboard": load_frozen_split("public_leaderboard"),
    }

    try:
        private_dataset_root = discover_private_dataset_root()
    except FileNotFoundError:
        private_dataset_root = None
    else:
        if private_dataset_root is not None:
            frozen_splits["private_leaderboard"] = load_private_split(private_dataset_root)

    leaderboard_rows = [
        row
        for partition in ("public_leaderboard", "private_leaderboard")
        if partition in frozen_splits
        for row in _build_rows(partition, frozen_splits[partition])
    ]
    leaderboard_df = pd.DataFrame(leaderboard_rows)
    if leaderboard_df.empty:
        raise ValueError("leaderboard_df cannot be empty")
    return private_dataset_root, frozen_splits, leaderboard_df


def run_binary_task(
    *,
    llm: object,
    prompt_binary: str,
    probe_targets: tuple[str, ...] | tuple[InteractionLabel, ...],
) -> tuple[int, int]:
    try:
        response = llm.prompt(prompt_binary, schema=BinaryResponse)
    except Exception:
        response = None
    return score_episode(normalize_binary_response(response), probe_targets)


def normalize_binary_response(response: object) -> tuple[str, ...] | None:
    if response is None:
        return None
    if isinstance(response, BinaryResponse):
        return response.as_tuple()
    if isinstance(response, str):
        return _parse_binary_output(response)

    binary_response = _try_coerce_to_binary_response(response)
    if binary_response is not None:
        return binary_response.as_tuple()
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


def _build_rows(partition: str, records: Any) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        episode = record.episode
        rows.append(
            {
                "episode_id": episode.episode_id,
                "split": partition,
                "prompt_binary": render_binary_prompt(episode),
                "probe_targets": tuple(label.value for label in episode.probe_targets),
            }
        )
    return rows


def _parse_binary_output(text: str) -> tuple[str, ...] | None:
    normalized_text = text.strip().strip("`").replace("\n", ",")
    tokens = tuple(token.strip().lower() for token in normalized_text.split(",") if token.strip())
    if len(tokens) != PROBE_COUNT:
        return None
    try:
        return tuple(parse_label(token).value for token in tokens)
    except ValueError:
        return None


def _try_coerce_to_binary_response(response: object) -> BinaryResponse | None:
    values: dict[str, object]

    if isinstance(response, dict):
        values = response
    elif hasattr(response, "__getitem__") and hasattr(response, "keys"):
        try:
            values = {field: response[field] for field in ("probe_6", "probe_7", "probe_8", "probe_9")}
        except (KeyError, TypeError):
            return None
    elif all(hasattr(response, field) for field in ("probe_6", "probe_7", "probe_8", "probe_9")):
        values = {field: getattr(response, field) for field in ("probe_6", "probe_7", "probe_8", "probe_9")}
    else:
        return None

    try:
        labels = tuple(
            Label(_extract_label_value(values[field]))
            for field in ("probe_6", "probe_7", "probe_8", "probe_9")
        )
    except (KeyError, TypeError, ValueError):
        return None
    return BinaryResponse(*labels)


def _extract_label_value(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


def _normalize_labels(
    labels: tuple[str, ...] | tuple[InteractionLabel, ...] | None,
) -> tuple[InteractionLabel, ...] | None:
    if labels is None:
        return None
    normalized_labels = tuple(parse_label(label) for label in labels)
    if len(normalized_labels) != PROBE_COUNT:
        return None
    return normalized_labels
