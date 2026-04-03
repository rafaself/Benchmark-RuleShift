from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import os
from pathlib import Path
from typing import Final


class Label(str, Enum):
    type_a = "type_a"
    type_b = "type_b"


PROBE_COUNT: Final[int] = 4
PRIVATE_DATASET_ROOT_ENV_VAR: Final[str] = "RULESHIFT_PRIVATE_DATASET_ROOT"

_PUBLIC_ROWS_FILENAME: Final[str] = "public_leaderboard_rows.json"
_PRIVATE_ROWS_FILENAME: Final[str] = "private_leaderboard_rows.json"
_FROZEN_SPLITS_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "frozen_splits"
_ALLOWED_LABELS: Final[frozenset[str]] = frozenset(label.value for label in Label)

_PROBE_FIELDS: Final[tuple[str, ...]] = ("probe_6", "probe_7", "probe_8", "probe_9")


@dataclass(frozen=True)
class BinaryResponse:
    probe_6: Label
    probe_7: Label
    probe_8: Label
    probe_9: Label

    def as_tuple(self) -> tuple[str, str, str, str]:
        return (
            _coerce_label(self.probe_6, "probe_6"),
            _coerce_label(self.probe_7, "probe_7"),
            _coerce_label(self.probe_8, "probe_8"),
            _coerce_label(self.probe_9, "probe_9"),
        )


class KaggleExecutionError(RuntimeError):
    pass


def load_public_rows() -> list[dict[str, object]]:
    return _load_rows(_FROZEN_SPLITS_DIR / _PUBLIC_ROWS_FILENAME)


def load_private_rows(
    private_dataset_root: Path | str | None = None,
) -> list[dict[str, object]]:
    if private_dataset_root is None:
        env = os.environ.get(PRIVATE_DATASET_ROOT_ENV_VAR)
        if not env:
            raise FileNotFoundError(
                f"Private dataset not found. Set {PRIVATE_DATASET_ROOT_ENV_VAR} or "
                "pass private_dataset_root explicitly."
            )
        private_dataset_root = env
    root = Path(private_dataset_root)
    return _load_rows(root / _PRIVATE_ROWS_FILENAME)


def normalize_binary_response(response: object) -> tuple[str, ...] | None:
    if response is None:
        return None
    if isinstance(response, BinaryResponse):
        return response.as_tuple()
    if isinstance(response, str):
        return _parse_text_response(response)
    br = _try_coerce(response)
    return br.as_tuple() if br is not None else None


def score_episode(
    predictions: tuple[str, ...] | tuple[Label, ...] | None,
    targets: tuple[str, ...] | tuple[Label, ...],
) -> tuple[int, int]:
    norm_targets = _norm_labels(targets)
    if norm_targets is None:
        raise ValueError(f"targets must contain exactly {PROBE_COUNT} valid labels")
    norm_preds = _norm_labels(predictions)
    if norm_preds is None:
        return (0, PROBE_COUNT)
    return (
        sum(p == t for p, t in zip(norm_preds, norm_targets)),
        PROBE_COUNT,
    )


def run_binary_task(
    *,
    llm: object,
    prompt_binary: str,
    probe_targets: tuple[str, ...] | tuple[Label, ...],
) -> tuple[int, int]:
    try:
        response = llm.prompt(prompt_binary, schema=BinaryResponse)
    except Exception as exc:
        raise KaggleExecutionError("llm.prompt failed") from exc

    try:
        normalized = normalize_binary_response(response)
    except ValueError as exc:
        raise KaggleExecutionError(f"invalid binary response: {exc}") from exc

    if normalized is None:
        raise KaggleExecutionError(
            f"unscoreable response of type {type(response).__name__}"
        )
    return score_episode(normalized, probe_targets)


def _load_rows(path: Path) -> list[dict[str, object]]:
    rows = json.loads(path.read_text("utf-8"))
    return [
        {
            "episode_id": row["episode_id"],
            "split": row["split"],
            "prompt_binary": row["prompt_binary"],
            "probe_targets": _normalize_probe_targets(row["probe_targets"]),
        }
        for row in rows
    ]


def _normalize_probe_targets(values: object) -> tuple[str, ...]:
    if not isinstance(values, list | tuple):
        raise ValueError("probe_targets must be a list or tuple")
    labels = _norm_labels(tuple(values))
    if labels is None:
        raise ValueError(f"probe_targets must contain exactly {PROBE_COUNT} valid labels")
    return labels


def _parse_text_response(text: str) -> tuple[str, ...] | None:
    tokens = tuple(
        t.strip().lower()
        for t in text.strip().strip("`").replace("\n", ",").split(",")
        if t.strip()
    )
    return _norm_labels(tokens)


def _try_coerce(response: object) -> BinaryResponse | None:
    if isinstance(response, dict):
        vals = response
    elif hasattr(response, "__getitem__") and hasattr(response, "keys"):
        try:
            vals = {f: response[f] for f in _PROBE_FIELDS}
        except (KeyError, TypeError):
            return None
    elif all(hasattr(response, f) for f in _PROBE_FIELDS):
        vals = {f: getattr(response, f) for f in _PROBE_FIELDS}
    else:
        return None
    try:
        labels = tuple(Label(_coerce_label(vals[f], f)) for f in _PROBE_FIELDS)
    except (KeyError, TypeError):
        return None
    return BinaryResponse(*labels)


def _coerce_label(value: object, field: str) -> str:
    try:
        return _normalize_label(value)
    except ValueError as exc:
        raise ValueError(f"invalid field {field}: {value!r}") from exc


def _norm_labels(
    labels: tuple[str, ...] | tuple[Label, ...] | None,
) -> tuple[str, ...] | None:
    if labels is None:
        return None
    try:
        result = tuple(
            _normalize_label(lbl.value if isinstance(lbl, Label) else lbl)
            for lbl in labels
        )
    except ValueError:
        return None
    return result if len(result) == PROBE_COUNT else None


def _normalize_label(value: object) -> str:
    if isinstance(value, Enum):
        value = value.value
    elif hasattr(value, "value"):
        value = value.value
    normalized = str(value).strip().lower()
    if normalized not in _ALLOWED_LABELS:
        raise ValueError(f"unknown label: {value}")
    return normalized
