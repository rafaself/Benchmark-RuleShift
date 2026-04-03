from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, StrEnum
import json
import os
from pathlib import Path
from typing import Final


class InteractionLabel(StrEnum):
    ZARK = "zark"
    BLIM = "blim"


class Label(str, Enum):
    type_a = "type_a"
    type_b = "type_b"


PROBE_COUNT: Final[int] = 4
MANIFEST_VERSION: Final[str] = "R14"
PRIVATE_DATASET_ROOT_ENV_VAR: Final[str] = "RULESHIFT_PRIVATE_DATASET_ROOT"

_PUBLIC_ROWS_FILENAME: Final[str] = "public_leaderboard_rows.json"
_PRIVATE_ROWS_FILENAME: Final[str] = "private_leaderboard_rows.json"
_MANIFEST_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "frozen_splits"

_PUBLIC_LABEL_MAP: Final[dict[str, InteractionLabel]] = {
    "type_a": InteractionLabel.ZARK,
    "type_b": InteractionLabel.BLIM,
}
_INTERNAL_TO_PUBLIC: Final[dict[InteractionLabel, str]] = {
    InteractionLabel.ZARK: "type_a",
    InteractionLabel.BLIM: "type_b",
}

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


def parse_public_label(value: str) -> InteractionLabel:
    normalized = value.strip().lower()
    if normalized in _PUBLIC_LABEL_MAP:
        return _PUBLIC_LABEL_MAP[normalized]
    raise ValueError(f"unknown public label: {value}")


def format_public_label(lbl: InteractionLabel | str) -> str:
    if isinstance(lbl, str):
        lbl = InteractionLabel(lbl)
    return _INTERNAL_TO_PUBLIC[lbl]


def load_public_rows() -> list[dict[str, object]]:
    rows = json.loads((_MANIFEST_DIR / _PUBLIC_ROWS_FILENAME).read_text("utf-8"))
    return [
        {
            "episode_id": r["episode_id"],
            "split": r["split"],
            "prompt_binary": r["prompt_binary"],
            "probe_targets": tuple(r["probe_targets"]),
        }
        for r in rows
    ]


def discover_private_dataset_root(
    private_dataset_root: Path | str | None = None,
) -> Path | None:
    if private_dataset_root is not None:
        root = Path(private_dataset_root)
        if (root / _PRIVATE_ROWS_FILENAME).is_file():
            return root
        raise FileNotFoundError(f"{_PRIVATE_ROWS_FILENAME} not found at {root}")

    env = os.environ.get(PRIVATE_DATASET_ROOT_ENV_VAR)
    if env:
        root = Path(env)
        if (root / _PRIVATE_ROWS_FILENAME).is_file():
            return root
        raise FileNotFoundError(f"{_PRIVATE_ROWS_FILENAME} not found at {root}")

    return None


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
    rows = json.loads((root / _PRIVATE_ROWS_FILENAME).read_text("utf-8"))
    return [
        {
            "episode_id": r["episode_id"],
            "split": r["split"],
            "prompt_binary": r["prompt_binary"],
            "probe_targets": tuple(r["probe_targets"]),
        }
        for r in rows
    ]


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
    predictions: tuple[str, ...] | tuple[InteractionLabel, ...] | None,
    targets: tuple[str, ...] | tuple[InteractionLabel, ...],
) -> tuple[int, int]:
    norm_targets = _norm_labels(targets)
    if norm_targets is None:
        raise ValueError(f"targets must contain exactly {PROBE_COUNT} valid labels")
    norm_preds = _norm_labels(predictions)
    if norm_preds is None:
        return (0, PROBE_COUNT)
    return (
        sum(p is t for p, t in zip(norm_preds, norm_targets)),
        PROBE_COUNT,
    )


def run_binary_task(
    *,
    llm: object,
    prompt_binary: str,
    probe_targets: tuple[str, ...] | tuple[InteractionLabel, ...],
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


def _parse_text_response(text: str) -> tuple[str, ...] | None:
    tokens = tuple(
        t.strip().lower()
        for t in text.strip().strip("`").replace("\n", ",").split(",")
        if t.strip()
    )
    if len(tokens) != PROBE_COUNT:
        return None
    try:
        return tuple(format_public_label(parse_public_label(t)) for t in tokens)
    except ValueError:
        return None


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
    if isinstance(value, str):
        raw = value
    elif isinstance(value, Enum):
        raw = value.value
    elif hasattr(value, "value"):
        raw = str(value.value)
    else:
        raw = str(value)
    try:
        return format_public_label(parse_public_label(raw))
    except ValueError as exc:
        raise ValueError(f"invalid field {field}: {raw!r}") from exc


def _norm_labels(
    labels: tuple[str, ...] | tuple[InteractionLabel, ...] | None,
) -> tuple[InteractionLabel, ...] | None:
    if labels is None:
        return None
    try:
        result = tuple(
            lbl if isinstance(lbl, InteractionLabel) else parse_public_label(lbl)
            for lbl in labels
        )
    except ValueError:
        return None
    return result if len(result) == PROBE_COUNT else None
