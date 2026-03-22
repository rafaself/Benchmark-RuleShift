from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re

from tasks.iron_find_electric.protocol import (
    PROBE_COUNT,
    InteractionLabel,
    parse_label,
)

__all__ = [
    "ParseStatus",
    "ParsedPrediction",
    "parse_binary_output",
    "parse_narrative_output",
]

_FINAL_LABELS_PATTERN = re.compile(r"final labels:", re.IGNORECASE)
_SEPARATOR_PATTERN = re.compile(r"[\n,]+")


class ParseStatus(StrEnum):
    VALID = "valid"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class ParsedPrediction:
    labels: tuple[InteractionLabel, ...]
    status: ParseStatus


_INVALID_PREDICTION = ParsedPrediction(labels=(), status=ParseStatus.INVALID)


def parse_binary_output(text: str) -> ParsedPrediction:
    return _parse_labels_payload(text)


def parse_narrative_output(text: str) -> ParsedPrediction:
    matches = tuple(_FINAL_LABELS_PATTERN.finditer(text))
    if not matches:
        return _INVALID_PREDICTION

    return _parse_labels_payload(text[matches[-1].end() :])


def _parse_labels_payload(text: str) -> ParsedPrediction:
    normalized_text = text.strip()
    if not normalized_text:
        return _INVALID_PREDICTION

    raw_tokens = tuple(_SEPARATOR_PATTERN.split(normalized_text))
    normalized_tokens = tuple(token.strip().lower() for token in raw_tokens if token.strip())
    if len(normalized_tokens) != PROBE_COUNT:
        return _INVALID_PREDICTION

    try:
        labels = tuple(parse_label(token) for token in normalized_tokens)
    except ValueError:
        return _INVALID_PREDICTION

    return ParsedPrediction(labels=labels, status=ParseStatus.VALID)
