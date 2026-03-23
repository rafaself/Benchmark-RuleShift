from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re

from tasks.ruleshift_benchmark.protocol import (
    PROBE_COUNT,
    InteractionLabel,
    parse_label,
)

__all__ = [
    "PARSER_VERSION",
    "ParseStatus",
    "ParsedPrediction",
    "parse_binary_output",
    "parse_narrative_output",
]

PARSER_VERSION = "v1"
_FINAL_LABELS_PATTERN = re.compile(r"final labels:", re.IGNORECASE)
_FINAL_ANSWER_PATTERN = re.compile(r"final answers?:", re.IGNORECASE)
_SEPARATOR_PATTERN = re.compile(r"[\n,]+")
_NUMBER_PREFIX_RE = re.compile(r"^\d+\.?\s*")
_BOLD_MARKER_RE = re.compile(r"\*+")


class ParseStatus(StrEnum):
    VALID = "valid"
    INVALID = "invalid"
    SKIPPED_PROVIDER_FAILURE = "skipped_provider_failure"


@dataclass(frozen=True, slots=True)
class ParsedPrediction:
    labels: tuple[InteractionLabel, ...]
    status: ParseStatus

    @classmethod
    def skipped_provider_failure(cls) -> "ParsedPrediction":
        return cls(labels=(), status=ParseStatus.SKIPPED_PROVIDER_FAILURE)


_INVALID_PREDICTION = ParsedPrediction(labels=(), status=ParseStatus.INVALID)


def parse_binary_output(text: str) -> ParsedPrediction:
    return _parse_labels_payload(text)


def parse_narrative_output(text: str) -> ParsedPrediction:
    normalized_text = text.strip()
    if not normalized_text:
        return _INVALID_PREDICTION

    # Layer 1: "final labels:" marker (strict — no fallback).
    fl_matches = tuple(_FINAL_LABELS_PATTERN.finditer(text))
    if fl_matches:
        return _parse_labels_payload(text[fl_matches[-1].end() :])

    nonempty_lines = tuple(
        line.strip() for line in normalized_text.splitlines() if line.strip()
    )
    if not nonempty_lines:
        return _INVALID_PREDICTION

    # Layer 2: "final answer(s):" marker (fallible).
    fa_matches = tuple(_FINAL_ANSWER_PATTERN.finditer(text))
    if fa_matches:
        result = _parse_labels_payload(text[fa_matches[-1].end() :])
        if result.status is ParseStatus.VALID:
            return result

    # Layer 3: single last non-empty line (fallible).
    result = _parse_labels_payload(nonempty_lines[-1])
    if result.status is ParseStatus.VALID:
        return result

    # Layer 4: last PROBE_COUNT non-empty lines, cleaned.
    return _parse_trailing_label_lines(nonempty_lines)


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


def _parse_trailing_label_lines(lines: tuple[str, ...]) -> ParsedPrediction:
    if len(lines) < PROBE_COUNT:
        return _INVALID_PREDICTION

    tail = lines[-PROBE_COUNT:]
    cleaned: list[str] = []
    for line in tail:
        token = _BOLD_MARKER_RE.sub("", line).strip()
        token = _NUMBER_PREFIX_RE.sub("", token).strip().lower()
        if not token:
            return _INVALID_PREDICTION
        cleaned.append(token)

    try:
        labels = tuple(parse_label(t) for t in cleaned)
    except ValueError:
        return _INVALID_PREDICTION

    return ParsedPrediction(labels=labels, status=ParseStatus.VALID)
