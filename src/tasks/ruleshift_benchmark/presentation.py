from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Final

from tasks.ruleshift_benchmark.protocol import (
    TemplateFamily,
    format_public_state,
)
from tasks.ruleshift_benchmark.schema import EpisodeItem

__all__ = [
    "BinaryPresentation",
    "BINARY_PRESENTATIONS",
]


@dataclass(frozen=True, slots=True)
class BinaryPresentation:
    intro: str
    labeled_heading: str
    probe_heading: str
    outro: str
    line_renderer: Callable[[EpisodeItem], str]


def render_binary_line(item: EpisodeItem) -> str:
    return (
        f"{item.position}. r1={_format_marker_value(item.q1)}, "
        f"r2={_format_marker_value(item.q2)} -> {_render_outcome(item)}"
    )


def render_binary_log_line(item: EpisodeItem) -> str:
    return (
        f"[{item.position:02d}] r1={_format_marker_value(item.q1)} | "
        f"r2={_format_marker_value(item.q2)} | observed={_render_outcome(item)}"
    )


def render_binary_ledger_line(item: EpisodeItem) -> str:
    return (
        f"row {item.position:02d} | r1={_format_marker_value(item.q1)} | "
        f"r2={_format_marker_value(item.q2)} | state={_render_outcome(item)}"
    )


def _render_outcome(item: EpisodeItem) -> str:
    return format_public_state(item.label) if item.label is not None else "?"


def _format_marker_value(marker_value: int) -> str:
    return f"{marker_value:+d}"


_BINARY_OUTRO: Final[str] = (
    "Return exactly 4 outputs in order, one per probe. "
    "Use only type_a or type_b. Map zark to type_a and blim to type_b."
)

BINARY_PRESENTATIONS: Final[dict[TemplateFamily, BinaryPresentation]] = {
    TemplateFamily.CANONICAL: BinaryPresentation(
        intro=(
            "You are given labeled records for two markers.\n"
            "Each labeled line shows r1, r2, and the observed state.\n"
            "Use the full sequence to infer which sign combinations were revised by the later evidence, "
            "then answer the final unlabeled cases."
        ),
        labeled_heading="Labeled examples:",
        probe_heading="Probes:",
        outro=_BINARY_OUTRO,
        line_renderer=render_binary_line,
    ),
    TemplateFamily.OBSERVATION_LOG: BinaryPresentation(
        intro=(
            "Review the observation log for two markers.\n"
            "Each entry records r1, r2, and the observed state.\n"
            "Use the full log to infer which sign combinations were revised later, then answer the unlabeled probe entries."
        ),
        labeled_heading="Resolved log entries:",
        probe_heading="Unresolved probe entries:",
        outro=_BINARY_OUTRO,
        line_renderer=render_binary_log_line,
    ),
    TemplateFamily.CASE_LEDGER: BinaryPresentation(
        intro=(
            "Review the case ledger for two markers.\n"
            "Each row records r1, r2, and the observed state.\n"
            "Use the full ledger to infer which sign combinations were revised by the later evidence, "
            "then complete the pending rows."
        ),
        labeled_heading="Confirmed ledger rows:",
        probe_heading="Pending ledger rows:",
        outro=_BINARY_OUTRO,
        line_renderer=render_binary_ledger_line,
    ),
}
