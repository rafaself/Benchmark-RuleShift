from __future__ import annotations

from tasks.ruleshift_benchmark.protocol import (
    LABELED_ITEM_COUNT,
    TemplateFamily,
)
from tasks.ruleshift_benchmark.schema import Episode, EpisodeItem

__all__ = ["render_binary_prompt"]

_BINARY_INTROS = {
    TemplateFamily.CANONICAL: (
        "You are given labeled interactions between two electric charges.\n"
        "Each labeled line shows q1, q2, and the observed result.\n"
        "Use the full sequence to infer which sign combinations were revised by the later evidence, "
        "then answer the final unlabeled cases."
    ),
    TemplateFamily.OBSERVATION_LOG: (
        "Review the observation log for interactions between two electric charges.\n"
        "Each entry records q1, q2, and the observed outcome.\n"
        "Use the full log to infer which sign combinations were revised later, then answer the unlabeled probe entries."
    ),
    TemplateFamily.CASE_LEDGER: (
        "Review the case ledger for interactions between two electric charges.\n"
        "Each row records the charge pair and the observed result.\n"
        "Use the full ledger to infer which sign combinations were revised by the later evidence, "
        "then complete the pending rows."
    ),
}
_BINARY_OUTRO = "Return exactly 4 labels in order, one per probe. Use only attract or repel."


def render_binary_prompt(episode: Episode) -> str:
    labeled_items, probe_items = _partition_items(episode)
    line_renderer = _binary_line_renderer(episode.template_family)
    return "\n".join(
        (
            _BINARY_INTROS[episode.template_family],
            "",
            _binary_labeled_heading(episode.template_family),
            *(line_renderer(item) for item in labeled_items),
            "",
            _binary_probe_heading(episode.template_family),
            *(line_renderer(item) for item in probe_items),
            "",
            _BINARY_OUTRO,
        )
    )


def _partition_items(episode: Episode) -> tuple[tuple[EpisodeItem, ...], tuple[EpisodeItem, ...]]:
    labeled_items = episode.items[:LABELED_ITEM_COUNT]
    probe_items = episode.items[LABELED_ITEM_COUNT:]
    return labeled_items, probe_items


def _binary_line_renderer(
    template_family: TemplateFamily,
):
    if template_family is TemplateFamily.OBSERVATION_LOG:
        return _render_binary_log_line
    if template_family is TemplateFamily.CASE_LEDGER:
        return _render_binary_ledger_line
    return _render_binary_line


def _binary_labeled_heading(template_family: TemplateFamily) -> str:
    if template_family is TemplateFamily.OBSERVATION_LOG:
        return "Resolved log entries:"
    if template_family is TemplateFamily.CASE_LEDGER:
        return "Confirmed ledger rows:"
    return "Labeled examples:"


def _binary_probe_heading(template_family: TemplateFamily) -> str:
    if template_family is TemplateFamily.OBSERVATION_LOG:
        return "Unresolved probe entries:"
    if template_family is TemplateFamily.CASE_LEDGER:
        return "Pending ledger rows:"
    return "Probes:"


def _render_binary_line(item: EpisodeItem) -> str:
    return (
        f"{item.position}. q1={_format_charge(item.q1)}, "
        f"q2={_format_charge(item.q2)} -> {_render_outcome(item)}"
    )


def _render_binary_log_line(item: EpisodeItem) -> str:
    return (
        f"[{item.position:02d}] q1={_format_charge(item.q1)} | "
        f"q2={_format_charge(item.q2)} | observed={_render_outcome(item)}"
    )


def _render_binary_ledger_line(item: EpisodeItem) -> str:
    return (
        f"row {item.position:02d} | pair=({_format_charge(item.q1)}, {_format_charge(item.q2)}) "
        f"| result={_render_outcome(item)}"
    )


def _render_outcome(item: EpisodeItem) -> str:
    return item.label.value if item.label is not None else "?"


def _format_charge(charge: int) -> str:
    return f"{charge:+d}"
