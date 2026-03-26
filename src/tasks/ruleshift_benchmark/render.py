from __future__ import annotations

from tasks.ruleshift_benchmark.protocol import (
    LABELED_ITEM_COUNT,
    TemplateFamily,
)
from tasks.ruleshift_benchmark.schema import Episode, EpisodeItem

__all__ = ["render_binary_prompt", "render_narrative_prompt"]

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
}
_NARRATIVE_INTROS = {
    TemplateFamily.CANONICAL: (
        "Two electric charges were observed interacting in the following sequence.\n"
        "Use the full sequence to infer which sign combinations were revised by the later evidence, "
        "then answer the unlabeled observations at the end."
    ),
    TemplateFamily.OBSERVATION_LOG: (
        "An observation log recorded interactions between two electric charges over time.\n"
        "Use the full log to infer which sign combinations were revised by the later evidence, "
        "then answer the unlabeled observations at the end."
    ),
}
_BINARY_OUTRO = "Return exactly 4 labels in order, one per probe. Use only attract or repel."
_NARRATIVE_OUTRO = (
    "Return your analysis as a JSON object with exactly these four fields:\n"
    '  "inferred_rule_before": your inferred rule from the pre-shift examples (string),\n'
    '  "shift_evidence": which observations indicated the rule changed (string),\n'
    '  "inferred_rule_after": your inferred rule from the post-shift examples (string),\n'
    '  "final_binary_answer": list of exactly 4 labels in probe order'
    ' (each "attract" or "repel").\n'
    "Output only the JSON object."
)


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


def render_narrative_prompt(episode: Episode) -> str:
    labeled_items, probe_items = _partition_items(episode)
    line_renderer = _narrative_line_renderer(episode.template_family)
    return "\n".join(
        (
            _NARRATIVE_INTROS[episode.template_family],
            "",
            _narrative_labeled_heading(episode.template_family),
            *(line_renderer(item) for item in labeled_items),
            "",
            _narrative_probe_heading(episode.template_family),
            *(line_renderer(item) for item in probe_items),
            "",
            _NARRATIVE_OUTRO,
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
    return _render_binary_line


def _narrative_line_renderer(
    template_family: TemplateFamily,
):
    if template_family is TemplateFamily.OBSERVATION_LOG:
        return _render_narrative_log_line
    return _render_narrative_line


def _binary_labeled_heading(template_family: TemplateFamily) -> str:
    if template_family is TemplateFamily.OBSERVATION_LOG:
        return "Resolved log entries:"
    return "Labeled examples:"


def _binary_probe_heading(template_family: TemplateFamily) -> str:
    if template_family is TemplateFamily.OBSERVATION_LOG:
        return "Unresolved probe entries:"
    return "Probes:"


def _narrative_labeled_heading(template_family: TemplateFamily) -> str:
    if template_family is TemplateFamily.OBSERVATION_LOG:
        return "Resolved log entries:"
    return "Labeled examples:"


def _narrative_probe_heading(template_family: TemplateFamily) -> str:
    if template_family is TemplateFamily.OBSERVATION_LOG:
        return "Unresolved probe entries:"
    return "Probes:"


def _render_binary_line(item: EpisodeItem) -> str:
    return (
        f"{item.position}. q1={_format_charge(item.q1)}, "
        f"q2={_format_charge(item.q2)} -> {_render_outcome(item)}"
    )


def _render_narrative_line(item: EpisodeItem) -> str:
    return (
        f"{item.position}. A {_format_charge(item.q1)} charge and a {_format_charge(item.q2)} "
        f"charge were observed to {_render_outcome(item)}."
    )


def _render_binary_log_line(item: EpisodeItem) -> str:
    return (
        f"[{item.position:02d}] q1={_format_charge(item.q1)} | "
        f"q2={_format_charge(item.q2)} | observed={_render_outcome(item)}"
    )


def _render_narrative_log_line(item: EpisodeItem) -> str:
    return (
        f"[{item.position:02d}] charges({_format_charge(item.q1)}, {_format_charge(item.q2)}) "
        f"=> observed {_render_outcome(item)}."
    )


def _render_outcome(item: EpisodeItem) -> str:
    return item.label.value if item.label is not None else "?"


def _format_charge(charge: int) -> str:
    return f"{charge:+d}"
