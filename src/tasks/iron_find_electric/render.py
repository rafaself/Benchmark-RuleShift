from __future__ import annotations

from tasks.iron_find_electric.protocol import LABELED_ITEM_COUNT
from tasks.iron_find_electric.schema import Episode, EpisodeItem

__all__ = ["render_binary_prompt", "render_narrative_prompt"]

_BINARY_INTRO = (
    "You are given labeled interactions between two electric charges.\n"
    "Each labeled line shows q1, q2, and the observed result.\n"
    "Use the full sequence to infer which sign combinations were revised by the later evidence, "
    "then answer the final unlabeled cases."
)
_NARRATIVE_INTRO = (
    "Two electric charges were observed interacting in the following sequence.\n"
    "Use the full sequence to infer which sign combinations were revised by the later evidence, "
    "then answer the unlabeled observations at the end."
)
_BINARY_OUTRO = (
    "Return exactly 4 labels in order, one per probe. "
    "Use only attract or repel."
)
_NARRATIVE_OUTRO = (
    "Give brief reasoning, then write a final line in the form "
    "'Final labels: label1, label2, label3, label4'. "
    "Use exactly 4 labels in order, one per probe. "
    "Each label must be either attract or repel."
)


def render_binary_prompt(episode: Episode) -> str:
    labeled_items, probe_items = _partition_items(episode)
    return "\n".join(
        (
            _BINARY_INTRO,
            "",
            "Labeled examples:",
            *(_render_binary_line(item) for item in labeled_items),
            "",
            "Probes:",
            *(_render_binary_line(item) for item in probe_items),
            "",
            _BINARY_OUTRO,
        )
    )


def render_narrative_prompt(episode: Episode) -> str:
    labeled_items, probe_items = _partition_items(episode)
    return "\n".join(
        (
            _NARRATIVE_INTRO,
            "",
            "Labeled examples:",
            *(_render_narrative_line(item) for item in labeled_items),
            "",
            "Probes:",
            *(_render_narrative_line(item) for item in probe_items),
            "",
            _NARRATIVE_OUTRO,
        )
    )


def _partition_items(episode: Episode) -> tuple[tuple[EpisodeItem, ...], tuple[EpisodeItem, ...]]:
    labeled_items = episode.items[:LABELED_ITEM_COUNT]
    probe_items = episode.items[LABELED_ITEM_COUNT:]
    return labeled_items, probe_items


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


def _render_outcome(item: EpisodeItem) -> str:
    return item.label.value if item.label is not None else "?"


def _format_charge(charge: int) -> str:
    return f"{charge:+d}"
