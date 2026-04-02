from __future__ import annotations

from tasks.ruleshift_benchmark.protocol import (
    LABELED_ITEM_COUNT,
)
from tasks.ruleshift_benchmark.presentation import BINARY_PRESENTATIONS
from tasks.ruleshift_benchmark.schema import Episode, EpisodeItem

__all__ = ["render_binary_prompt"]


def render_binary_prompt(episode: Episode) -> str:
    labeled_items, probe_items = _partition_items(episode)
    presentation = BINARY_PRESENTATIONS[episode.template_family]
    return "\n".join(
        (
            presentation.intro,
            "",
            presentation.labeled_heading,
            *(presentation.line_renderer(item) for item in labeled_items),
            "",
            presentation.probe_heading,
            *(presentation.line_renderer(item) for item in probe_items),
            "",
            presentation.outro,
        )
    )


def _partition_items(episode: Episode) -> tuple[tuple[EpisodeItem, ...], tuple[EpisodeItem, ...]]:
    labeled_items = episode.items[:LABELED_ITEM_COUNT]
    probe_items = episode.items[LABELED_ITEM_COUNT:]
    return labeled_items, probe_items
