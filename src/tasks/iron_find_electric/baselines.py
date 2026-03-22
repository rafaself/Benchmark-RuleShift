from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import random
from typing import TypeAlias

from core.parser import ParsedPrediction, ParseStatus
from tasks.iron_find_electric.protocol import (
    LABELED_ITEM_COUNT,
    InteractionLabel,
    RuleName,
)
from tasks.iron_find_electric.rules import label
from tasks.iron_find_electric.schema import Episode, EpisodeItem

__all__ = [
    "BaselinePrediction",
    "BaselineFn",
    "BaselineEpisodeResult",
    "BaselineRunResult",
    "random_baseline",
    "never_update_baseline",
    "last_evidence_baseline",
    "physics_prior_baseline",
    "template_position_baseline",
    "run_baseline",
    "run_baselines",
]

BaselinePrediction: TypeAlias = tuple[
    InteractionLabel,
    InteractionLabel,
    InteractionLabel,
    InteractionLabel,
]
BaselineFn: TypeAlias = Callable[[Episode], BaselinePrediction]

_RULE_CHOICES: tuple[RuleName, ...] = (RuleName.R_STD, RuleName.R_INV)
_LABEL_CHOICES: tuple[InteractionLabel, ...] = (
    InteractionLabel.ATTRACT,
    InteractionLabel.REPEL,
)


@dataclass(frozen=True, slots=True)
class BaselineEpisodeResult:
    episode_id: str
    prediction: BaselinePrediction
    parsed_prediction: ParsedPrediction
    target: BaselinePrediction


@dataclass(frozen=True, slots=True)
class BaselineRunResult:
    baseline_name: str
    rows: tuple[BaselineEpisodeResult, ...]


def random_baseline(episode: Episode, *, seed: int = 0) -> BaselinePrediction:
    rng = random.Random(f"{seed}:{episode.episode_id}")
    return tuple(rng.choice(_LABEL_CHOICES) for _ in _probe_items(episode))  # type: ignore[return-value]


def never_update_baseline(episode: Episode) -> BaselinePrediction:
    return _predict_with_rule(
        _infer_rule_from_labeled_items(episode.items[: episode.pre_count]),
        episode,
    )


def last_evidence_baseline(episode: Episode) -> BaselinePrediction:
    return _predict_with_rule(
        _infer_rule_from_labeled_items((episode.items[LABELED_ITEM_COUNT - 1],)),
        episode,
    )


def physics_prior_baseline(episode: Episode) -> BaselinePrediction:
    return _predict_with_rule(RuleName.R_STD, episode)


def template_position_baseline(episode: Episode) -> BaselinePrediction:
    post_labeled_labels = tuple(
        item.label for item in episode.items[episode.pre_count:LABELED_ITEM_COUNT]
    )
    repeated_labels = tuple(
        post_labeled_labels[index % len(post_labeled_labels)] for index in range(4)
    )
    return repeated_labels  # type: ignore[return-value]


def run_baseline(
    name: str,
    baseline: BaselineFn,
    episodes: Iterable[Episode],
) -> BaselineRunResult:
    normalized_episodes = tuple(episodes)
    return BaselineRunResult(
        baseline_name=name,
        rows=tuple(_build_episode_result(baseline, episode) for episode in normalized_episodes),
    )


def run_baselines(
    episodes: Iterable[Episode],
    baselines: Mapping[str, BaselineFn] | Iterable[tuple[str, BaselineFn]],
) -> tuple[BaselineRunResult, ...]:
    normalized_episodes = tuple(episodes)
    normalized_baselines = (
        tuple(baselines.items()) if isinstance(baselines, Mapping) else tuple(baselines)
    )
    return tuple(
        run_baseline(name, baseline, normalized_episodes)
        for name, baseline in normalized_baselines
    )


def _build_episode_result(
    baseline: BaselineFn,
    episode: Episode,
) -> BaselineEpisodeResult:
    prediction = baseline(episode)
    return BaselineEpisodeResult(
        episode_id=episode.episode_id,
        prediction=prediction,
        parsed_prediction=ParsedPrediction(
            labels=prediction,
            status=ParseStatus.VALID,
        ),
        target=episode.probe_targets,  # type: ignore[arg-type]
    )


def _probe_items(episode: Episode) -> tuple[EpisodeItem, ...]:
    return episode.items[LABELED_ITEM_COUNT:]


def _predict_with_rule(rule_name: RuleName, episode: Episode) -> BaselinePrediction:
    return tuple(label(rule_name, item.q1, item.q2) for item in _probe_items(episode))  # type: ignore[return-value]


def _infer_rule_from_labeled_items(items: Iterable[EpisodeItem]) -> RuleName:
    normalized_items = tuple(items)
    for rule_name in _RULE_CHOICES:
        if all(item.label == label(rule_name, item.q1, item.q2) for item in normalized_items):
            return rule_name
    return RuleName.R_STD
