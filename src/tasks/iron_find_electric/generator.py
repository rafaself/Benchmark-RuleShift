from __future__ import annotations

import random

from tasks.iron_find_electric.protocol import (
    CASE_SPACE,
    LABELED_ITEM_COUNT,
    Difficulty,
    InteractionLabel,
    ItemKind,
    Phase,
    RuleName,
    Split,
    TemplateId,
    TEMPLATES,
    Transition,
)
from tasks.iron_find_electric.rules import label
from tasks.iron_find_electric.schema import (
    DIFFICULTY_VERSION,
    Episode,
    EpisodeItem,
    ProbeMetadata,
)

RULE_CHOICES: tuple[RuleName, ...] = (RuleName.R_STD, RuleName.R_INV)
TEMPLATE_CHOICES: tuple[TemplateId, ...] = (TemplateId.T1, TemplateId.T2)
_PROBE_LABEL_ORDER: tuple[InteractionLabel, ...] = (
    InteractionLabel.ATTRACT,
    InteractionLabel.REPEL,
)
_PROBE_SIGN_PATTERN_ORDER: tuple[str, ...] = ("++", "--", "+-", "-+")


def _is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _sample_pairs(
    rng: random.Random,
    total_items: int,
) -> tuple[tuple[int, int], ...]:
    return tuple(rng.sample(CASE_SPACE, k=total_items))


def _probe_sign_pattern(q1: int, q2: int) -> str:
    if q1 > 0 and q2 > 0:
        return "++"
    if q1 < 0 and q2 < 0:
        return "--"
    if q1 > 0 and q2 < 0:
        return "+-"
    return "-+"


def _build_items(
    sampled_pairs: tuple[tuple[int, int], ...],
    pre_count: int,
    rule_a: RuleName,
    rule_b: RuleName,
) -> tuple[EpisodeItem, ...]:
    items: list[EpisodeItem] = []
    for position, (q1, q2) in enumerate(sampled_pairs, start=1):
        if position <= pre_count:
            active_rule = rule_a
            phase = Phase.PRE
            kind = ItemKind.LABELED
            item_label = label(active_rule, q1, q2)
        elif position <= LABELED_ITEM_COUNT:
            active_rule = rule_b
            phase = Phase.POST
            kind = ItemKind.LABELED
            item_label = label(active_rule, q1, q2)
        else:
            phase = Phase.POST
            kind = ItemKind.PROBE
            item_label = None

        items.append(
            EpisodeItem(
                position=position,
                phase=phase,
                kind=kind,
                q1=q1,
                q2=q2,
                label=item_label,
            )
        )

    return tuple(items)


def _build_probe_targets(
    items: tuple[EpisodeItem, ...],
    rule_b: RuleName,
) -> tuple[InteractionLabel, ...]:
    return tuple(label(rule_b, item.q1, item.q2) for item in items[LABELED_ITEM_COUNT:])


def _build_probe_metadata(
    items: tuple[EpisodeItem, ...],
    rule_a: RuleName,
    rule_b: RuleName,
) -> tuple[ProbeMetadata, ...]:
    probe_items = items[LABELED_ITEM_COUNT:]
    return tuple(
        ProbeMetadata(
            position=item.position,
            is_disagreement_probe=label(RuleName.R_STD, item.q1, item.q2)
            != label(RuleName.R_INV, item.q1, item.q2),
            old_rule_label=label(rule_a, item.q1, item.q2),
            new_rule_label=label(rule_b, item.q1, item.q2),
        )
        for item in probe_items
    )


def _build_contradiction_count_post(
    items: tuple[EpisodeItem, ...],
    pre_count: int,
    rule_a: RuleName,
    rule_b: RuleName,
) -> int:
    return sum(
        label(rule_a, item.q1, item.q2) != label(rule_b, item.q1, item.q2)
        for item in items[pre_count:LABELED_ITEM_COUNT]
    )


def _build_probe_label_counts(
    probe_targets: tuple[InteractionLabel, ...],
) -> tuple[tuple[InteractionLabel, int], ...]:
    return tuple(
        (target_label, probe_targets.count(target_label))
        for target_label in _PROBE_LABEL_ORDER
    )


def _build_probe_sign_pattern_counts(
    items: tuple[EpisodeItem, ...],
) -> tuple[tuple[str, int], ...]:
    probe_items = items[LABELED_ITEM_COUNT:]
    return tuple(
        (
            pattern,
            sum(_probe_sign_pattern(item.q1, item.q2) == pattern for item in probe_items),
        )
        for pattern in _PROBE_SIGN_PATTERN_ORDER
    )


def _has_both_probe_labels(probe_targets: tuple[InteractionLabel, ...]) -> bool:
    return len(set(probe_targets)) >= 2


def _is_valid_candidate(
    contradiction_count_post: int,
    probe_targets: tuple[InteractionLabel, ...],
) -> bool:
    return contradiction_count_post >= 1 and _has_both_probe_labels(probe_targets)


def _assign_difficulty(
    template_id: TemplateId,
    contradiction_count_post: int,
    probe_label_counts: tuple[tuple[InteractionLabel, int], ...],
) -> Difficulty:
    has_both_probe_labels = all(count > 0 for _, count in probe_label_counts)
    if (
        template_id is TemplateId.T1
        and contradiction_count_post >= 1
        and has_both_probe_labels
    ):
        return Difficulty.EASY
    return Difficulty.MEDIUM


def generate_episode(seed: int, split: Split | str = Split.DEV) -> Episode:
    if not _is_plain_int(seed):
        raise TypeError("seed must be an int")

    rng = random.Random(seed)
    rule_a = rng.choice(RULE_CHOICES)
    rule_b = rule_a.opposite
    template_id = rng.choice(TEMPLATE_CHOICES)
    template = TEMPLATES[template_id]

    while True:
        sampled_pairs = _sample_pairs(rng, template.total_items)
        items = _build_items(sampled_pairs, template.pre_count, rule_a, rule_b)
        probe_targets = _build_probe_targets(items, rule_b)
        contradiction_count_post = _build_contradiction_count_post(
            items,
            template.pre_count,
            rule_a,
            rule_b,
        )
        if _is_valid_candidate(contradiction_count_post, probe_targets):
            break

    probe_label_counts = _build_probe_label_counts(probe_targets)
    probe_sign_pattern_counts = _build_probe_sign_pattern_counts(items)
    probe_metadata = _build_probe_metadata(items, rule_a, rule_b)

    return Episode(
        episode_id=f"ife-r3-{seed}",
        split=split,
        difficulty=_assign_difficulty(
            template_id,
            contradiction_count_post,
            probe_label_counts,
        ),
        template_id=template_id,
        rule_A=rule_a,
        rule_B=rule_b,
        transition=Transition.from_rules(rule_a, rule_b),
        pre_count=template.pre_count,
        post_labeled_count=template.post_labeled_count,
        shift_after_position=template.shift_after_position,
        contradiction_count_post=contradiction_count_post,
        items=items,
        probe_targets=probe_targets,
        probe_label_counts=probe_label_counts,
        probe_sign_pattern_counts=probe_sign_pattern_counts,
        probe_metadata=probe_metadata,
        difficulty_version=DIFFICULTY_VERSION,
    )
