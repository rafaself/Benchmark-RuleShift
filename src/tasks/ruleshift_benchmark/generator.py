from __future__ import annotations

import random

from tasks.ruleshift_benchmark.protocol import (
    CASE_SPACE,
    LABELED_ITEM_COUNT,
    Difficulty,
    InteractionLabel,
    ItemKind,
    Phase,
    RuleName,
    Split,
    TemplateFamily,
    TemplateId,
    TEMPLATES,
    Transition,
)
from tasks.ruleshift_benchmark.rules import label
from tasks.ruleshift_benchmark.schema import (
    DIFFICULTY_VERSION,
    Episode,
    EpisodeItem,
    ProbeMetadata,
    _build_effective_probe_targets,
    _build_updated_sign_patterns,
    derive_difficulty_factors,
    _has_mixed_polarity_sign_patterns,
    _is_global_rule_probe_block,
    derive_difficulty_profile,
)

TEMPLATE_CHOICES: tuple[TemplateId, ...] = (
    TemplateId.T1,
    TemplateId.T2,
    TemplateId.T3,
)
TEMPLATE_FAMILY_CHOICES: tuple[TemplateFamily, ...] = (
    TemplateFamily.CANONICAL,
    TemplateFamily.OBSERVATION_LOG,
    TemplateFamily.CASE_LEDGER,
)
TRANSITION_CHOICES: tuple[Transition, ...] = (
    Transition.R_STD_TO_R_INV,
    Transition.R_INV_TO_R_STD,
)
_PROBE_LABEL_ORDER: tuple[InteractionLabel, ...] = (
    InteractionLabel.ATTRACT,
    InteractionLabel.REPEL,
)
_PROBE_SIGN_PATTERN_ORDER: tuple[str, ...] = ("++", "--", "+-", "-+")
_DIFFICULTY_ORDER: tuple[Difficulty, ...] = (
    Difficulty.EASY,
    Difficulty.MEDIUM,
    Difficulty.HARD,
)


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
    pre_count: int,
    rule_a: RuleName,
    rule_b: RuleName,
) -> tuple[InteractionLabel, ...]:
    probe_items = items[LABELED_ITEM_COUNT:]
    updated_sign_patterns = _build_updated_sign_patterns(items[pre_count:LABELED_ITEM_COUNT])
    return _build_effective_probe_targets(
        probe_items,
        rule_a,
        rule_b,
        updated_sign_patterns,
    )


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


def _has_full_probe_sign_pattern_coverage(items: tuple[EpisodeItem, ...]) -> bool:
    return _build_probe_sign_pattern_counts(items) == tuple(
        (pattern, 1) for pattern in _PROBE_SIGN_PATTERN_ORDER
    )


def _is_valid_candidate(
    contradiction_count_post: int,
    items: tuple[EpisodeItem, ...],
    pre_count: int,
    rule_a: RuleName,
    rule_b: RuleName,
    probe_targets: tuple[InteractionLabel, ...],
) -> bool:
    probe_items = items[LABELED_ITEM_COUNT:]
    updated_sign_patterns = _build_updated_sign_patterns(items[pre_count:LABELED_ITEM_COUNT])
    return (
        contradiction_count_post >= 1
        and _has_mixed_polarity_sign_patterns(updated_sign_patterns)
        and _has_full_probe_sign_pattern_coverage(items)
        and _has_both_probe_labels(probe_targets)
        and not _is_global_rule_probe_block(probe_items, probe_targets, rule_a)
        and not _is_global_rule_probe_block(probe_items, probe_targets, rule_b)
    )


def _target_difficulty_for_seed(seed: int) -> Difficulty:
    return _DIFFICULTY_ORDER[seed % len(_DIFFICULTY_ORDER)]


def _target_template_for_seed(seed: int) -> TemplateId:
    return TEMPLATE_CHOICES[(seed // len(_DIFFICULTY_ORDER)) % len(TEMPLATE_CHOICES)]


def _target_template_family_for_seed(seed: int) -> TemplateFamily:
    stride = len(_DIFFICULTY_ORDER) * len(TEMPLATE_CHOICES)
    return TEMPLATE_FAMILY_CHOICES[(seed // stride) % len(TEMPLATE_FAMILY_CHOICES)]


def _target_transition_for_seed(seed: int) -> Transition:
    stride = len(_DIFFICULTY_ORDER) * len(TEMPLATE_CHOICES) * len(
        TEMPLATE_FAMILY_CHOICES
    )
    return TRANSITION_CHOICES[(seed // stride) % len(TRANSITION_CHOICES)]


def generate_episode(seed: int, split: Split | str = Split.DEV) -> Episode:
    if not _is_plain_int(seed):
        raise TypeError("seed must be an int")

    rng = random.Random(seed)
    target_difficulty = _target_difficulty_for_seed(seed)
    target_template_id = _target_template_for_seed(seed)
    target_template_family = _target_template_family_for_seed(seed)
    target_transition = _target_transition_for_seed(seed)
    rule_a = (
        RuleName.R_STD
        if target_transition is Transition.R_STD_TO_R_INV
        else RuleName.R_INV
    )
    rule_b = rule_a.opposite
    template_id = target_template_id
    template_family = target_template_family
    template = TEMPLATES[template_id]

    while True:
        sampled_pairs = _sample_pairs(rng, template.total_items)
        items = _build_items(sampled_pairs, template.pre_count, rule_a, rule_b)
        probe_targets = _build_probe_targets(items, template.pre_count, rule_a, rule_b)
        contradiction_count_post = _build_contradiction_count_post(
            items,
            template.pre_count,
            rule_a,
            rule_b,
        )
        if _is_valid_candidate(
            contradiction_count_post,
            items,
            template.pre_count,
            rule_a,
            rule_b,
            probe_targets,
        ):
            difficulty_factors = derive_difficulty_factors(items, template.pre_count)
            difficulty, difficulty_profile_id = derive_difficulty_profile(
                difficulty_factors
            )
            if difficulty is target_difficulty:
                break
    probe_label_counts = _build_probe_label_counts(probe_targets)
    probe_sign_pattern_counts = _build_probe_sign_pattern_counts(items)
    probe_metadata = _build_probe_metadata(items, rule_a, rule_b)

    return Episode(
        episode_id=f"ife-r13-{seed}",
        split=split,
        difficulty=difficulty,
        template_id=template_id,
        template_family=template_family,
        rule_A=rule_a,
        rule_B=rule_b,
        transition=Transition.from_rules(rule_a, rule_b),
        pre_count=template.pre_count,
        post_labeled_count=template.post_labeled_count,
        shift_after_position=template.shift_after_position,
        contradiction_count_post=contradiction_count_post,
        difficulty_profile_id=difficulty_profile_id,
        difficulty_factors=difficulty_factors,
        items=items,
        probe_targets=probe_targets,
        probe_label_counts=probe_label_counts,
        probe_sign_pattern_counts=probe_sign_pattern_counts,
        probe_metadata=probe_metadata,
        difficulty_version=DIFFICULTY_VERSION,
    )
