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
    DEFAULT_GENERATION_MAX_ATTEMPTS,
    DIFFICULTY_VERSION,
    PROBE_SIGN_PATTERN_ORDER,
    Episode,
    EpisodeItem,
    ProbeMetadata,
    build_contradiction_count_post,
    build_effective_probe_targets,
    build_probe_label_counts,
    build_probe_metadata,
    build_probe_sign_pattern_counts,
    build_updated_sign_patterns,
    derive_difficulty_factors,
    derive_difficulty_profile,
    has_both_probe_labels,
    has_mixed_polarity_sign_patterns,
    is_global_rule_probe_block,
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
    updated_sign_patterns = build_updated_sign_patterns(items[pre_count:LABELED_ITEM_COUNT])
    return build_effective_probe_targets(
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
    return build_probe_metadata(
        probe_items,
        rule_a,
        rule_b,
    )


def _has_full_probe_sign_pattern_coverage(items: tuple[EpisodeItem, ...]) -> bool:
    return build_probe_sign_pattern_counts(items[LABELED_ITEM_COUNT:]) == tuple(
        (pattern, 1) for pattern in PROBE_SIGN_PATTERN_ORDER
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
    updated_sign_patterns = build_updated_sign_patterns(items[pre_count:LABELED_ITEM_COUNT])
    return (
        contradiction_count_post >= 1
        and has_mixed_polarity_sign_patterns(updated_sign_patterns)
        and _has_full_probe_sign_pattern_coverage(items)
        and has_both_probe_labels(probe_targets)
        and not is_global_rule_probe_block(probe_items, probe_targets, rule_a)
        and not is_global_rule_probe_block(probe_items, probe_targets, rule_b)
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


def generate_episode(
    seed: int,
    split: Split | str = Split.DEV,
    *,
    max_attempts: int = DEFAULT_GENERATION_MAX_ATTEMPTS,
) -> Episode:
    if not _is_plain_int(seed):
        raise TypeError("seed must be an int")
    if not _is_plain_int(max_attempts):
        raise TypeError("max_attempts must be an int")
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

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

    for _attempt in range(max_attempts):
        sampled_pairs = _sample_pairs(rng, template.total_items)
        items = _build_items(sampled_pairs, template.pre_count, rule_a, rule_b)
        probe_targets = _build_probe_targets(items, template.pre_count, rule_a, rule_b)
        contradiction_count_post = build_contradiction_count_post(
            items[:LABELED_ITEM_COUNT],
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
    else:
        raise RuntimeError(
            "generate_episode exhausted max_attempts "
            f"for seed={seed}, split={split}, max_attempts={max_attempts}"
        )

    probe_label_counts = build_probe_label_counts(probe_targets)
    probe_sign_pattern_counts = build_probe_sign_pattern_counts(items[LABELED_ITEM_COUNT:])
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
