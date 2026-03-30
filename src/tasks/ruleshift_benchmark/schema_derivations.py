from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from tasks.ruleshift_benchmark.protocol import (
    Difficulty,
    DifficultyProfileId,
    FactorLevel,
    InteractionLabel,
    LABELED_ITEM_COUNT,
    RuleName,
    parse_factor_level,
)
from tasks.ruleshift_benchmark.rules import label

PROBE_LABEL_ORDER: Final[tuple[InteractionLabel, ...]] = (
    InteractionLabel.ATTRACT,
    InteractionLabel.REPEL,
)
PROBE_SIGN_PATTERN_ORDER: Final[tuple[str, ...]] = ("++", "--", "+-", "-+")
_FACTOR_LEVEL_SCORE: Final[dict[FactorLevel, int]] = {
    FactorLevel.LOW: 0,
    FactorLevel.MEDIUM: 1,
    FactorLevel.HIGH: 2,
}
_EASY_DIFFICULTY_SCORE_MAX: Final[int] = 3
_HARD_DIFFICULTY_SCORE_MIN: Final[int] = 6


def probe_sign_pattern(q1: int, q2: int) -> str:
    if q1 > 0 and q2 > 0:
        return "++"
    if q1 < 0 and q2 < 0:
        return "--"
    if q1 > 0 and q2 < 0:
        return "+-"
    return "-+"


def build_updated_sign_patterns(
    post_labeled_items: tuple["EpisodeItem", ...],
) -> frozenset[str]:
    return frozenset(probe_sign_pattern(item.q1, item.q2) for item in post_labeled_items)


def has_mixed_polarity_sign_patterns(patterns: frozenset[str]) -> bool:
    return (
        len(patterns) == 2
        and any(_is_same_sign_pattern(pattern) for pattern in patterns)
        and any(not _is_same_sign_pattern(pattern) for pattern in patterns)
    )


def build_effective_probe_targets(
    probe_items: tuple["EpisodeItem", ...],
    rule_a: RuleName,
    rule_b: RuleName,
    updated_sign_patterns: frozenset[str],
) -> tuple[InteractionLabel, ...]:
    return tuple(
        _effective_probe_label(
            rule_a,
            rule_b,
            item.q1,
            item.q2,
            updated_sign_patterns,
        )
        for item in probe_items
    )


def is_global_rule_probe_block(
    probe_items: tuple["EpisodeItem", ...],
    probe_targets: tuple[InteractionLabel, ...],
    rule_name: RuleName,
) -> bool:
    return probe_targets == tuple(
        label(rule_name, item.q1, item.q2) for item in probe_items
    )


def build_probe_label_counts(
    probe_targets: tuple[InteractionLabel, ...],
) -> tuple[tuple[InteractionLabel, int], ...]:
    return tuple(
        (target_label, probe_targets.count(target_label))
        for target_label in PROBE_LABEL_ORDER
    )


def build_probe_sign_pattern_counts(
    probe_items: tuple["EpisodeItem", ...],
) -> tuple[tuple[str, int], ...]:
    return tuple(
        (
            pattern,
            sum(probe_sign_pattern(item.q1, item.q2) == pattern for item in probe_items),
        )
        for pattern in PROBE_SIGN_PATTERN_ORDER
    )


def build_contradiction_count_post(
    labeled_items: tuple["EpisodeItem", ...],
    pre_count: int,
    rule_a: RuleName,
    rule_b: RuleName,
) -> int:
    return sum(
        label(rule_a, item.q1, item.q2) != label(rule_b, item.q1, item.q2)
        for item in labeled_items[pre_count:]
    )


def has_both_probe_labels(probe_targets: tuple[InteractionLabel, ...]) -> bool:
    return len(set(probe_targets)) >= 2


def build_probe_metadata(
    probe_items: tuple["EpisodeItem", ...],
    rule_a: RuleName,
    rule_b: RuleName,
) -> tuple["ProbeMetadata", ...]:
    from tasks.ruleshift_benchmark.schema import ProbeMetadata

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


@dataclass(frozen=True, slots=True)
class DifficultyFactors:
    conflict_strength: FactorLevel
    post_shift_evidence_clarity: FactorLevel
    probe_ambiguity: FactorLevel
    evidence_to_final_probe_distance: FactorLevel
    pre_shift_distractor_pressure: FactorLevel

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "conflict_strength",
            parse_factor_level(self.conflict_strength),
        )
        object.__setattr__(
            self,
            "post_shift_evidence_clarity",
            parse_factor_level(self.post_shift_evidence_clarity),
        )
        object.__setattr__(
            self,
            "probe_ambiguity",
            parse_factor_level(self.probe_ambiguity),
        )
        object.__setattr__(
            self,
            "evidence_to_final_probe_distance",
            parse_factor_level(self.evidence_to_final_probe_distance),
        )
        object.__setattr__(
            self,
            "pre_shift_distractor_pressure",
            parse_factor_level(self.pre_shift_distractor_pressure),
        )


def derive_difficulty_factors(
    items: tuple["EpisodeItem", ...],
    pre_count: int,
) -> DifficultyFactors:
    labeled_items = items[:LABELED_ITEM_COUNT]
    post_labeled_items = labeled_items[pre_count:]
    probe_items = items[LABELED_ITEM_COUNT:]
    pre_sign_patterns = frozenset(
        probe_sign_pattern(item.q1, item.q2) for item in labeled_items[:pre_count]
    )
    updated_sign_patterns = build_updated_sign_patterns(post_labeled_items)
    switch_count = _count_probe_rule_switches(probe_items, updated_sign_patterns)
    retained_overlap_count = _retained_probe_overlap_count(
        probe_items,
        updated_sign_patterns=updated_sign_patterns,
        pre_sign_patterns=pre_sign_patterns,
    )
    probe_pre_overlap_count = _probe_pattern_overlap_count(probe_items, pre_sign_patterns)

    if switch_count <= 1:
        conflict_strength = FactorLevel.LOW
    elif switch_count == 2:
        conflict_strength = FactorLevel.MEDIUM
    else:
        conflict_strength = FactorLevel.HIGH

    if retained_overlap_count == 0:
        probe_ambiguity = FactorLevel.LOW
    elif retained_overlap_count == 1:
        probe_ambiguity = FactorLevel.MEDIUM
    else:
        probe_ambiguity = FactorLevel.HIGH

    if probe_pre_overlap_count <= 1:
        distractor_pressure = FactorLevel.LOW
    elif probe_pre_overlap_count == 2:
        distractor_pressure = FactorLevel.MEDIUM
    else:
        distractor_pressure = FactorLevel.HIGH

    return DifficultyFactors(
        conflict_strength=conflict_strength,
        post_shift_evidence_clarity=_clarity_factor_for_probe_block(
            probe_items=probe_items,
            post_labeled_items=post_labeled_items,
            updated_sign_patterns=updated_sign_patterns,
        ),
        probe_ambiguity=probe_ambiguity,
        evidence_to_final_probe_distance=_distance_factor_for_final_probe(
            probe_items=probe_items,
            post_labeled_items=post_labeled_items,
            updated_sign_patterns=updated_sign_patterns,
        ),
        pre_shift_distractor_pressure=distractor_pressure,
    )


def derive_difficulty_profile(
    factors: DifficultyFactors,
) -> tuple[Difficulty, DifficultyProfileId]:
    clarity_load = {
        FactorLevel.HIGH: FactorLevel.LOW,
        FactorLevel.MEDIUM: FactorLevel.MEDIUM,
        FactorLevel.LOW: FactorLevel.HIGH,
    }[factors.post_shift_evidence_clarity]
    difficulty_score = sum(
        _FACTOR_LEVEL_SCORE[level]
        for level in (
            factors.conflict_strength,
            clarity_load,
            factors.probe_ambiguity,
            factors.evidence_to_final_probe_distance,
            factors.pre_shift_distractor_pressure,
        )
    )

    if difficulty_score <= _EASY_DIFFICULTY_SCORE_MAX:
        return Difficulty.EASY, DifficultyProfileId.EASY_ANCHORED

    if difficulty_score >= _HARD_DIFFICULTY_SCORE_MIN:
        return Difficulty.HARD, DifficultyProfileId.HARD_INTERLEAVED

    return Difficulty.MEDIUM, DifficultyProfileId.MEDIUM_BALANCED


def _is_same_sign_pattern(pattern: str) -> bool:
    return pattern in {"++", "--"}


def _effective_probe_label(
    rule_a: RuleName,
    rule_b: RuleName,
    q1: int,
    q2: int,
    updated_sign_patterns: frozenset[str],
) -> InteractionLabel:
    active_rule = (
        rule_b
        if probe_sign_pattern(q1, q2) in updated_sign_patterns
        else rule_a
    )
    return label(active_rule, q1, q2)


def _count_probe_rule_switches(
    probe_items: tuple["EpisodeItem", ...],
    updated_sign_patterns: frozenset[str],
) -> int:
    active_rules = tuple(
        "new"
        if probe_sign_pattern(item.q1, item.q2) in updated_sign_patterns
        else "old"
        for item in probe_items
    )
    return sum(
        left != right for left, right in zip(active_rules, active_rules[1:])
    )


def _build_post_sign_pattern_counts(
    post_labeled_items: tuple["EpisodeItem", ...],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in post_labeled_items:
        pattern = probe_sign_pattern(item.q1, item.q2)
        counts[pattern] = counts.get(pattern, 0) + 1
    return counts


def _probe_pattern_overlap_count(
    probe_items: tuple["EpisodeItem", ...],
    patterns: frozenset[str],
) -> int:
    return sum(
        probe_sign_pattern(item.q1, item.q2) in patterns
        for item in probe_items
    )


def _retained_probe_overlap_count(
    probe_items: tuple["EpisodeItem", ...],
    *,
    updated_sign_patterns: frozenset[str],
    pre_sign_patterns: frozenset[str],
) -> int:
    return sum(
        probe_sign_pattern(item.q1, item.q2) not in updated_sign_patterns
        and probe_sign_pattern(item.q1, item.q2) in pre_sign_patterns
        for item in probe_items
    )


def _distance_factor_for_final_probe(
    *,
    probe_items: tuple["EpisodeItem", ...],
    post_labeled_items: tuple["EpisodeItem", ...],
    updated_sign_patterns: frozenset[str],
) -> FactorLevel:
    final_probe = probe_items[-1]
    final_pattern = probe_sign_pattern(final_probe.q1, final_probe.q2)
    if final_pattern not in updated_sign_patterns:
        return FactorLevel.HIGH

    matching_positions = tuple(
        item.position
        for item in post_labeled_items
        if probe_sign_pattern(item.q1, item.q2) == final_pattern
    )
    last_position = max(matching_positions)
    if last_position == LABELED_ITEM_COUNT:
        return FactorLevel.LOW
    if last_position == LABELED_ITEM_COUNT - 1:
        return FactorLevel.MEDIUM
    return FactorLevel.HIGH


def _clarity_factor_for_probe_block(
    *,
    probe_items: tuple["EpisodeItem", ...],
    post_labeled_items: tuple["EpisodeItem", ...],
    updated_sign_patterns: frozenset[str],
) -> FactorLevel:
    post_pattern_counts = _build_post_sign_pattern_counts(post_labeled_items)
    final_pattern = probe_sign_pattern(probe_items[-1].q1, probe_items[-1].q2)
    if max(post_pattern_counts.values()) >= 2 and final_pattern in updated_sign_patterns:
        return FactorLevel.HIGH
    if max(post_pattern_counts.values()) >= 2:
        return FactorLevel.MEDIUM
    return FactorLevel.LOW
