from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from tasks.ruleshift_benchmark.protocol import (
    CHARGES,
    Difficulty,
    DifficultyProfileId,
    EPISODE_LENGTH,
    FactorLevel,
    LABELED_ITEM_COUNT,
    PROBE_COUNT,
    InteractionLabel,
    ItemKind,
    Phase,
    RuleName,
    Split,
    TemplateFamily,
    TemplateId,
    TEMPLATES,
    Transition,
    parse_difficulty,
    parse_difficulty_profile_id,
    parse_factor_level,
    parse_item_kind,
    parse_label,
    parse_phase,
    parse_rule,
    parse_split,
    parse_template_family,
    parse_template_id,
    parse_transition,
)
from tasks.ruleshift_benchmark.rules import label

__all__ = [
    "SPEC_VERSION",
    "GENERATOR_VERSION",
    "TEMPLATE_SET_VERSION",
    "DIFFICULTY_VERSION",
    "DifficultyFactors",
    "EpisodeItem",
    "ProbeMetadata",
    "Episode",
    "derive_difficulty_factors",
    "derive_difficulty_profile",
]

SPEC_VERSION: Final[str] = "v1"
GENERATOR_VERSION: Final[str] = "R13"
TEMPLATE_SET_VERSION: Final[str] = "v2"
DIFFICULTY_VERSION: Final[str] = "R13"

_PROBE_LABEL_ORDER: Final[tuple[InteractionLabel, ...]] = (
    InteractionLabel.ATTRACT,
    InteractionLabel.REPEL,
)
_PROBE_SIGN_PATTERN_ORDER: Final[tuple[str, ...]] = ("++", "--", "+-", "-+")


def _is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _probe_sign_pattern(q1: int, q2: int) -> str:
    if q1 > 0 and q2 > 0:
        return "++"
    if q1 < 0 and q2 < 0:
        return "--"
    if q1 > 0 and q2 < 0:
        return "+-"
    return "-+"


def _is_same_sign_pattern(pattern: str) -> bool:
    return pattern in {"++", "--"}


def _build_updated_sign_patterns(
    post_labeled_items: tuple["EpisodeItem", ...],
) -> frozenset[str]:
    return frozenset(_probe_sign_pattern(item.q1, item.q2) for item in post_labeled_items)


def _has_mixed_polarity_sign_patterns(patterns: frozenset[str]) -> bool:
    return (
        len(patterns) == 2
        and any(_is_same_sign_pattern(pattern) for pattern in patterns)
        and any(not _is_same_sign_pattern(pattern) for pattern in patterns)
    )


def _effective_probe_label(
    rule_a: RuleName,
    rule_b: RuleName,
    q1: int,
    q2: int,
    updated_sign_patterns: frozenset[str],
) -> InteractionLabel:
    active_rule = (
        rule_b
        if _probe_sign_pattern(q1, q2) in updated_sign_patterns
        else rule_a
    )
    return label(active_rule, q1, q2)


def _build_effective_probe_targets(
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


def _is_global_rule_probe_block(
    probe_items: tuple["EpisodeItem", ...],
    probe_targets: tuple[InteractionLabel, ...],
    rule_name: RuleName,
) -> bool:
    return probe_targets == tuple(
        label(rule_name, item.q1, item.q2) for item in probe_items
    )


def _build_probe_label_counts(
    probe_targets: tuple[InteractionLabel, ...],
) -> tuple[tuple[InteractionLabel, int], ...]:
    return tuple(
        (target_label, probe_targets.count(target_label))
        for target_label in _PROBE_LABEL_ORDER
    )


def _build_probe_sign_pattern_counts(
    probe_items: tuple["EpisodeItem", ...],
) -> tuple[tuple[str, int], ...]:
    return tuple(
        (
            pattern,
            sum(_probe_sign_pattern(item.q1, item.q2) == pattern for item in probe_items),
        )
        for pattern in _PROBE_SIGN_PATTERN_ORDER
    )


def _build_contradiction_count_post(
    labeled_items: tuple["EpisodeItem", ...],
    pre_count: int,
    rule_a: RuleName,
    rule_b: RuleName,
) -> int:
    return sum(
        label(rule_a, item.q1, item.q2) != label(rule_b, item.q1, item.q2)
        for item in labeled_items[pre_count:]
    )


def _has_both_probe_labels(probe_targets: tuple[InteractionLabel, ...]) -> bool:
    return len(set(probe_targets)) >= 2


def _count_probe_rule_switches(
    probe_items: tuple["EpisodeItem", ...],
    updated_sign_patterns: frozenset[str],
) -> int:
    active_rules = tuple(
        "new"
        if _probe_sign_pattern(item.q1, item.q2) in updated_sign_patterns
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
        pattern = _probe_sign_pattern(item.q1, item.q2)
        counts[pattern] = counts.get(pattern, 0) + 1
    return counts


def _probe_pattern_overlap_count(
    probe_items: tuple["EpisodeItem", ...],
    patterns: frozenset[str],
) -> int:
    return sum(
        _probe_sign_pattern(item.q1, item.q2) in patterns
        for item in probe_items
    )


def _retained_probe_overlap_count(
    probe_items: tuple["EpisodeItem", ...],
    *,
    updated_sign_patterns: frozenset[str],
    pre_sign_patterns: frozenset[str],
) -> int:
    return sum(
        _probe_sign_pattern(item.q1, item.q2) not in updated_sign_patterns
        and _probe_sign_pattern(item.q1, item.q2) in pre_sign_patterns
        for item in probe_items
    )


def _distance_factor_for_final_probe(
    *,
    probe_items: tuple["EpisodeItem", ...],
    post_labeled_items: tuple["EpisodeItem", ...],
    updated_sign_patterns: frozenset[str],
) -> FactorLevel:
    final_probe = probe_items[-1]
    final_pattern = _probe_sign_pattern(final_probe.q1, final_probe.q2)
    if final_pattern not in updated_sign_patterns:
        return FactorLevel.HIGH

    matching_positions = tuple(
        item.position
        for item in post_labeled_items
        if _probe_sign_pattern(item.q1, item.q2) == final_pattern
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
    final_pattern = _probe_sign_pattern(probe_items[-1].q1, probe_items[-1].q2)
    if max(post_pattern_counts.values()) >= 2 and final_pattern in updated_sign_patterns:
        return FactorLevel.HIGH
    if max(post_pattern_counts.values()) >= 2:
        return FactorLevel.MEDIUM
    return FactorLevel.LOW


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
        _probe_sign_pattern(item.q1, item.q2) for item in labeled_items[:pre_count]
    )
    updated_sign_patterns = _build_updated_sign_patterns(post_labeled_items)
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
    if (
        factors.conflict_strength is FactorLevel.LOW
        and factors.post_shift_evidence_clarity is FactorLevel.HIGH
        and factors.probe_ambiguity is FactorLevel.LOW
        and factors.evidence_to_final_probe_distance is FactorLevel.LOW
        and factors.pre_shift_distractor_pressure is FactorLevel.LOW
    ):
        return Difficulty.EASY, DifficultyProfileId.EASY_ANCHORED

    if (
        factors.conflict_strength is FactorLevel.HIGH
        and factors.post_shift_evidence_clarity is FactorLevel.LOW
        and factors.probe_ambiguity is FactorLevel.HIGH
        and factors.evidence_to_final_probe_distance is FactorLevel.HIGH
        and factors.pre_shift_distractor_pressure is FactorLevel.HIGH
    ):
        return Difficulty.HARD, DifficultyProfileId.HARD_INTERLEAVED

    return Difficulty.MEDIUM, DifficultyProfileId.MEDIUM_BALANCED


def _normalize_probe_label_counts(
    probe_label_counts: tuple[tuple[InteractionLabel, int], ...],
) -> tuple[tuple[InteractionLabel, int], ...]:
    normalized_probe_label_counts = tuple(probe_label_counts)
    if len(normalized_probe_label_counts) != len(_PROBE_LABEL_ORDER):
        raise ValueError("probe_label_counts must contain canonical attract/repel counts")

    normalized_pairs: list[tuple[InteractionLabel, int]] = []
    expected_labels = _PROBE_LABEL_ORDER
    for index, expected_label in enumerate(expected_labels):
        pair = normalized_probe_label_counts[index]
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise TypeError("probe_label_counts entries must be (label, count) pairs")
        label_value, count = pair
        resolved_label = parse_label(label_value)
        if resolved_label is not expected_label:
            raise ValueError("probe_label_counts must use canonical attract/repel order")
        if not _is_plain_int(count):
            raise TypeError("probe_label_counts counts must be ints")
        normalized_pairs.append((resolved_label, count))

    return tuple(normalized_pairs)


def _normalize_probe_sign_pattern_counts(
    probe_sign_pattern_counts: tuple[tuple[str, int], ...],
) -> tuple[tuple[str, int], ...]:
    normalized_probe_sign_pattern_counts = tuple(probe_sign_pattern_counts)
    if len(normalized_probe_sign_pattern_counts) != len(_PROBE_SIGN_PATTERN_ORDER):
        raise ValueError(
            "probe_sign_pattern_counts must contain canonical sign-pattern counts"
        )

    normalized_pairs: list[tuple[str, int]] = []
    for index, expected_pattern in enumerate(_PROBE_SIGN_PATTERN_ORDER):
        pair = normalized_probe_sign_pattern_counts[index]
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise TypeError(
                "probe_sign_pattern_counts entries must be (pattern, count) pairs"
            )
        pattern, count = pair
        if pattern != expected_pattern:
            raise ValueError(
                "probe_sign_pattern_counts must use canonical ++/--/+-/-+ order"
            )
        if not _is_plain_int(count):
            raise TypeError("probe_sign_pattern_counts counts must be ints")
        normalized_pairs.append((pattern, count))

    return tuple(normalized_pairs)


@dataclass(frozen=True, slots=True)
class EpisodeItem:
    position: int
    phase: Phase
    kind: ItemKind
    q1: int
    q2: int
    label: InteractionLabel | None = None

    def __post_init__(self) -> None:
        if not _is_plain_int(self.position):
            raise TypeError("position must be an int")
        if self.position < 1 or self.position > EPISODE_LENGTH:
            raise ValueError(f"position must be between 1 and {EPISODE_LENGTH}")

        object.__setattr__(self, "phase", parse_phase(self.phase))
        object.__setattr__(self, "kind", parse_item_kind(self.kind))

        if not _is_plain_int(self.q1):
            raise TypeError("q1 must be an int")
        if not _is_plain_int(self.q2):
            raise TypeError("q2 must be an int")
        if self.q1 not in CHARGES:
            raise ValueError(f"unsupported q1: {self.q1}")
        if self.q2 not in CHARGES:
            raise ValueError(f"unsupported q2: {self.q2}")

        if self.kind is ItemKind.LABELED:
            if self.label is None:
                raise ValueError("labeled items must include a label")
            object.__setattr__(self, "label", parse_label(self.label))
            return

        if self.phase is not Phase.POST:
            raise ValueError("probe items must use the post phase")
        if self.label is not None:
            raise ValueError("probe items must not include a label")


@dataclass(frozen=True, slots=True)
class ProbeMetadata:
    position: int
    is_disagreement_probe: bool
    old_rule_label: InteractionLabel
    new_rule_label: InteractionLabel

    def __post_init__(self) -> None:
        if not _is_plain_int(self.position):
            raise TypeError("position must be an int")
        if self.position <= LABELED_ITEM_COUNT or self.position > EPISODE_LENGTH:
            raise ValueError(
                f"probe metadata position must be between {LABELED_ITEM_COUNT + 1} and {EPISODE_LENGTH}"
            )
        if not isinstance(self.is_disagreement_probe, bool):
            raise TypeError("is_disagreement_probe must be a bool")

        object.__setattr__(self, "old_rule_label", parse_label(self.old_rule_label))
        object.__setattr__(self, "new_rule_label", parse_label(self.new_rule_label))


@dataclass(frozen=True, slots=True)
class Episode:
    episode_id: str
    split: Split
    difficulty: Difficulty
    template_id: TemplateId
    template_family: TemplateFamily
    rule_A: RuleName
    rule_B: RuleName
    transition: Transition
    pre_count: int
    post_labeled_count: int
    shift_after_position: int
    contradiction_count_post: int
    difficulty_profile_id: DifficultyProfileId
    difficulty_factors: DifficultyFactors
    items: tuple[EpisodeItem, ...]
    probe_targets: tuple[InteractionLabel, ...]
    probe_label_counts: tuple[tuple[InteractionLabel, int], ...]
    probe_sign_pattern_counts: tuple[tuple[str, int], ...]
    probe_metadata: tuple[ProbeMetadata, ...]
    difficulty_version: str = DIFFICULTY_VERSION
    spec_version: str = SPEC_VERSION
    generator_version: str = GENERATOR_VERSION
    template_set_version: str = TEMPLATE_SET_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, str) or not self.episode_id:
            raise ValueError("episode_id must be a non-empty string")

        object.__setattr__(self, "split", parse_split(self.split))
        object.__setattr__(self, "difficulty", parse_difficulty(self.difficulty))
        object.__setattr__(self, "template_id", parse_template_id(self.template_id))
        object.__setattr__(
            self,
            "template_family",
            parse_template_family(self.template_family),
        )
        object.__setattr__(self, "rule_A", parse_rule(self.rule_A))
        object.__setattr__(self, "rule_B", parse_rule(self.rule_B))
        object.__setattr__(self, "transition", parse_transition(self.transition))

        if self.rule_B is not self.rule_A.opposite:
            raise ValueError("rule_B must be the opposite of rule_A")
        if self.transition is not Transition.from_rules(self.rule_A, self.rule_B):
            raise ValueError("transition must match rule_A and rule_B")

        for field_name in (
            "pre_count",
            "post_labeled_count",
            "shift_after_position",
            "contradiction_count_post",
        ):
            value = getattr(self, field_name)
            if not _is_plain_int(value):
                raise TypeError(f"{field_name} must be an int")

        template = TEMPLATES[self.template_id]
        if self.pre_count != template.pre_count:
            raise ValueError("pre_count does not match template_id")
        if self.post_labeled_count != template.post_labeled_count:
            raise ValueError("post_labeled_count does not match template_id")
        if self.shift_after_position != self.pre_count:
            raise ValueError("shift_after_position must equal pre_count")

        normalized_items = tuple(self.items)
        if len(normalized_items) != EPISODE_LENGTH:
            raise ValueError(f"items must contain exactly {EPISODE_LENGTH} entries")
        if not all(isinstance(item, EpisodeItem) for item in normalized_items):
            raise TypeError("items must contain EpisodeItem values")
        object.__setattr__(self, "items", normalized_items)

        if self.pre_count + self.post_labeled_count != LABELED_ITEM_COUNT:
            raise ValueError(
                f"pre_count + post_labeled_count must equal {LABELED_ITEM_COUNT}"
            )

        expected_positions = tuple(range(1, EPISODE_LENGTH + 1))
        actual_positions = tuple(item.position for item in normalized_items)
        if actual_positions != expected_positions:
            raise ValueError("items must use positions 1..9 in order")

        labeled_items = normalized_items[:LABELED_ITEM_COUNT]
        probe_items = normalized_items[LABELED_ITEM_COUNT:]
        post_labeled_items = labeled_items[self.pre_count :]

        if len(probe_items) != PROBE_COUNT:
            raise ValueError(f"items must contain exactly {PROBE_COUNT} probes")

        if any(item.kind is not ItemKind.LABELED for item in labeled_items):
            raise ValueError("the first 5 items must be labeled")
        if any(item.kind is not ItemKind.PROBE for item in probe_items):
            raise ValueError("the last 4 items must be probes")

        for item in labeled_items[: self.pre_count]:
            if item.phase is not Phase.PRE:
                raise ValueError("pre-shift labeled items must use the pre phase")

        for item in labeled_items[self.pre_count :]:
            if item.phase is not Phase.POST:
                raise ValueError("post-shift labeled items must use the post phase")

        if any(item.phase is not Phase.POST for item in probe_items):
            raise ValueError("probe items must use the post phase")

        updated_sign_patterns = _build_updated_sign_patterns(post_labeled_items)
        if len(updated_sign_patterns) != 2:
            raise ValueError(
                "post-shift labeled items must cover exactly two distinct sign patterns"
            )
        if not _has_mixed_polarity_sign_patterns(updated_sign_patterns):
            raise ValueError(
                "post-shift labeled items must cover one same-sign and one opposite-sign pattern"
            )

        pairs = tuple((item.q1, item.q2) for item in normalized_items)
        if len(set(pairs)) != len(pairs):
            raise ValueError("items must not repeat a (q1, q2) pair")

        normalized_probe_targets = tuple(
            parse_label(target) for target in self.probe_targets
        )
        if len(normalized_probe_targets) != PROBE_COUNT:
            raise ValueError(f"probe_targets must contain exactly {PROBE_COUNT} entries")
        object.__setattr__(self, "probe_targets", normalized_probe_targets)

        normalized_probe_label_counts = _normalize_probe_label_counts(
            self.probe_label_counts
        )
        object.__setattr__(self, "probe_label_counts", normalized_probe_label_counts)

        normalized_probe_sign_pattern_counts = _normalize_probe_sign_pattern_counts(
            self.probe_sign_pattern_counts
        )
        object.__setattr__(
            self, "probe_sign_pattern_counts", normalized_probe_sign_pattern_counts
        )

        normalized_probe_metadata = tuple(self.probe_metadata)
        if len(normalized_probe_metadata) != PROBE_COUNT:
            raise ValueError(f"probe_metadata must contain exactly {PROBE_COUNT} entries")
        if not all(isinstance(item, ProbeMetadata) for item in normalized_probe_metadata):
            raise TypeError("probe_metadata must contain ProbeMetadata values")
        object.__setattr__(self, "probe_metadata", normalized_probe_metadata)

        expected_probe_positions = tuple(range(LABELED_ITEM_COUNT + 1, EPISODE_LENGTH + 1))
        actual_probe_positions = tuple(item.position for item in normalized_probe_metadata)
        if actual_probe_positions != expected_probe_positions:
            raise ValueError("probe_metadata positions must match probe item positions")

        expected_probe_targets = _build_effective_probe_targets(
            probe_items,
            self.rule_A,
            self.rule_B,
            updated_sign_patterns,
        )
        if normalized_probe_targets != expected_probe_targets:
            raise ValueError(
                "probe_targets must match slice-local derived labels for probe items"
            )
        if not _has_both_probe_labels(normalized_probe_targets):
            raise ValueError("probe_targets must contain at least two distinct labels")
        if _is_global_rule_probe_block(probe_items, normalized_probe_targets, self.rule_A):
            raise ValueError(
                "probe_targets must not collapse to the global rule_A probe block"
            )
        if _is_global_rule_probe_block(probe_items, normalized_probe_targets, self.rule_B):
            raise ValueError(
                "probe_targets must not collapse to the global rule_B probe block"
            )

        expected_probe_metadata = tuple(
            ProbeMetadata(
                position=item.position,
                is_disagreement_probe=label(RuleName.R_STD, item.q1, item.q2)
                != label(RuleName.R_INV, item.q1, item.q2),
                old_rule_label=label(self.rule_A, item.q1, item.q2),
                new_rule_label=label(self.rule_B, item.q1, item.q2),
            )
            for item in probe_items
        )
        if normalized_probe_metadata != expected_probe_metadata:
            raise ValueError(
                "probe_metadata must match the derived rule labels for probe items"
            )

        expected_contradiction_count_post = _build_contradiction_count_post(
            labeled_items,
            self.pre_count,
            self.rule_A,
            self.rule_B,
        )
        if self.contradiction_count_post != expected_contradiction_count_post:
            raise ValueError(
                "contradiction_count_post must match derived post-shift contradictions"
            )
        if self.contradiction_count_post < 1:
            raise ValueError("contradiction_count_post must be at least 1")

        expected_probe_label_counts = _build_probe_label_counts(normalized_probe_targets)
        if normalized_probe_label_counts != expected_probe_label_counts:
            raise ValueError(
                "probe_label_counts must match canonical label counts for probe_targets"
            )

        expected_probe_sign_pattern_counts = _build_probe_sign_pattern_counts(probe_items)
        if normalized_probe_sign_pattern_counts != expected_probe_sign_pattern_counts:
            raise ValueError(
                "probe_sign_pattern_counts must match canonical sign-pattern counts for probe items"
            )
        if normalized_probe_sign_pattern_counts != tuple(
            (pattern, 1) for pattern in _PROBE_SIGN_PATTERN_ORDER
        ):
            raise ValueError(
                "probe_sign_pattern_counts must cover each sign pattern exactly once"
            )

        object.__setattr__(
            self,
            "difficulty_profile_id",
            parse_difficulty_profile_id(self.difficulty_profile_id),
        )
        if not isinstance(self.difficulty_factors, DifficultyFactors):
            raise TypeError("difficulty_factors must be a DifficultyFactors")

        expected_difficulty_factors = derive_difficulty_factors(
            normalized_items,
            self.pre_count,
        )
        if self.difficulty_factors != expected_difficulty_factors:
            raise ValueError(
                "difficulty_factors must match the canonical factor derivation"
            )

        expected_difficulty, expected_profile_id = derive_difficulty_profile(
            expected_difficulty_factors
        )
        if self.difficulty is not expected_difficulty:
            raise ValueError("difficulty must match the derived R13 difficulty rules")
        if self.difficulty_profile_id is not expected_profile_id:
            raise ValueError(
                "difficulty_profile_id must match the derived R13 difficulty profile"
            )

        if self.difficulty_version != DIFFICULTY_VERSION:
            raise ValueError(
                f"difficulty_version must equal {DIFFICULTY_VERSION}"
            )
        if self.spec_version != SPEC_VERSION:
            raise ValueError(f"spec_version must equal {SPEC_VERSION}")
        if self.generator_version != GENERATOR_VERSION:
            raise ValueError(f"generator_version must equal {GENERATOR_VERSION}")
        if self.template_set_version != TEMPLATE_SET_VERSION:
            raise ValueError(
                f"template_set_version must equal {TEMPLATE_SET_VERSION}"
            )
