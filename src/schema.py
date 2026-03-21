from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from protocol import (
    CHARGES,
    Difficulty,
    EPISODE_LENGTH,
    LABELED_ITEM_COUNT,
    PROBE_COUNT,
    InteractionLabel,
    ItemKind,
    Phase,
    RuleName,
    Split,
    TemplateId,
    TEMPLATES,
    Transition,
    parse_difficulty,
    parse_item_kind,
    parse_label,
    parse_phase,
    parse_rule,
    parse_split,
    parse_template_id,
    parse_transition,
)
from rules import label

__all__ = [
    "SPEC_VERSION",
    "GENERATOR_VERSION",
    "TEMPLATE_SET_VERSION",
    "DIFFICULTY_VERSION",
    "EpisodeItem",
    "ProbeMetadata",
    "Episode",
]

SPEC_VERSION: Final[str] = "v1"
GENERATOR_VERSION: Final[str] = "R3"
TEMPLATE_SET_VERSION: Final[str] = "v1"
DIFFICULTY_VERSION: Final[str] = "R3"

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


def _derive_difficulty(
    template_id: TemplateId,
    contradiction_count_post: int,
    probe_targets: tuple[InteractionLabel, ...],
) -> Difficulty:
    if (
        template_id is TemplateId.T1
        and contradiction_count_post >= 1
        and _has_both_probe_labels(probe_targets)
    ):
        return Difficulty.EASY
    return Difficulty.MEDIUM


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
    rule_A: RuleName
    rule_B: RuleName
    transition: Transition
    pre_count: int
    post_labeled_count: int
    shift_after_position: int
    contradiction_count_post: int
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

        expected_probe_targets = tuple(
            label(self.rule_B, item.q1, item.q2) for item in probe_items
        )
        if normalized_probe_targets != expected_probe_targets:
            raise ValueError("probe_targets must match rule_B labels for probe items")
        if not _has_both_probe_labels(normalized_probe_targets):
            raise ValueError("probe_targets must contain at least two distinct labels")

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
        if normalized_probe_targets != tuple(
            metadata.new_rule_label for metadata in normalized_probe_metadata
        ):
            raise ValueError("probe_targets must match probe_metadata new_rule_label values")

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

        expected_difficulty = _derive_difficulty(
            self.template_id,
            self.contradiction_count_post,
            normalized_probe_targets,
        )
        if self.difficulty is not expected_difficulty:
            raise ValueError("difficulty must match the derived R3 difficulty rules")

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
