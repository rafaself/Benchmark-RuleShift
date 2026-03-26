from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final, TypeVar

__all__ = [
    "CHARGES",
    "CASE_SPACE",
    "PROBE_COUNT",
    "LABELED_ITEM_COUNT",
    "EPISODE_LENGTH",
    "RULES",
    "LABELS",
    "TEMPLATE_IDS",
    "TEMPLATE_FAMILIES",
    "TRANSITIONS",
    "SPLITS",
    "DIFFICULTIES",
    "PHASES",
    "ITEM_KINDS",
    "RuleName",
    "InteractionLabel",
    "TemplateId",
    "TemplateFamily",
    "Transition",
    "Split",
    "Difficulty",
    "Phase",
    "ItemKind",
    "TemplateSpec",
    "TEMPLATES",
    "parse_rule",
    "parse_label",
    "parse_template_id",
    "parse_template_family",
    "parse_transition",
    "parse_split",
    "parse_difficulty",
    "parse_phase",
    "parse_item_kind",
]

ProtocolEnumT = TypeVar("ProtocolEnumT", bound=StrEnum)


def _is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _parse_enum(
    enum_type: type[ProtocolEnumT],
    value: ProtocolEnumT | str,
    field_name: str,
) -> ProtocolEnumT:
    if isinstance(value, enum_type):
        return value

    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(member.value for member in enum_type)
        raise ValueError(
            f"unknown {field_name}: {value}. expected one of: {allowed}"
        ) from exc


class RuleName(StrEnum):
    R_STD = "R_std"
    R_INV = "R_inv"

    @property
    def opposite(self) -> "RuleName":
        return RuleName.R_INV if self is RuleName.R_STD else RuleName.R_STD


class InteractionLabel(StrEnum):
    ATTRACT = "attract"
    REPEL = "repel"


class TemplateId(StrEnum):
    T1 = "T1"
    T2 = "T2"


class TemplateFamily(StrEnum):
    CANONICAL = "canonical"
    OBSERVATION_LOG = "observation_log"


class Transition(StrEnum):
    R_STD_TO_R_INV = "R_std_to_R_inv"
    R_INV_TO_R_STD = "R_inv_to_R_std"

    @classmethod
    def from_rules(
        cls,
        rule_a: RuleName | str,
        rule_b: RuleName | str,
    ) -> "Transition":
        start = parse_rule(rule_a)
        end = parse_rule(rule_b)

        if start is end:
            raise ValueError("transition requires two distinct rules")

        if start is RuleName.R_STD:
            return cls.R_STD_TO_R_INV

        return cls.R_INV_TO_R_STD


class Split(StrEnum):
    DEV = "dev"
    PUBLIC = "public"
    PRIVATE = "private"


class Difficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Phase(StrEnum):
    PRE = "pre"
    POST = "post"


class ItemKind(StrEnum):
    LABELED = "labeled"
    PROBE = "probe"


CHARGES: Final[tuple[int, ...]] = (-3, -2, -1, 1, 2, 3)

# Ordered pairs are intentional: labels are symmetric in q1/q2, but the
# benchmark preserves presentation order for deterministic generation and replay.
CASE_SPACE: Final[tuple[tuple[int, int], ...]] = tuple(
    (q1, q2) for q1 in CHARGES for q2 in CHARGES
)

PROBE_COUNT: Final[int] = 4
LABELED_ITEM_COUNT: Final[int] = 5
EPISODE_LENGTH: Final[int] = LABELED_ITEM_COUNT + PROBE_COUNT

RULES: Final[frozenset[RuleName]] = frozenset(RuleName)
LABELS: Final[frozenset[InteractionLabel]] = frozenset(InteractionLabel)
TEMPLATE_IDS: Final[frozenset[TemplateId]] = frozenset(TemplateId)
TEMPLATE_FAMILIES: Final[frozenset[TemplateFamily]] = frozenset(TemplateFamily)
TRANSITIONS: Final[frozenset[Transition]] = frozenset(Transition)
SPLITS: Final[frozenset[Split]] = frozenset(Split)
DIFFICULTIES: Final[frozenset[Difficulty]] = frozenset(Difficulty)
PHASES: Final[frozenset[Phase]] = frozenset(Phase)
ITEM_KINDS: Final[frozenset[ItemKind]] = frozenset(ItemKind)


def _validate_charges(charges: tuple[int, ...]) -> None:
    if not charges:
        raise ValueError("CHARGES must not be empty")

    for charge in charges:
        if not _is_plain_int(charge):
            raise TypeError(f"unsupported charge value type: {charge!r}")
        if charge == 0:
            raise ValueError("CHARGES must not include 0")

    if len(set(charges)) != len(charges):
        raise ValueError("CHARGES must contain unique values")


def _validate_case_space(
    charges: tuple[int, ...],
    case_space: tuple[tuple[int, int], ...],
) -> None:
    expected = tuple((q1, q2) for q1 in charges for q2 in charges)
    if case_space != expected:
        raise ValueError(
            "CASE_SPACE must exactly match the ordered cartesian product of CHARGES"
        )


@dataclass(frozen=True, slots=True)
class TemplateSpec:
    """Frozen template counts; `shift_after_position` equals `pre_count`."""

    template_id: TemplateId
    pre_count: int
    post_labeled_count: int
    probe_count: int = PROBE_COUNT

    def __post_init__(self) -> None:
        if not isinstance(self.template_id, TemplateId):
            raise TypeError("template_id must be a TemplateId")

        for field_name in ("pre_count", "post_labeled_count", "probe_count"):
            value = getattr(self, field_name)
            if not _is_plain_int(value):
                raise TypeError(f"{field_name} must be an int")

        if self.pre_count <= 0:
            raise ValueError("pre_count must be positive")
        if self.post_labeled_count <= 0:
            raise ValueError("post_labeled_count must be positive")
        if self.probe_count != PROBE_COUNT:
            raise ValueError(f"probe_count must equal {PROBE_COUNT}")
        if self.pre_count + self.post_labeled_count != LABELED_ITEM_COUNT:
            raise ValueError(
                f"pre_count + post_labeled_count must equal {LABELED_ITEM_COUNT}"
            )
        if self.total_items != EPISODE_LENGTH:
            raise ValueError(f"total_items must equal {EPISODE_LENGTH}")

    @property
    def shift_after_position(self) -> int:
        return self.pre_count

    @property
    def total_items(self) -> int:
        return self.pre_count + self.post_labeled_count + self.probe_count


def _validate_templates(
    templates: dict[TemplateId, TemplateSpec],
) -> dict[TemplateId, TemplateSpec]:
    actual_ids = frozenset(templates)
    if actual_ids != TEMPLATE_IDS:
        raise ValueError(
            "TEMPLATES must define exactly these template ids: "
            f"{sorted(template_id.value for template_id in TEMPLATE_IDS)}"
        )

    for template_id, spec in templates.items():
        if not isinstance(template_id, TemplateId):
            raise TypeError("TEMPLATES keys must be TemplateId values")
        if not isinstance(spec, TemplateSpec):
            raise TypeError("TEMPLATES values must be TemplateSpec instances")
        if spec.template_id is not template_id:
            raise ValueError(
                "TEMPLATES key does not match TemplateSpec.template_id: "
                f"{template_id} != {spec.template_id}"
            )

    return templates


_validate_charges(CHARGES)
_validate_case_space(CHARGES, CASE_SPACE)

TEMPLATES: Final[Mapping[TemplateId, TemplateSpec]] = MappingProxyType(
    _validate_templates(
        {
            TemplateId.T1: TemplateSpec(
                template_id=TemplateId.T1,
                pre_count=2,
                post_labeled_count=3,
            ),
            TemplateId.T2: TemplateSpec(
                template_id=TemplateId.T2,
                pre_count=3,
                post_labeled_count=2,
            ),
        }
    )
)


def parse_rule(value: RuleName | str) -> RuleName:
    return _parse_enum(RuleName, value, "rule")


def parse_label(value: InteractionLabel | str) -> InteractionLabel:
    return _parse_enum(InteractionLabel, value, "label")


def parse_template_id(value: TemplateId | str) -> TemplateId:
    return _parse_enum(TemplateId, value, "template_id")


def parse_template_family(value: TemplateFamily | str) -> TemplateFamily:
    return _parse_enum(TemplateFamily, value, "template_family")


def parse_transition(value: Transition | str) -> Transition:
    return _parse_enum(Transition, value, "transition")


def parse_split(value: Split | str) -> Split:
    return _parse_enum(Split, value, "split")


def parse_difficulty(value: Difficulty | str) -> Difficulty:
    return _parse_enum(Difficulty, value, "difficulty")


def parse_phase(value: Phase | str) -> Phase:
    return _parse_enum(Phase, value, "phase")


def parse_item_kind(value: ItemKind | str) -> ItemKind:
    return _parse_enum(ItemKind, value, "item_kind")
