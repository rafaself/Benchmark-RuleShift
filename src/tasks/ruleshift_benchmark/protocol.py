from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final, TypeVar

__all__ = [
    "MARKER_VALUES",
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
    "DIFFICULTY_PROFILES",
    "FACTOR_LEVELS",
    "PHASES",
    "ITEM_KINDS",
    "RuleName",
    "InteractionLabel",
    "TemplateId",
    "TemplateFamily",
    "Transition",
    "Split",
    "Difficulty",
    "DifficultyProfileId",
    "FactorLevel",
    "Phase",
    "ItemKind",
    "TemplateSpec",
    "TEMPLATES",
    "PUBLIC_CONTRACT_VERSION",
    "format_public_label",
    "format_public_state",
    "parse_public_label",
    "parse_rule",
    "parse_label",
    "parse_template_id",
    "parse_template_family",
    "parse_transition",
    "parse_split",
    "parse_difficulty",
    "parse_difficulty_profile_id",
    "parse_factor_level",
    "parse_phase",
    "parse_item_kind",
]

ProtocolEnumT = TypeVar("ProtocolEnumT", bound=StrEnum)
PUBLIC_CONTRACT_VERSION: Final[str] = "markers-v1"


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
    ZARK = "zark"
    BLIM = "blim"


_PUBLIC_LABEL_ALIASES: Final[Mapping[str, InteractionLabel]] = MappingProxyType(
    {
        "type_a": InteractionLabel.ZARK,
        "type_b": InteractionLabel.BLIM,
    }
)


class TemplateId(StrEnum):
    T1 = "T1"
    T2 = "T2"
    T3 = "T3"


class TemplateFamily(StrEnum):
    CANONICAL = "canonical"
    OBSERVATION_LOG = "observation_log"
    CASE_LEDGER = "case_ledger"


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
    PUBLIC = "public"
    PRIVATE = "private"


class Difficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class DifficultyProfileId(StrEnum):
    EASY_ANCHORED = "easy_anchored"
    MEDIUM_BALANCED = "medium_balanced"
    HARD_INTERLEAVED = "hard_interleaved"


class FactorLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Phase(StrEnum):
    PRE = "pre"
    POST = "post"


class ItemKind(StrEnum):
    LABELED = "labeled"
    PROBE = "probe"


MARKER_VALUES: Final[tuple[int, ...]] = (-3, -2, -1, 1, 2, 3)

# Ordered pairs are intentional: state labels are symmetric in q1/q2, but the
# benchmark preserves presentation order for deterministic generation and replay.
CASE_SPACE: Final[tuple[tuple[int, int], ...]] = tuple(
    (q1, q2) for q1 in MARKER_VALUES for q2 in MARKER_VALUES
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
DIFFICULTY_PROFILES: Final[frozenset[DifficultyProfileId]] = frozenset(
    DifficultyProfileId
)
FACTOR_LEVELS: Final[frozenset[FactorLevel]] = frozenset(FactorLevel)
PHASES: Final[frozenset[Phase]] = frozenset(Phase)
ITEM_KINDS: Final[frozenset[ItemKind]] = frozenset(ItemKind)


def _validate_marker_values(marker_values: tuple[int, ...]) -> None:
    if not marker_values:
        raise ValueError("MARKER_VALUES must not be empty")

    for marker_value in marker_values:
        if not _is_plain_int(marker_value):
            raise TypeError(f"unsupported marker value type: {marker_value!r}")
        if marker_value == 0:
            raise ValueError("MARKER_VALUES must not include 0")

    if len(set(marker_values)) != len(marker_values):
        raise ValueError("MARKER_VALUES must contain unique values")


def _validate_case_space(
    marker_values: tuple[int, ...],
    case_space: tuple[tuple[int, int], ...],
) -> None:
    expected = tuple((q1, q2) for q1 in marker_values for q2 in marker_values)
    if case_space != expected:
        raise ValueError(
            "CASE_SPACE must exactly match the ordered cartesian product of MARKER_VALUES"
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


_validate_marker_values(MARKER_VALUES)
_validate_case_space(MARKER_VALUES, CASE_SPACE)

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
            TemplateId.T3: TemplateSpec(
                template_id=TemplateId.T3,
                pre_count=1,
                post_labeled_count=4,
            ),
        }
    )
)


def parse_rule(value: RuleName | str) -> RuleName:
    return _parse_enum(RuleName, value, "rule")


def parse_label(value: InteractionLabel | str) -> InteractionLabel:
    return _parse_enum(InteractionLabel, value, "label")


def parse_public_label(value: InteractionLabel | str) -> InteractionLabel:
    if isinstance(value, InteractionLabel):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _PUBLIC_LABEL_ALIASES:
            return _PUBLIC_LABEL_ALIASES[normalized]
    allowed = ", ".join(_PUBLIC_LABEL_ALIASES)
    raise ValueError(f"unknown public label: {value}. expected one of: {allowed}")


def format_public_label(value: InteractionLabel | str) -> str:
    return {
        InteractionLabel.ZARK: "type_a",
        InteractionLabel.BLIM: "type_b",
    }[parse_label(value)]


def format_public_state(value: InteractionLabel | str) -> str:
    return parse_label(value).value


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


def parse_difficulty_profile_id(
    value: DifficultyProfileId | str,
) -> DifficultyProfileId:
    return _parse_enum(DifficultyProfileId, value, "difficulty_profile_id")


def parse_factor_level(value: FactorLevel | str) -> FactorLevel:
    return _parse_enum(FactorLevel, value, "factor_level")


def parse_phase(value: Phase | str) -> Phase:
    return _parse_enum(Phase, value, "phase")


def parse_item_kind(value: ItemKind | str) -> ItemKind:
    return _parse_enum(ItemKind, value, "item_kind")
