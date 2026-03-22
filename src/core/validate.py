from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import json
import math
import re
from typing import Iterable

from tasks.iron_find_electric.generator import generate_episode
from tasks.iron_find_electric.protocol import (
    EPISODE_LENGTH,
    LABELED_ITEM_COUNT,
    PROBE_COUNT,
    Difficulty,
    InteractionLabel,
    ItemKind,
    Phase,
    RuleName,
    TEMPLATES,
    TemplateId,
    Transition,
    parse_difficulty,
    parse_label,
    parse_rule,
    parse_template_id,
    parse_transition,
)
from tasks.iron_find_electric.rules import label
from tasks.iron_find_electric.schema import (
    DIFFICULTY_VERSION,
    GENERATOR_VERSION,
    SPEC_VERSION,
    TEMPLATE_SET_VERSION,
    Episode,
    ProbeMetadata,
)

__all__ = [
    "ValidationIssue",
    "RegenerationCheck",
    "EpisodeValidationResult",
    "DatasetDistributionSummary",
    "DatasetValidationResult",
    "normalize_episode_payload",
    "validate_episode",
    "validate_dataset",
]

_EPISODE_ID_PATTERN = re.compile(r"^ife-r3-(\d+)$")
_TEMPLATE_ORDER = (TemplateId.T1.value, TemplateId.T2.value)
_TRANSITION_ORDER = (
    Transition.R_STD_TO_R_INV.value,
    Transition.R_INV_TO_R_STD.value,
)
_PROBE_LABEL_ORDER = (
    InteractionLabel.ATTRACT.value,
    InteractionLabel.REPEL.value,
)
_PROBE_SIGN_PATTERN_ORDER = ("++", "--", "+-", "-+")
_VERSION_FIELD_ORDER = (
    "spec_version",
    "generator_version",
    "template_set_version",
    "difficulty_version",
)
_EPISODE_FIELD_ORDER = tuple(field.name for field in fields(Episode))


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class RegenerationCheck:
    checked: bool
    passed: bool | None
    expected_seed: int | None


@dataclass(frozen=True, slots=True)
class EpisodeValidationResult:
    episode_id: str
    ok: bool
    issues: tuple[ValidationIssue, ...]
    regeneration: RegenerationCheck


@dataclass(frozen=True, slots=True)
class DatasetDistributionSummary:
    template_counts: tuple[tuple[str, int], ...]
    transition_counts: tuple[tuple[str, int], ...]
    probe_label_counts: tuple[tuple[str, int], ...]
    sign_pattern_counts: tuple[tuple[str, int], ...]
    version_values: tuple[tuple[str, tuple[str, ...]], ...]


@dataclass(frozen=True, slots=True)
class DatasetValidationResult:
    ok: bool
    episode_results: tuple[EpisodeValidationResult, ...]
    issues: tuple[ValidationIssue, ...]
    summary: DatasetDistributionSummary


def validate_episode(
    episode: Episode,
    *,
    seed: int | None = None,
) -> EpisodeValidationResult:
    issues: dict[str, ValidationIssue] = {}

    def add_issue(code: str, message: str) -> None:
        if code not in issues:
            issues[code] = ValidationIssue(code=code, message=message)
            return

        existing = issues[code]
        if message not in existing.message:
            issues[code] = ValidationIssue(
                code=code,
                message=f"{existing.message}; {message}",
            )

    episode_id = str(getattr(episode, "episode_id", "<missing episode_id>"))
    missing_fields = tuple(
        field_name for field_name in _EPISODE_FIELD_ORDER if not hasattr(episode, field_name)
    )
    if missing_fields:
        add_issue(
            "missing_episode_fields",
            f"episode is missing required fields: {', '.join(missing_fields)}",
        )

    if not missing_fields:
        try:
            Episode(
                **{
                    field_name: getattr(episode, field_name)
                    for field_name in _EPISODE_FIELD_ORDER
                }
            )
        except (TypeError, ValueError) as exc:
            add_issue("schema_rehydration_failed", str(exc))

    items = tuple(getattr(episode, "items", ()))
    if len(items) != EPISODE_LENGTH:
        add_issue(
            "invalid_episode_length",
            f"items must contain exactly {EPISODE_LENGTH} entries",
        )

    template_id = _parse_template_id(getattr(episode, "template_id", None), add_issue)
    difficulty = _parse_difficulty(getattr(episode, "difficulty", None), add_issue)
    rule_a = _parse_rule(getattr(episode, "rule_A", None), "rule_A", add_issue)
    rule_b = _parse_rule(getattr(episode, "rule_B", None), "rule_B", add_issue)
    transition = _parse_transition(
        getattr(episode, "transition", None),
        add_issue,
    )

    pre_count = getattr(episode, "pre_count", None)
    post_labeled_count = getattr(episode, "post_labeled_count", None)
    shift_after_position = getattr(episode, "shift_after_position", None)
    contradiction_count_post = getattr(episode, "contradiction_count_post", None)

    if template_id is not None:
        template = TEMPLATES[template_id]
        metadata_messages: list[str] = []
        if pre_count != template.pre_count:
            metadata_messages.append("pre_count does not match template_id")
        if post_labeled_count != template.post_labeled_count:
            metadata_messages.append("post_labeled_count does not match template_id")
        if pre_count + post_labeled_count != LABELED_ITEM_COUNT:
            metadata_messages.append(
                f"pre_count + post_labeled_count must equal {LABELED_ITEM_COUNT}"
            )
        if metadata_messages:
            add_issue("invalid_episode_metadata", "; ".join(metadata_messages))

    if shift_after_position != pre_count:
        add_issue(
            "invalid_shift_boundary",
            "shift_after_position must equal pre_count",
        )

    if rule_a is not None and rule_b is not None and rule_b is not rule_a.opposite:
        add_issue(
            "invalid_episode_metadata",
            "rule_B must be the opposite of rule_A",
        )
    if (
        rule_a is not None
        and rule_b is not None
        and transition is not None
        and transition is not Transition.from_rules(rule_a, rule_b)
    ):
        add_issue(
            "invalid_episode_metadata",
            "transition must match rule_A and rule_B",
        )

    if items:
        positions = tuple(getattr(item, "position", None) for item in items)
        expected_positions = tuple(range(1, len(items) + 1))
        if positions != expected_positions:
            add_issue(
                "invalid_item_boundaries",
                "items must use contiguous positions starting at 1",
            )

    labeled_items = items[:LABELED_ITEM_COUNT]
    probe_items = items[LABELED_ITEM_COUNT:]
    if len(items) >= LABELED_ITEM_COUNT and any(
        getattr(item, "kind", None) is not ItemKind.LABELED for item in labeled_items
    ):
        add_issue(
            "invalid_item_boundaries",
            f"the first {LABELED_ITEM_COUNT} items must be labeled",
        )
    if len(items) >= LABELED_ITEM_COUNT and any(
        getattr(item, "kind", None) is not ItemKind.PROBE for item in probe_items
    ):
        add_issue(
            "invalid_item_boundaries",
            f"the last {PROBE_COUNT} items must be probes",
        )

    if isinstance(pre_count, int) and len(items) >= LABELED_ITEM_COUNT:
        invalid_phases = False
        for item in labeled_items[:pre_count]:
            invalid_phases = invalid_phases or getattr(item, "phase", None) is not Phase.PRE
        for item in labeled_items[pre_count:]:
            invalid_phases = invalid_phases or getattr(item, "phase", None) is not Phase.POST
        for item in probe_items:
            invalid_phases = invalid_phases or getattr(item, "phase", None) is not Phase.POST
        if invalid_phases:
            add_issue(
                "invalid_phase_boundaries",
                "labeled/probe phases must respect pre/post boundaries",
            )

    pair_values = tuple(
        (getattr(item, "q1", None), getattr(item, "q2", None))
        for item in items
        if hasattr(item, "q1") and hasattr(item, "q2")
    )
    if len(pair_values) != len(items) or len(set(pair_values)) != len(pair_values):
        add_issue(
            "duplicate_item_pairs",
            "items must not repeat a (q1, q2) pair within the episode",
        )

    if (
        rule_a is not None
        and rule_b is not None
        and isinstance(pre_count, int)
        and len(items) == EPISODE_LENGTH
    ):
        invalid_item_labels = False
        for item in labeled_items:
            expected_rule = rule_a if getattr(item, "position", 0) <= pre_count else rule_b
            raw_label = getattr(item, "label", None)
            try:
                normalized_label = parse_label(raw_label)
            except (TypeError, ValueError):
                invalid_item_labels = True
                continue
            expected_label = _safe_label(
                expected_rule,
                getattr(item, "q1", None),
                getattr(item, "q2", None),
            )
            if expected_label is None or normalized_label is not expected_label:
                invalid_item_labels = True
        for item in probe_items:
            if getattr(item, "label", None) is not None:
                invalid_item_labels = True
        if invalid_item_labels:
            add_issue(
                "invalid_item_labels",
                "labeled items must use the active rule label and probes must have no label",
            )

    normalized_probe_targets = _normalize_probe_targets(
        getattr(episode, "probe_targets", ()),
        add_issue,
    )
    if normalized_probe_targets is not None and len(set(normalized_probe_targets)) < 2:
        add_issue(
            "trivial_probe_block",
            "probe_targets must contain at least two distinct labels",
        )

    if (
        normalized_probe_targets is not None
        and rule_b is not None
        and len(probe_items) == PROBE_COUNT
    ):
        expected_probe_targets = tuple(
            _safe_label(rule_b, getattr(item, "q1", None), getattr(item, "q2", None))
            for item in probe_items
        )
        if None in expected_probe_targets or normalized_probe_targets != expected_probe_targets:
            add_issue(
                "invalid_probe_targets",
                "probe_targets must match rule_B labels for the probe items",
            )

    normalized_probe_metadata = _normalize_probe_metadata(
        getattr(episode, "probe_metadata", ()),
        add_issue,
    )
    if (
        normalized_probe_metadata is not None
        and rule_a is not None
        and rule_b is not None
        and len(probe_items) == PROBE_COUNT
    ):
        expected_probe_metadata = _build_expected_probe_metadata(
            probe_items=probe_items,
            rule_a=rule_a,
            rule_b=rule_b,
        )
        if expected_probe_metadata is None or normalized_probe_metadata != expected_probe_metadata:
            add_issue(
                "invalid_probe_metadata",
                "probe_metadata must match derived rule labels for the probe items",
            )
        elif normalized_probe_targets is not None and normalized_probe_targets != tuple(
            metadata.new_rule_label for metadata in normalized_probe_metadata
        ):
            add_issue(
                "invalid_probe_metadata",
                "probe_metadata new_rule_label values must match probe_targets",
            )

    if rule_a is not None and rule_b is not None and isinstance(pre_count, int):
        derived_contradiction_count = _derive_contradiction_count(
            labeled_items=labeled_items,
            pre_count=pre_count,
            rule_a=rule_a,
            rule_b=rule_b,
        )
        if derived_contradiction_count is None:
            add_issue(
                "invalid_contradiction_count_post",
                "post-shift labeled items must use supported charge values",
            )
        else:
            if contradiction_count_post != derived_contradiction_count:
                add_issue(
                    "invalid_contradiction_count_post",
                    "contradiction_count_post must match derived post-shift contradictions",
                )
            if derived_contradiction_count < 1:
                add_issue(
                    "invalid_contradiction_count_post",
                    "at least one post-shift contradiction is required",
                )

    if normalized_probe_targets is not None:
        expected_probe_label_counts = tuple(
            (
                label_value,
                normalized_probe_targets.count(parse_label(label_value)),
            )
            for label_value in _PROBE_LABEL_ORDER
        )
        actual_probe_label_counts = _normalize_count_pairs(
            getattr(episode, "probe_label_counts", ()),
            order=_PROBE_LABEL_ORDER,
            label_pair=True,
            add_issue=add_issue,
            code="invalid_episode_metadata",
            field_name="probe_label_counts",
        )
        if (
            actual_probe_label_counts is not None
            and actual_probe_label_counts != expected_probe_label_counts
        ):
            add_issue(
                "invalid_episode_metadata",
                "probe_label_counts must match canonical counts for probe_targets",
            )

    expected_sign_pattern_counts = tuple(
        (
            pattern,
            sum(
                _probe_sign_pattern(item.q1, item.q2) == pattern for item in probe_items
            ),
        )
        for pattern in _PROBE_SIGN_PATTERN_ORDER
    )
    actual_sign_pattern_counts = _normalize_count_pairs(
        getattr(episode, "probe_sign_pattern_counts", ()),
        order=_PROBE_SIGN_PATTERN_ORDER,
        label_pair=False,
        add_issue=add_issue,
        code="invalid_episode_metadata",
        field_name="probe_sign_pattern_counts",
    )
    if (
        actual_sign_pattern_counts is not None
        and len(probe_items) == PROBE_COUNT
        and actual_sign_pattern_counts != expected_sign_pattern_counts
    ):
        add_issue(
            "invalid_episode_metadata",
            "probe_sign_pattern_counts must match canonical counts for the probe items",
        )

    if (
        difficulty is not None
        and template_id is not None
        and normalized_probe_targets is not None
        and isinstance(contradiction_count_post, int)
    ):
        expected_difficulty = _derive_difficulty(
            template_id=template_id,
            contradiction_count_post=contradiction_count_post,
            probe_targets=normalized_probe_targets,
        )
        if difficulty is not expected_difficulty:
            add_issue(
                "invalid_episode_metadata",
                "difficulty must match the derived R3 difficulty rules",
            )

    version_messages: list[str] = []
    if getattr(episode, "spec_version", None) != SPEC_VERSION:
        version_messages.append(f"spec_version must equal {SPEC_VERSION}")
    if getattr(episode, "generator_version", None) != GENERATOR_VERSION:
        version_messages.append(f"generator_version must equal {GENERATOR_VERSION}")
    if getattr(episode, "template_set_version", None) != TEMPLATE_SET_VERSION:
        version_messages.append(
            f"template_set_version must equal {TEMPLATE_SET_VERSION}"
        )
    if getattr(episode, "difficulty_version", None) != DIFFICULTY_VERSION:
        version_messages.append(f"difficulty_version must equal {DIFFICULTY_VERSION}")
    if version_messages:
        add_issue("invalid_version_metadata", "; ".join(version_messages))

    regeneration = _run_regeneration_check(episode, seed=seed)
    if regeneration.checked and regeneration.passed is False:
        add_issue(
            "regeneration_mismatch",
            f"episode payload does not match deterministic regeneration for seed {regeneration.expected_seed}",
        )
    return EpisodeValidationResult(
        episode_id=episode_id,
        ok=not issues and (not regeneration.checked or regeneration.passed is True),
        issues=tuple(issues.values()),
        regeneration=regeneration,
    )


def validate_dataset(episodes: Iterable[Episode]) -> DatasetValidationResult:
    normalized_episodes = tuple(episodes)
    episode_results = tuple(validate_episode(episode) for episode in normalized_episodes)
    issues: list[ValidationIssue] = []

    invalid_episode_ids = tuple(
        result.episode_id for result in episode_results if not result.ok
    )
    if invalid_episode_ids:
        issues.append(
            ValidationIssue(
                code="invalid_episode",
                message="invalid episodes: " + ", ".join(invalid_episode_ids),
            )
        )

    duplicate_ids = tuple(
        episode_id
        for episode_id, count in _count_values(
            str(getattr(episode, "episode_id", "")) for episode in normalized_episodes
        )
        if count > 1
    )
    if duplicate_ids:
        issues.append(
            ValidationIssue(
                code="duplicate_episode_id",
                message="duplicate episode_id values: " + ", ".join(duplicate_ids),
            )
        )

    payload_groups = _group_payload_episode_ids(normalized_episodes)
    duplicate_payload_groups = tuple(
        episode_ids for episode_ids in payload_groups if len(episode_ids) > 1
    )
    if duplicate_payload_groups:
        issues.append(
            ValidationIssue(
                code="duplicate_episode_payload",
                message="duplicate episode payloads: "
                + "; ".join(", ".join(group) for group in duplicate_payload_groups),
            )
        )

    summary = DatasetDistributionSummary(
        template_counts=_count_in_canonical_order(
            (_safe_value(getattr(episode, "template_id", None)) for episode in normalized_episodes),
            _TEMPLATE_ORDER,
        ),
        transition_counts=_count_in_canonical_order(
            (_safe_value(getattr(episode, "transition", None)) for episode in normalized_episodes),
            _TRANSITION_ORDER,
        ),
        probe_label_counts=_count_in_canonical_order(
            (
                _safe_value(label_value)
                for episode in normalized_episodes
                for label_value in getattr(episode, "probe_targets", ())
            ),
            _PROBE_LABEL_ORDER,
        ),
        sign_pattern_counts=_count_probe_sign_patterns(normalized_episodes),
        version_values=tuple(
            (
                field_name,
                tuple(
                    sorted(
                        {
                            str(getattr(episode, field_name, None))
                            for episode in normalized_episodes
                        }
                    )
                ),
            )
            for field_name in _VERSION_FIELD_ORDER
        ),
    )

    _append_balance_issue(
        issues,
        code="template_balance",
        field_name="template usage",
        counts=summary.template_counts,
        threshold=max(1, math.ceil(0.2 * len(normalized_episodes))),
    )
    _append_balance_issue(
        issues,
        code="transition_balance",
        field_name="transition usage",
        counts=summary.transition_counts,
        threshold=max(1, math.ceil(0.2 * len(normalized_episodes))),
    )
    if summary.probe_label_counts:
        probe_label_gap = abs(
            summary.probe_label_counts[0][1] - summary.probe_label_counts[1][1]
        )
        probe_label_threshold = max(
            2,
            math.ceil(
                0.2 * sum(count for _, count in summary.probe_label_counts)
            ),
        )
        if probe_label_gap > probe_label_threshold:
            issues.append(
                ValidationIssue(
                    code="probe_label_balance",
                    message=(
                        f"probe label counts differ by {probe_label_gap}, "
                        f"which exceeds the allowed threshold of {probe_label_threshold}"
                    ),
                )
            )
    missing_sign_patterns = tuple(
        pattern for pattern, count in summary.sign_pattern_counts if count == 0
    )
    if missing_sign_patterns:
        issues.append(
            ValidationIssue(
                code="sign_pattern_coverage",
                message="missing probe sign-pattern coverage for: "
                + ", ".join(missing_sign_patterns),
            )
        )

    inconsistent_version_fields = tuple(
        field_name for field_name, values in summary.version_values if len(values) > 1
    )
    if inconsistent_version_fields:
        issues.append(
            ValidationIssue(
                code="version_consistency",
                message="version fields vary across dataset: "
                + ", ".join(inconsistent_version_fields),
            )
        )

    return DatasetValidationResult(
        ok=not issues,
        episode_results=episode_results,
        issues=tuple(issues),
        summary=summary,
    )


def _parse_template_id(
    value: object,
    add_issue,
) -> TemplateId | None:
    try:
        return parse_template_id(value)
    except (TypeError, ValueError):
        add_issue("invalid_template_id", "template_id must be one of: T1, T2")
        return None


def _parse_difficulty(
    value: object,
    add_issue,
) -> Difficulty | None:
    try:
        return parse_difficulty(value)
    except (TypeError, ValueError):
        add_issue(
            "invalid_episode_metadata",
            "difficulty must be a valid benchmark difficulty",
        )
        return None


def _parse_rule(
    value: object,
    field_name: str,
    add_issue,
) -> RuleName | None:
    try:
        return parse_rule(value)
    except (TypeError, ValueError):
        add_issue(
            "invalid_episode_metadata",
            f"{field_name} must be a valid benchmark rule",
        )
        return None


def _parse_transition(
    value: object,
    add_issue,
) -> Transition | None:
    try:
        return parse_transition(value)
    except (TypeError, ValueError):
        add_issue(
            "invalid_episode_metadata",
            "transition must be a valid benchmark transition",
        )
        return None


def _normalize_probe_targets(
    probe_targets: object,
    add_issue,
) -> tuple[InteractionLabel, ...] | None:
    normalized_targets = tuple(probe_targets) if isinstance(probe_targets, tuple) else tuple(probe_targets or ())
    if len(normalized_targets) != PROBE_COUNT:
        add_issue(
            "invalid_probe_targets",
            f"probe_targets must contain exactly {PROBE_COUNT} entries",
        )
        return None

    try:
        return tuple(parse_label(target) for target in normalized_targets)
    except (TypeError, ValueError):
        add_issue(
            "invalid_probe_targets",
            "probe_targets must contain only valid benchmark labels",
        )
        return None


def _normalize_probe_metadata(
    probe_metadata: object,
    add_issue,
) -> tuple[ProbeMetadata, ...] | None:
    normalized_probe_metadata = tuple(probe_metadata) if isinstance(probe_metadata, tuple) else tuple(probe_metadata or ())
    if len(normalized_probe_metadata) != PROBE_COUNT:
        add_issue(
            "invalid_probe_metadata",
            f"probe_metadata must contain exactly {PROBE_COUNT} entries",
        )
        return None
    if not all(isinstance(item, ProbeMetadata) for item in normalized_probe_metadata):
        add_issue(
            "invalid_probe_metadata",
            "probe_metadata must contain ProbeMetadata values",
        )
        return None
    return normalized_probe_metadata


def _normalize_count_pairs(
    pairs: object,
    *,
    order: tuple[str, ...],
    label_pair: bool,
    add_issue,
    code: str,
    field_name: str,
) -> tuple[tuple[str, int], ...] | None:
    normalized_pairs = tuple(pairs) if isinstance(pairs, tuple) else tuple(pairs or ())
    if len(normalized_pairs) != len(order):
        add_issue(
            code,
            f"{field_name} must contain canonical count pairs",
        )
        return None

    result: list[tuple[str, int]] = []
    for expected_key, pair in zip(order, normalized_pairs):
        if not isinstance(pair, tuple) or len(pair) != 2:
            add_issue(code, f"{field_name} entries must be two-item pairs")
            return None
        raw_key, raw_count = pair
        try:
            key = parse_label(raw_key).value if label_pair else str(raw_key)
        except (TypeError, ValueError):
            add_issue(code, f"{field_name} must use canonical keys and int counts")
            return None
        if key != expected_key or not isinstance(raw_count, int) or isinstance(raw_count, bool):
            add_issue(code, f"{field_name} must use canonical keys and int counts")
            return None
        result.append((key, raw_count))
    return tuple(result)


def _run_regeneration_check(
    episode: Episode,
    *,
    seed: int | None,
) -> RegenerationCheck:
    expected_seed = seed if seed is not None else _infer_seed_from_episode_id(
        str(getattr(episode, "episode_id", ""))
    )
    if expected_seed is None:
        return RegenerationCheck(checked=False, passed=None, expected_seed=None)

    try:
        regenerated_episode = generate_episode(
            expected_seed,
            split=getattr(episode, "split", None),
        )
    except (TypeError, ValueError):
        return RegenerationCheck(
            checked=True,
            passed=False,
            expected_seed=expected_seed,
        )

    return RegenerationCheck(
        checked=True,
        passed=normalize_episode_payload(regenerated_episode)
        == normalize_episode_payload(episode),
        expected_seed=expected_seed,
    )


def _infer_seed_from_episode_id(episode_id: str) -> int | None:
    match = _EPISODE_ID_PATTERN.match(episode_id)
    if match is None:
        return None
    return int(match.group(1))


def _group_payload_episode_ids(
    episodes: tuple[Episode, ...],
) -> tuple[tuple[str, ...], ...]:
    groups: dict[str, list[str]] = {}
    for episode in episodes:
        normalized_payload = normalize_episode_payload(episode)
        normalized_payload = {
            key: value for key, value in normalized_payload.items() if key != "episode_id"
        }
        payload_key = json.dumps(
            normalized_payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        groups.setdefault(payload_key, []).append(str(getattr(episode, "episode_id", "")))
    return tuple(
        tuple(episode_ids)
        for _, episode_ids in sorted(groups.items(), key=lambda item: item[0])
    )


def _count_values(values: Iterable[str]) -> tuple[tuple[str, int], ...]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return tuple(sorted(counts.items()))


def _count_in_canonical_order(
    values: Iterable[str],
    order: tuple[str, ...],
) -> tuple[tuple[str, int], ...]:
    counts = {key: 0 for key in order}
    for value in values:
        if value in counts:
            counts[value] += 1
    return tuple((key, counts[key]) for key in order)


def _count_probe_sign_patterns(
    episodes: Iterable[Episode],
) -> tuple[tuple[str, int], ...]:
    counts = {pattern: 0 for pattern in _PROBE_SIGN_PATTERN_ORDER}
    for episode in episodes:
        for item in tuple(getattr(episode, "items", ()))[LABELED_ITEM_COUNT:]:
            pattern = _probe_sign_pattern(getattr(item, "q1", 0), getattr(item, "q2", 0))
            counts[pattern] += 1
    return tuple((pattern, counts[pattern]) for pattern in _PROBE_SIGN_PATTERN_ORDER)


def _append_balance_issue(
    issues: list[ValidationIssue],
    *,
    code: str,
    field_name: str,
    counts: tuple[tuple[str, int], ...],
    threshold: int,
) -> None:
    if not counts:
        return
    count_values = tuple(count for _, count in counts)
    if max(count_values) - min(count_values) > threshold:
        issues.append(
            ValidationIssue(
                code=code,
                message=(
                    f"{field_name} differs by more than the allowed threshold of "
                    f"{threshold}: {counts}"
                ),
            )
        )


def _safe_label(
    rule_name: RuleName,
    q1: object,
    q2: object,
) -> InteractionLabel | None:
    try:
        return label(rule_name, q1, q2)
    except (TypeError, ValueError):
        return None


def _build_expected_probe_metadata(
    *,
    probe_items: tuple[object, ...],
    rule_a: RuleName,
    rule_b: RuleName,
) -> tuple[ProbeMetadata, ...] | None:
    expected_probe_metadata: list[ProbeMetadata] = []
    for item in probe_items:
        position = getattr(item, "position", None)
        old_rule_label = _safe_label(rule_a, getattr(item, "q1", None), getattr(item, "q2", None))
        new_rule_label = _safe_label(rule_b, getattr(item, "q1", None), getattr(item, "q2", None))
        std_label = _safe_label(
            RuleName.R_STD,
            getattr(item, "q1", None),
            getattr(item, "q2", None),
        )
        inv_label = _safe_label(
            RuleName.R_INV,
            getattr(item, "q1", None),
            getattr(item, "q2", None),
        )
        if (
            not isinstance(position, int)
            or old_rule_label is None
            or new_rule_label is None
            or std_label is None
            or inv_label is None
        ):
            return None
        expected_probe_metadata.append(
            ProbeMetadata(
                position=position,
                is_disagreement_probe=std_label != inv_label,
                old_rule_label=old_rule_label,
                new_rule_label=new_rule_label,
            )
        )
    return tuple(expected_probe_metadata)


def _derive_contradiction_count(
    *,
    labeled_items: tuple[object, ...],
    pre_count: int,
    rule_a: RuleName,
    rule_b: RuleName,
) -> int | None:
    contradiction_count = 0
    for item in labeled_items[pre_count:]:
        old_rule_label = _safe_label(rule_a, getattr(item, "q1", None), getattr(item, "q2", None))
        new_rule_label = _safe_label(rule_b, getattr(item, "q1", None), getattr(item, "q2", None))
        if old_rule_label is None or new_rule_label is None:
            return None
        contradiction_count += old_rule_label != new_rule_label
    return contradiction_count


def _probe_sign_pattern(q1: int, q2: int) -> str:
    if q1 > 0 and q2 > 0:
        return "++"
    if q1 < 0 and q2 < 0:
        return "--"
    if q1 > 0 and q2 < 0:
        return "+-"
    return "-+"


def _derive_difficulty(
    *,
    template_id: TemplateId,
    contradiction_count_post: int,
    probe_targets: tuple[InteractionLabel, ...],
) -> Difficulty:
    if (
        template_id is TemplateId.T1
        and contradiction_count_post >= 1
        and len(set(probe_targets)) >= 2
    ):
        return Difficulty.EASY
    return Difficulty.MEDIUM


def _safe_value(value: object) -> str:
    if hasattr(value, "value"):
        return str(getattr(value, "value"))
    return str(value)


def normalize_episode_payload(episode: Episode) -> dict[str, object]:
    return {
        field_name: _normalize_value(getattr(episode, field_name))
        for field_name in _EPISODE_FIELD_ORDER
    }


_normalize_episode_payload = normalize_episode_payload


def _normalize_value(value: object) -> object:
    if hasattr(value, "value"):
        return getattr(value, "value")
    if is_dataclass(value):
        return {
            field.name: _normalize_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, tuple):
        return [_normalize_value(item) for item in value]
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _normalize_value(nested_value)
            for key, nested_value in sorted(value.items(), key=lambda item: str(item[0]))
        }
    return value
