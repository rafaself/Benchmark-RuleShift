#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROWS_PATH = ROOT / "kaggle/dataset/public/public_leaderboard_rows.json"
PUBLIC_METADATA_PATH = ROOT / "kaggle/dataset/public/dataset-metadata.json"
PRIVATE_ROWS_PATH = ROOT / "kaggle/dataset/private/private_leaderboard_rows.json"
PRIVATE_ANSWER_KEY_PATH = ROOT / "kaggle/dataset/private/private_answer_key.json"
PRIVATE_MANIFEST_PATH = ROOT / "kaggle/dataset/private/private_split_manifest.json"
PRIVATE_METADATA_PATH = ROOT / "kaggle/dataset/private/dataset-metadata.json"

PUBLIC_DATASET_ID = "raptorengineer/ruleshift-cogflex-runtime-v2"
PRIVATE_DATASET_ID = "raptorengineer/ruleshift-cogflex-runtime-private-v2"
NOTEBOOK_ID = "raptorengineer/ruleshift-cogflex-notebook-v2"
TASK_NAME = "ruleshift_cogflex_v2_binary"
FACULTY_ID = "executive_functions/cognitive_flexibility"

VALUES = (-3, -2, -1, 1, 2, 3)
DOMAIN = [(r1, r2) for r1 in VALUES for r2 in VALUES]
TYPE_TRUE = "type_a"
TYPE_FALSE = "type_b"
GROUPS = ("explicit_switch", "reversal", "latent_switch", "context_switch")
CONTEXTS = ("alpha", "beta")
LEARN_EXAMPLE_COUNT = 4
SHIFT_EXAMPLE_COUNT = 4
FINAL_PROBE_COUNT = 4
PAIR_FAMILY_COUNT = 20
REVERSAL_FAMILY_COUNT = 20
FINAL_OUTPUT_INSTRUCTION = (
    "Return exactly 4 outputs in order, one per probe. "
    "Use only type_a or type_b."
)

GROUP_SHIFT_MODES = {
    "explicit_switch": "explicit_instruction",
    "reversal": "label_reversal",
    "latent_switch": "latent_example_change",
    "context_switch": "context_gate",
}


def fmt_signed(value: int) -> str:
    return f"{value:+d}"


@dataclass(frozen=True)
class RuleSpec:
    rule_id: str
    description: str
    predicate: Callable[[int, int], bool]

    def label(self, point: tuple[int, int]) -> bool:
        return self.predicate(*point)

    def explain(self, point: tuple[int, int], label: str) -> str:
        return (
            f"{self.description}. For r1={fmt_signed(point[0])}, "
            f"r2={fmt_signed(point[1])}, the correct label is {label}."
        )


@dataclass(frozen=True)
class TransitionFamily:
    family_id: str
    initial_rule_id: str
    shift_rule_id: str


def make_rule(rule_id: str, description: str, predicate: Callable[[int, int], bool]) -> RuleSpec:
    return RuleSpec(rule_id=rule_id, description=description, predicate=predicate)


def _candidate_rules() -> list[RuleSpec]:
    return [
        make_rule("r1_at_least_minus_1", "type_a iff r1 is at least -1", lambda r1, r2: r1 >= -1),
        make_rule("r1_at_least_plus_2", "type_a iff r1 is at least +2", lambda r1, r2: r1 >= 2),
        make_rule("r2_at_least_minus_1", "type_a iff r2 is at least -1", lambda r1, r2: r2 >= -1),
        make_rule("r2_at_least_plus_2", "type_a iff r2 is at least +2", lambda r1, r2: r2 >= 2),
        make_rule("r1_at_most_minus_1", "type_a iff r1 is at most -1", lambda r1, r2: r1 <= -1),
        make_rule("r1_at_most_plus_1", "type_a iff r1 is at most +1", lambda r1, r2: r1 <= 1),
        make_rule("r2_at_most_minus_1", "type_a iff r2 is at most -1", lambda r1, r2: r2 <= -1),
        make_rule("r2_at_most_plus_1", "type_a iff r2 is at most +1", lambda r1, r2: r2 <= 1),
        make_rule("sum_at_least_plus_1", "type_a iff r1 + r2 is at least +1", lambda r1, r2: r1 + r2 >= 1),
        make_rule("sum_at_least_plus_4", "type_a iff r1 + r2 is at least +4", lambda r1, r2: r1 + r2 >= 4),
        make_rule("sum_at_most_minus_1", "type_a iff r1 + r2 is at most -1", lambda r1, r2: r1 + r2 <= -1),
        make_rule("sum_at_most_minus_4", "type_a iff r1 + r2 is at most -4", lambda r1, r2: r1 + r2 <= -4),
        make_rule("diff_at_least_plus_1", "type_a iff r1 - r2 is at least +1", lambda r1, r2: r1 - r2 >= 1),
        make_rule("diff_at_least_plus_4", "type_a iff r1 - r2 is at least +4", lambda r1, r2: r1 - r2 >= 4),
        make_rule("diff_at_most_minus_1", "type_a iff r1 - r2 is at most -1", lambda r1, r2: r1 - r2 <= -1),
        make_rule("diff_at_most_minus_4", "type_a iff r1 - r2 is at most -4", lambda r1, r2: r1 - r2 <= -4),
        make_rule("abs_r1_at_least_2", "type_a iff |r1| is at least 2", lambda r1, r2: abs(r1) >= 2),
        make_rule("abs_r2_at_least_2", "type_a iff |r2| is at least 2", lambda r1, r2: abs(r2) >= 2),
        make_rule("same_sign", "type_a iff r1 and r2 share the same sign", lambda r1, r2: (r1 > 0) == (r2 > 0)),
        make_rule("r1_positive", "type_a iff r1 is positive", lambda r1, r2: r1 > 0),
        make_rule("abs_equal", "type_a iff |r1| equals |r2|", lambda r1, r2: abs(r1) == abs(r2)),
        make_rule(
            "abs_r1_greater_than_abs_r2",
            "type_a iff |r1| is greater than |r2|",
            lambda r1, r2: abs(r1) > abs(r2),
        ),
        make_rule(
            "abs_r1_less_than_abs_r2",
            "type_a iff |r1| is less than |r2|",
            lambda r1, r2: abs(r1) < abs(r2),
        ),
        make_rule("both_positive", "type_a iff both r1 and r2 are positive", lambda r1, r2: r1 > 0 and r2 > 0),
        make_rule("both_negative", "type_a iff both r1 and r2 are negative", lambda r1, r2: r1 < 0 and r2 < 0),
        make_rule("same_parity", "type_a iff r1 and r2 have the same parity", lambda r1, r2: (r1 % 2) == (r2 % 2)),
        make_rule("r1_even", "type_a iff r1 is even", lambda r1, r2: r1 % 2 == 0),
        make_rule("r2_even", "type_a iff r2 is even", lambda r1, r2: r2 % 2 == 0),
        make_rule("r2_positive", "type_a iff r2 is positive", lambda r1, r2: r2 > 0),
        make_rule("both_even", "type_a iff both r1 and r2 are even", lambda r1, r2: r1 % 2 == 0 and r2 % 2 == 0),
        make_rule(
            "abs_sum_at_least_4",
            "type_a iff |r1| + |r2| is at least 4",
            lambda r1, r2: abs(r1) + abs(r2) >= 4,
        ),
        make_rule(
            "abs_sum_at_least_5",
            "type_a iff |r1| + |r2| is at least 5",
            lambda r1, r2: abs(r1) + abs(r2) >= 5,
        ),
        make_rule(
            "abs_diff_at_least_2",
            "type_a iff ||r1| - |r2|| is at least 2",
            lambda r1, r2: abs(abs(r1) - abs(r2)) >= 2,
        ),
        make_rule(
            "abs_diff_at_most_1",
            "type_a iff ||r1| - |r2|| is at most 1",
            lambda r1, r2: abs(abs(r1) - abs(r2)) <= 1,
        ),
        make_rule("max_at_least_2", "type_a iff max(r1, r2) is at least +2", lambda r1, r2: max(r1, r2) >= 2),
        make_rule("min_at_most_minus_2", "type_a iff min(r1, r2) is at most -2", lambda r1, r2: min(r1, r2) <= -2),
        make_rule(
            "exactly_one_large_abs",
            "type_a iff exactly one marker has absolute value at least 2",
            lambda r1, r2: (abs(r1) >= 2) ^ (abs(r2) >= 2),
        ),
        make_rule(
            "at_least_one_positive",
            "type_a iff at least one of r1 and r2 is positive",
            lambda r1, r2: r1 > 0 or r2 > 0,
        ),
        make_rule(
            "at_least_one_negative",
            "type_a iff at least one of r1 and r2 is negative",
            lambda r1, r2: r1 < 0 or r2 < 0,
        ),
        make_rule("r1_odd", "type_a iff r1 is odd", lambda r1, r2: r1 % 2 != 0),
    ]


def _rule_signature(rule: RuleSpec) -> tuple[bool, ...]:
    return tuple(rule.label(point) for point in DOMAIN)


def _build_rule_catalog() -> tuple[RuleSpec, ...]:
    unique_rules: list[RuleSpec] = []
    seen_signatures: set[tuple[bool, ...]] = set()
    for rule in _candidate_rules():
        signature = _rule_signature(rule)
        if signature in seen_signatures:
            continue
        unique_rules.append(rule)
        seen_signatures.add(signature)
    if len(unique_rules) != 40:
        raise RuntimeError(f"Expected 40 unique rules, found {len(unique_rules)}")
    return tuple(unique_rules)


RULES = _build_rule_catalog()
RULE_BY_ID = {rule.rule_id: rule for rule in RULES}
PUBLIC_RULE_IDS = tuple(rule.rule_id for rule in RULES[:20])
PRIVATE_RULE_IDS = tuple(rule.rule_id for rule in RULES[20:])


def label_from_bool(value: bool) -> str:
    return TYPE_TRUE if value else TYPE_FALSE


def flip_label(label: str) -> str:
    return TYPE_FALSE if label == TYPE_TRUE else TYPE_TRUE


def rule_label(rule: RuleSpec, point: tuple[int, int], *, reversed_labels: bool = False) -> str:
    label = label_from_bool(rule.label(point))
    return flip_label(label) if reversed_labels else label


def disagreement_count(initial_rule_id: str, shift_rule_id: str) -> int:
    initial_rule = RULE_BY_ID[initial_rule_id]
    shift_rule = RULE_BY_ID[shift_rule_id]
    return sum(initial_rule.label(point) != shift_rule.label(point) for point in DOMAIN)


def build_pair_families(rule_ids: tuple[str, ...], split_name: str) -> tuple[TransitionFamily, ...]:
    offsets = (7, 9, 11, 13, 17, 3, 1)
    for offset in offsets:
        if math.gcd(offset, len(rule_ids)) != 1:
            continue
        families = []
        for index, initial_rule_id in enumerate(rule_ids):
            shift_rule_id = rule_ids[(index + offset) % len(rule_ids)]
            if disagreement_count(initial_rule_id, shift_rule_id) < 8:
                families = []
                break
            family_id = f"{split_name}::transition::{initial_rule_id}__to__{shift_rule_id}"
            families.append(TransitionFamily(family_id, initial_rule_id, shift_rule_id))
        if len(families) == PAIR_FAMILY_COUNT:
            return tuple(families)
    raise RuntimeError(f"Unable to derive pair families for split {split_name}")


PUBLIC_PAIR_FAMILIES = build_pair_families(PUBLIC_RULE_IDS, "public")
PRIVATE_PAIR_FAMILIES = build_pair_families(PRIVATE_RULE_IDS, "private")
PUBLIC_REVERSAL_FAMILIES = tuple(
    TransitionFamily(f"public::reversal::{rule_id}", rule_id, rule_id) for rule_id in PUBLIC_RULE_IDS
)
PRIVATE_REVERSAL_FAMILIES = tuple(
    TransitionFamily(f"private::reversal::{rule_id}", rule_id, rule_id) for rule_id in PRIVATE_RULE_IDS
)


def families_for_group(split_name: str, group_id: str) -> tuple[TransitionFamily, ...]:
    if group_id == "reversal":
        return PUBLIC_REVERSAL_FAMILIES if split_name == "public" else PRIVATE_REVERSAL_FAMILIES
    return PUBLIC_PAIR_FAMILIES if split_name == "public" else PRIVATE_PAIR_FAMILIES


def derive_episode_seed(split_name: str, group_id: str, family_id: str, variant: int, purpose: str, private_seed: str | None) -> int:
    parts = [split_name, group_id, family_id, str(variant), purpose]
    if private_seed is not None:
        parts.insert(0, private_seed)
    payload = "::".join(parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def has_label_balance(labels: list[str], min_true: int = 1, min_false: int = 1) -> bool:
    counts = Counter(labels)
    return counts[TYPE_TRUE] >= min_true and counts[TYPE_FALSE] >= min_false


def pick_points(
    rng: random.Random,
    candidates: list[tuple[int, int]],
    count: int,
    predicate: Callable[[list[tuple[int, int]]], bool],
    *,
    attempts: int = 6000,
) -> list[tuple[int, int]]:
    if len(candidates) < count:
        raise RuntimeError(f"Need {count} candidates, found {len(candidates)}")
    for _ in range(attempts):
        sample = rng.sample(candidates, count)
        if predicate(sample):
            rng.shuffle(sample)
            return sample
    raise RuntimeError("Unable to pick points that satisfy benchmark constraints")


def serialize_case(
    index: int,
    point: tuple[int, int],
    label: str,
    *,
    context: str | None = None,
    previous_rule_label: str | None = None,
    justification: str | None = None,
) -> dict[str, object]:
    item: dict[str, object] = {
        "index": index,
        "r1": point[0],
        "r2": point[1],
        "label": label,
    }
    if context is not None:
        item["context"] = context
    if previous_rule_label is not None:
        item["previous_rule_label"] = previous_rule_label
    if justification is not None:
        item["justification"] = justification
    return item


def render_case_text(point: tuple[int, int], context: str | None = None) -> str:
    if context is None:
        return f"r1={fmt_signed(point[0])}, r2={fmt_signed(point[1])}"
    return f"context={context} | r1={fmt_signed(point[0])}, r2={fmt_signed(point[1])}"


def render_labeled_lines(items: list[dict[str, object]]) -> str:
    return "\n".join(
        f"{item['index']}. {render_case_text((int(item['r1']), int(item['r2'])), item.get('context'))} -> {item['label']}"
        for item in items
    )


def render_probe_lines(items: list[dict[str, object]]) -> str:
    return "\n".join(
        f"{item['index']}. {render_case_text((int(item['r1']), int(item['r2'])), item.get('context'))} -> ?"
        for item in items
    )


def render_turn_header(episode_id: str, turn_index: int) -> str:
    return f"RuleShift cognitive flexibility task. Episode {episode_id}. Turn {turn_index} of 3."


def render_learn_turn(episode_id: str, examples: list[dict[str, object]], *, context: str | None = None) -> str:
    guidance = "Learn the current classification rule from these labeled examples."
    if context is not None:
        guidance = f"Learn the active rule for context={context} from these labeled examples."
    return "\n\n".join(
        [
            render_turn_header(episode_id, 1),
            guidance,
            "Examples:\n" + render_labeled_lines(examples),
        ]
    )


def render_shift_turn(
    episode_id: str,
    group_id: str,
    examples: list[dict[str, object]],
    *,
    context: str | None = None,
) -> str:
    if group_id == "explicit_switch":
        guidance = "The active rule has changed. Update your classification rule using these labeled examples."
    elif group_id == "reversal":
        guidance = (
            "The condition stays the same, but the label mapping is reversed. "
            "Items that were type_a are now type_b, and items that were type_b are now type_a."
        )
    elif group_id == "latent_switch":
        guidance = "Continue the task using the following labeled examples."
    elif group_id == "context_switch":
        assert context is not None
        guidance = f"Now learn the active rule for context={context} from these labeled examples."
    else:
        raise ValueError(f"Unsupported group_id {group_id}")
    return "\n\n".join(
        [
            render_turn_header(episode_id, 2),
            guidance,
            "Examples:\n" + render_labeled_lines(examples),
        ]
    )


def render_decision_turn(episode_id: str, probes: list[dict[str, object]], *, context_switch: bool) -> str:
    guidance = "Classify each probe with the active rule."
    if context_switch:
        guidance = "Classify each probe with the active rule for that probe's context."
    return "\n\n".join(
        [
            render_turn_header(episode_id, 3),
            guidance,
            "Probes:\n" + render_probe_lines(probes),
            FINAL_OUTPUT_INSTRUCTION,
        ]
    )


def pick_balanced_examples(
    rng: random.Random,
    rule: RuleSpec,
    count: int,
    *,
    reversed_labels: bool = False,
    exclude: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    excluded = set() if exclude is None else exclude
    candidates = [point for point in DOMAIN if point not in excluded]
    return pick_points(
        rng,
        candidates,
        count,
        lambda sample: has_label_balance([rule_label(rule, point, reversed_labels=reversed_labels) for point in sample]),
    )


def build_rule_examples(
    points: list[tuple[int, int]],
    rule: RuleSpec,
    *,
    start_index: int,
    reversed_labels: bool = False,
    context: str | None = None,
) -> list[dict[str, object]]:
    return [
        serialize_case(index, point, rule_label(rule, point, reversed_labels=reversed_labels), context=context)
        for index, point in enumerate(points, start=start_index)
    ]


def build_probe_items(
    points: list[tuple[int, int]],
    rule: RuleSpec,
    *,
    start_index: int,
    previous_rule: RuleSpec,
    reversed_labels: bool = False,
    context: str | None = None,
) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for index, point in enumerate(points, start=start_index):
        label = rule_label(rule, point, reversed_labels=reversed_labels)
        items.append(
            serialize_case(
                index,
                point,
                label,
                context=context,
                previous_rule_label=rule_label(previous_rule, point),
                justification=rule.explain(point, label),
            )
        )
    return items


def build_context_probe_items(
    items: list[tuple[str, tuple[int, int]]],
    initial_rule: RuleSpec,
    shift_rule: RuleSpec,
    *,
    start_index: int,
) -> list[dict[str, object]]:
    probes: list[dict[str, object]] = []
    for index, (context, point) in enumerate(items, start=start_index):
        active_rule = initial_rule if context == CONTEXTS[0] else shift_rule
        label = rule_label(active_rule, point)
        probes.append(
            serialize_case(
                index,
                point,
                label,
                context=context,
                previous_rule_label=rule_label(initial_rule, point),
                justification=active_rule.explain(point, label),
            )
        )
    return probes


def build_transition_episode(
    group_id: str,
    family: TransitionFamily,
    rng: random.Random,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    initial_rule = RULE_BY_ID[family.initial_rule_id]
    shift_rule = RULE_BY_ID[family.shift_rule_id]
    for _ in range(4000):
        initial_points = pick_balanced_examples(rng, initial_rule, LEARN_EXAMPLE_COUNT)
        used = set(initial_points)
        disagreement_points = [
            point for point in DOMAIN if initial_rule.label(point) != shift_rule.label(point) and point not in used
        ]
        if len(disagreement_points) < FINAL_PROBE_COUNT:
            continue
        try:
            shift_points = pick_points(
                rng,
                [point for point in DOMAIN if point not in used],
                SHIFT_EXAMPLE_COUNT,
                lambda sample: (
                    sum(point in disagreement_points for point in sample) >= 3
                    and has_label_balance([rule_label(shift_rule, point) for point in sample])
                ),
            )
        except RuntimeError:
            continue
        used.update(shift_points)
        remaining_disagreement = [point for point in disagreement_points if point not in used]
        if len(remaining_disagreement) < FINAL_PROBE_COUNT:
            continue
        probe_points = pick_points(
            rng,
            remaining_disagreement,
            FINAL_PROBE_COUNT,
            lambda sample: True,
        )
        return (
            build_rule_examples(initial_points, initial_rule, start_index=1),
            build_rule_examples(shift_points, shift_rule, start_index=1),
            build_probe_items(probe_points, shift_rule, start_index=1, previous_rule=initial_rule),
        )
    raise RuntimeError(f"Unable to build transition episode for family {family.family_id} ({group_id})")


def build_reversal_episode(
    family: TransitionFamily,
    rng: random.Random,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    rule = RULE_BY_ID[family.initial_rule_id]
    for _ in range(2000):
        initial_points = pick_balanced_examples(rng, rule, LEARN_EXAMPLE_COUNT)
        used = set(initial_points)
        try:
            shift_points = pick_balanced_examples(rng, rule, SHIFT_EXAMPLE_COUNT, reversed_labels=True, exclude=used)
            used.update(shift_points)
            probe_points = pick_balanced_examples(rng, rule, FINAL_PROBE_COUNT, reversed_labels=True, exclude=used)
        except RuntimeError:
            continue
        return (
            build_rule_examples(initial_points, rule, start_index=1),
            build_rule_examples(shift_points, rule, start_index=1, reversed_labels=True),
            build_probe_items(probe_points, rule, start_index=1, previous_rule=rule, reversed_labels=True),
        )
    raise RuntimeError(f"Unable to build reversal episode for family {family.family_id}")


def build_context_switch_episode(
    family: TransitionFamily,
    rng: random.Random,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    initial_rule = RULE_BY_ID[family.initial_rule_id]
    shift_rule = RULE_BY_ID[family.shift_rule_id]
    for _ in range(4000):
        initial_points = pick_balanced_examples(rng, initial_rule, LEARN_EXAMPLE_COUNT)
        used = set(initial_points)
        try:
            shift_points = pick_balanced_examples(rng, shift_rule, SHIFT_EXAMPLE_COUNT, exclude=used)
        except RuntimeError:
            continue
        used.update(shift_points)
        disagreement_points = [point for point in DOMAIN if initial_rule.label(point) != shift_rule.label(point) and point not in used]
        if len(disagreement_points) < FINAL_PROBE_COUNT:
            continue
        alpha_points = pick_points(rng, disagreement_points, 2, lambda sample: True)
        used.update(alpha_points)
        remaining = [point for point in disagreement_points if point not in used]
        if len(remaining) < 2:
            continue
        beta_points = pick_points(rng, remaining, 2, lambda sample: True)
        probes = [(CONTEXTS[0], point) for point in alpha_points] + [(CONTEXTS[1], point) for point in beta_points]
        rng.shuffle(probes)
        return (
            build_rule_examples(initial_points, initial_rule, start_index=1, context=CONTEXTS[0]),
            build_rule_examples(shift_points, shift_rule, start_index=1, context=CONTEXTS[1]),
            build_context_probe_items(probes, initial_rule, shift_rule, start_index=1),
        )
    raise RuntimeError(f"Unable to build context-switch episode for family {family.family_id}")


def episode_record(
    split_name: str,
    episode_id: str,
    group_id: str,
    family: TransitionFamily,
    learn_examples: list[dict[str, object]],
    shift_examples: list[dict[str, object]],
    final_probes: list[dict[str, object]],
) -> tuple[dict[str, object], dict[str, object]]:
    context_switch = group_id == "context_switch"
    turns = [
        render_learn_turn(episode_id, learn_examples, context=CONTEXTS[0] if context_switch else None),
        render_shift_turn(
            episode_id,
            group_id,
            shift_examples,
            context=CONTEXTS[1] if context_switch else None,
        ),
        render_decision_turn(episode_id, final_probes, context_switch=context_switch),
    ]
    analysis = {
        "faculty_id": FACULTY_ID,
        "group_id": group_id,
        "transition_family_id": family.family_id,
        "initial_rule_id": family.initial_rule_id,
        "shift_rule_id": family.shift_rule_id,
        "shift_mode": GROUP_SHIFT_MODES[group_id],
    }
    row = {
        "episode_id": episode_id,
        "inference": {
            "turns": turns,
        },
        "scoring": {
            "final_probe_targets": [str(item["label"]) for item in final_probes],
        },
        "analysis": analysis,
        "split": split_name,
    }
    answer = {
        "episode_id": episode_id,
        **analysis,
        "learn_turn_examples": learn_examples,
        "shift_turn_examples": shift_examples,
        "final_probes": final_probes,
        "final_probe_targets": [str(item["label"]) for item in final_probes],
        "turns": turns,
        "split": split_name,
    }
    return (row, answer)


def build_episode(
    split_name: str,
    episode_id: str,
    group_id: str,
    family: TransitionFamily,
    rng: random.Random,
) -> tuple[dict[str, object], dict[str, object]]:
    if group_id in {"explicit_switch", "latent_switch"}:
        learn_examples, shift_examples, final_probes = build_transition_episode(group_id, family, rng)
    elif group_id == "reversal":
        learn_examples, shift_examples, final_probes = build_reversal_episode(family, rng)
    elif group_id == "context_switch":
        learn_examples, shift_examples, final_probes = build_context_switch_episode(family, rng)
    else:
        raise ValueError(f"Unsupported group_id {group_id}")
    row, answer = episode_record(split_name, episode_id, group_id, family, learn_examples, shift_examples, final_probes)
    validate_episode_constraints(row, answer)
    return (row, answer)


def validate_episode_constraints(row: dict[str, object], answer: dict[str, object]) -> None:
    analysis = row["analysis"]
    group_id = str(analysis["group_id"])
    initial_rule = RULE_BY_ID[str(analysis["initial_rule_id"])]
    shift_rule = RULE_BY_ID[str(analysis["shift_rule_id"])]
    final_probes = answer["final_probes"]
    final_targets = tuple(answer["final_probe_targets"])
    previous_rule_targets = tuple(str(item["previous_rule_label"]) for item in final_probes)
    previous_accuracy = sum(pred == target for pred, target in zip(previous_rule_targets, final_targets))

    if group_id in {"explicit_switch", "reversal", "latent_switch"} and previous_accuracy > 1:
        raise ValueError(
            f"episode {row['episode_id']} violates perseveration threshold: previous-rule accuracy={previous_accuracy}"
        )

    if group_id == "context_switch":
        contexts = [str(item["context"]) for item in final_probes]
        counts = Counter(contexts)
        if counts != Counter({CONTEXTS[0]: 2, CONTEXTS[1]: 2}):
            raise ValueError(f"episode {row['episode_id']} must expose exactly two probes per context: {counts}")
        shift_rule_targets = tuple(
            rule_label(shift_rule, (int(item["r1"]), int(item["r2"])))
            for item in final_probes
        )
        if previous_accuracy > 2 or sum(pred == target for pred, target in zip(shift_rule_targets, final_targets)) > 2:
            raise ValueError(f"episode {row['episode_id']} violates context-switch one-rule thresholds")


def episode_signature(answer: dict[str, object]) -> tuple[object, ...]:
    return (
        answer["group_id"],
        answer["transition_family_id"],
        tuple(answer["turns"]),
        tuple(answer["final_probe_targets"]),
    )


def validate_answer_uniqueness(answers: list[dict[str, object]], split_name: str) -> None:
    seen: dict[tuple[object, ...], str] = {}
    for answer in answers:
        signature = episode_signature(answer)
        previous_episode_id = seen.get(signature)
        if previous_episode_id is not None:
            raise ValueError(
                f"{split_name} split contains duplicate semantic episode content: "
                f"{previous_episode_id} and {answer['episode_id']}"
            )
        seen[signature] = str(answer["episode_id"])


def validate_split_isolation(public_answers: list[dict[str, object]], private_answers: list[dict[str, object]]) -> None:
    validate_answer_uniqueness(public_answers, "public")
    validate_answer_uniqueness(private_answers, "private")

    public_signatures = {episode_signature(answer): str(answer["episode_id"]) for answer in public_answers}
    for answer in private_answers:
        signature = episode_signature(answer)
        public_episode_id = public_signatures.get(signature)
        if public_episode_id is not None:
            raise ValueError(
                "public/private split isolation violated: "
                f"public episode {public_episode_id} overlaps private episode {answer['episode_id']}"
            )

    public_families = {str(answer["transition_family_id"]) for answer in public_answers}
    private_families = {str(answer["transition_family_id"]) for answer in private_answers}
    overlap = sorted(public_families & private_families)
    if overlap:
        raise ValueError(f"public/private transition families overlap: {overlap}")


def load_private_manifest(path: Path = PRIVATE_MANIFEST_PATH) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing private split manifest at {path}. "
            "Create kaggle/dataset/private/private_split_manifest.json with a non-empty private_seed."
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("private split manifest must be a JSON object")

    private_seed = payload.get("private_seed")
    if not isinstance(private_seed, str) or not private_seed.strip():
        raise ValueError("private split manifest must define a non-empty string private_seed")
    return {"private_seed": private_seed.strip()}


def build_split(
    split_name: str,
    variants_per_rule: int,
    variant_start: int = 0,
    private_seed: str | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if split_name == "private" and private_seed is None:
        raise ValueError("private split generation requires a private_seed")

    rows: list[dict[str, object]] = []
    answers: list[dict[str, object]] = []
    episode_counter = 1

    for group_id in GROUPS:
        families = families_for_group(split_name, group_id)
        for variant in range(variant_start, variant_start + variants_per_rule):
            for family in families:
                seed = derive_episode_seed(
                    split_name,
                    group_id,
                    family.family_id,
                    variant,
                    "episode",
                    private_seed,
                )
                rng = random.Random(seed)
                episode_id = f"{episode_counter:04d}"
                row, answer = build_episode(split_name, episode_id, group_id, family, rng)
                rows.append(row)
                answers.append(answer)
                episode_counter += 1
    return (rows, answers)


def sanitize_private_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    sanitized_rows: list[dict[str, object]] = []
    for row in rows:
        sanitized_rows.append(
            {
                "episode_id": row["episode_id"],
                "inference": {
                    "turns": list(row["inference"]["turns"]),
                },
                "analysis": {
                    "faculty_id": row["analysis"]["faculty_id"],
                    "group_id": row["analysis"]["group_id"],
                    "transition_family_id": row["analysis"]["transition_family_id"],
                    "initial_rule_id": row["analysis"]["initial_rule_id"],
                    "shift_rule_id": row["analysis"]["shift_rule_id"],
                    "shift_mode": row["analysis"]["shift_mode"],
                },
            }
        )
    return sanitized_rows


def private_answer_key_payload(private_answers: list[dict[str, object]]) -> dict[str, object]:
    sanitized_answers: list[dict[str, object]] = []
    for answer in private_answers:
        item = dict(answer)
        item.pop("split", None)
        sanitized_answers.append(item)
    return {
        "version": "cogflex_v2_multi_turn",
        "split": "private",
        "episodes": sanitized_answers,
    }


def build_private_artifacts(
    manifest_path: Path = PRIVATE_MANIFEST_PATH,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    manifest = load_private_manifest(manifest_path)
    private_rows, private_answers = build_split(
        "private",
        variants_per_rule=5,
        variant_start=1,
        private_seed=manifest["private_seed"],
    )
    return (private_rows, private_answers, manifest)


def dataset_metadata(dataset_id: str, title: str) -> dict[str, object]:
    return {
        "id": dataset_id,
        "title": title,
        "licenses": [{"name": "CC0-1.0"}],
    }


def validate_rows(
    rows: list[dict[str, object]],
    expected_count: int,
    per_group: int,
    require_scoring: bool,
) -> None:
    assert len(rows) == expected_count, (len(rows), expected_count)
    counts = Counter(str(row["analysis"]["group_id"]) for row in rows)
    assert counts == Counter({group: per_group for group in GROUPS}), counts
    family_counts = Counter(str(row["analysis"]["transition_family_id"]) for row in rows)
    assert family_counts, "transition_family_id must be populated"
    for row in rows:
        turns = row["inference"]["turns"]
        assert isinstance(turns, list) and len(turns) == 3, row["episode_id"]
        for turn_index, turn in enumerate(turns, start=1):
            assert turn.startswith(f"RuleShift cognitive flexibility task. Episode {row['episode_id']}. Turn {turn_index} of 3.")
        if require_scoring:
            targets = row["scoring"]["final_probe_targets"]
            assert len(targets) == FINAL_PROBE_COUNT, row["episode_id"]
        else:
            assert "scoring" not in row, row["episode_id"]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    public_rows, public_answers = build_split("public", variants_per_rule=1)
    private_rows, private_answers, _manifest = build_private_artifacts()

    for row in public_rows:
        row.pop("split", None)
    private_publish_rows = sanitize_private_rows(private_rows)

    validate_rows(public_rows, expected_count=80, per_group=20, require_scoring=True)
    validate_rows(private_publish_rows, expected_count=400, per_group=100, require_scoring=False)
    validate_split_isolation(public_answers, private_answers)

    write_json(PUBLIC_ROWS_PATH, public_rows)
    write_json(PUBLIC_METADATA_PATH, dataset_metadata(PUBLIC_DATASET_ID, "RuleShift CogFlex Runtime v2"))
    write_json(PRIVATE_ROWS_PATH, private_publish_rows)
    write_json(PRIVATE_METADATA_PATH, dataset_metadata(PRIVATE_DATASET_ID, "RuleShift CogFlex Runtime Private v2"))
    write_json(PRIVATE_ANSWER_KEY_PATH, private_answer_key_payload(private_answers))

    print(f"Wrote {len(public_rows)} public episodes to {PUBLIC_ROWS_PATH}")
    print(f"Wrote public dataset metadata to {PUBLIC_METADATA_PATH}")
    print(f"Wrote {len(private_publish_rows)} private inference-only episodes to {PRIVATE_ROWS_PATH}")
    print(f"Wrote private dataset metadata to {PRIVATE_METADATA_PATH}")
    print(f"Wrote private answer key to {PRIVATE_ANSWER_KEY_PATH}")


if __name__ == "__main__":
    main()
