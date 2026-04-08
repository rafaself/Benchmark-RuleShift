#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROWS_PATH = ROOT / "kaggle/dataset/public/public_leaderboard_rows.json"
PUBLIC_METADATA_PATH = ROOT / "kaggle/dataset/public/dataset-metadata.json"
PRIVATE_ROWS_PATH = ROOT / "kaggle/dataset/private/private_leaderboard_rows.json"
PRIVATE_ANSWER_KEY_PATH = ROOT / "kaggle/dataset/private/private_answer_key.json"
PRIVATE_MANIFEST_PATH = ROOT / "kaggle/dataset/private/private_split_manifest.json"
PRIVATE_METADATA_PATH = ROOT / "kaggle/dataset/private/dataset-metadata.json"

PUBLIC_DATASET_ID = "raptorengineer/cogflex-suite-runtime"
PRIVATE_DATASET_ID = "raptorengineer/cogflex-suite-runtime-private"
NOTEBOOK_ID = "raptorengineer/cogflex-suite-notebook"
TASK_NAME = "cogflex_suite_binary"
FACULTY_ID = "executive_functions/cognitive_flexibility"

Point = tuple[int, int, str, str]

VALUES = (-3, -2, -1, 1, 2, 3)
SHAPES = ("circle", "triangle", "square")
TONES = ("warm", "cool")
DOMAIN: list[Point] = [
    (r1, r2, shape, tone)
    for r1 in VALUES
    for r2 in VALUES
    for shape in SHAPES
    for tone in TONES
]
TYPE_TRUE = "type_a"
TYPE_FALSE = "type_b"
SUITE_TASKS = (
    "explicit_rule_update",
    "latent_rule_update",
    "context_binding",
    "trial_cued_switch",
)
PUBLIC_CONTEXTS = ("alpha", "beta")
PRIVATE_CONTEXTS = ("mesa", "fjord")
PUBLIC_CUES = ("sun", "moon")
PRIVATE_CUES = ("copper", "silver")
LEARN_EXAMPLE_COUNT = 4
SHIFT_EXAMPLE_COUNT = 4
FINAL_PROBE_COUNT = 4
PAIR_FAMILY_COUNT = 10
FINAL_OUTPUT_INSTRUCTION = (
    "Return exactly 4 outputs in order, one per probe. "
    "Use only type_a or type_b."
)

SHIFT_MODES = {
    "explicit_rule_update": "explicit_instruction",
    "latent_rule_update": "latent_example_change",
    "context_binding": "context_gate",
    "trial_cued_switch": "cue_switching",
}


@dataclass(frozen=True)
class RuleSpec:
    rule_id: str
    family_id: str
    template_id: str
    description: str
    predicate: Callable[[Point], bool]

    def label(self, point: Point) -> bool:
        return self.predicate(point)


@dataclass(frozen=True)
class TransitionFamily:
    family_id: str
    initial_rule_id: str
    shift_rule_id: str


def fmt_signed(value: int) -> str:
    return f"{value:+d}"


def label_from_bool(value: bool) -> str:
    return TYPE_TRUE if value else TYPE_FALSE


def rule_label(rule: RuleSpec, point: Point) -> str:
    return label_from_bool(rule.label(point))


def make_rule(
    rule_id: str,
    family_id: str,
    template_id: str,
    description: str,
    predicate: Callable[[Point], bool],
) -> RuleSpec:
    return RuleSpec(
        rule_id=rule_id,
        family_id=family_id,
        template_id=template_id,
        description=description,
        predicate=predicate,
    )


def _public_rules() -> list[RuleSpec]:
    return [
        make_rule(
            "axis_r1_ge_-1",
            "axis_threshold",
            "axis_threshold::r1::ge::-1",
            "type_a iff r1 is at least -1",
            lambda point: point[0] >= -1,
        ),
        make_rule(
            "axis_r1_le_1",
            "axis_threshold",
            "axis_threshold::r1::le::+1",
            "type_a iff r1 is at most +1",
            lambda point: point[0] <= 1,
        ),
        make_rule(
            "axis_r2_ge_-1",
            "axis_threshold",
            "axis_threshold::r2::ge::-1",
            "type_a iff r2 is at least -1",
            lambda point: point[1] >= -1,
        ),
        make_rule(
            "axis_r2_le_1",
            "axis_threshold",
            "axis_threshold::r2::le::+1",
            "type_a iff r2 is at most +1",
            lambda point: point[1] <= 1,
        ),
        make_rule(
            "sign_r1_positive",
            "sign",
            "sign::r1_positive",
            "type_a iff r1 is positive",
            lambda point: point[0] > 0,
        ),
        make_rule(
            "sign_r2_positive",
            "sign",
            "sign::r2_positive",
            "type_a iff r2 is positive",
            lambda point: point[1] > 0,
        ),
        make_rule(
            "parity_r1_even",
            "parity",
            "parity::r1_even",
            "type_a iff r1 is even",
            lambda point: point[0] % 2 == 0,
        ),
        make_rule(
            "parity_r2_even",
            "parity",
            "parity::r2_even",
            "type_a iff r2 is even",
            lambda point: point[1] % 2 == 0,
        ),
        make_rule(
            "shape_circle",
            "shape_identity",
            "shape_identity::circle",
            "type_a iff shape is circle",
            lambda point: point[2] == "circle",
        ),
        make_rule(
            "shape_triangle",
            "shape_identity",
            "shape_identity::triangle",
            "type_a iff shape is triangle",
            lambda point: point[2] == "triangle",
        ),
        make_rule(
            "shape_square",
            "shape_identity",
            "shape_identity::square",
            "type_a iff shape is square",
            lambda point: point[2] == "square",
        ),
        make_rule(
            "tone_warm",
            "tone_state",
            "tone_state::warm",
            "type_a iff tone is warm",
            lambda point: point[3] == "warm",
        ),
        make_rule(
            "tone_cool",
            "tone_state",
            "tone_state::cool",
            "type_a iff tone is cool",
            lambda point: point[3] == "cool",
        ),
        make_rule(
            "sign_same_sign",
            "sign",
            "sign::same_sign",
            "type_a iff r1 and r2 share the same sign",
            lambda point: (point[0] > 0) == (point[1] > 0),
        ),
        make_rule(
            "parity_same_parity",
            "parity",
            "parity::same_parity",
            "type_a iff r1 and r2 have the same parity",
            lambda point: (point[0] % 2) == (point[1] % 2),
        ),
    ]


def _private_rules() -> list[RuleSpec]:
    return [
        make_rule(
            "linear_sum_ge_1",
            "linear",
            "linear::sum::ge::+1",
            "type_a iff r1 + r2 is at least +1",
            lambda point: point[0] + point[1] >= 1,
        ),
        make_rule(
            "linear_sum_ge_4",
            "linear",
            "linear::sum::ge::+4",
            "type_a iff r1 + r2 is at least +4",
            lambda point: point[0] + point[1] >= 4,
        ),
        make_rule(
            "linear_sum_le_-1",
            "linear",
            "linear::sum::le::-1",
            "type_a iff r1 + r2 is at most -1",
            lambda point: point[0] + point[1] <= -1,
        ),
        make_rule(
            "linear_diff_ge_1",
            "linear",
            "linear::diff::ge::+1",
            "type_a iff r1 - r2 is at least +1",
            lambda point: point[0] - point[1] >= 1,
        ),
        make_rule(
            "linear_diff_le_-1",
            "linear",
            "linear::diff::le::-1",
            "type_a iff r1 - r2 is at most -1",
            lambda point: point[0] - point[1] <= -1,
        ),
        make_rule(
            "rel_r1_gt_r2",
            "relational",
            "relational::r1_gt_r2",
            "type_a iff r1 is greater than r2",
            lambda point: point[0] > point[1],
        ),
        make_rule(
            "rel_r1_ge_r2_plus_2",
            "relational",
            "relational::r1_ge_r2_plus_2",
            "type_a iff r1 is at least r2 + 2",
            lambda point: point[0] >= point[1] + 2,
        ),
        make_rule(
            "rel_r2_ge_r1_plus_2",
            "relational",
            "relational::r2_ge_r1_plus_2",
            "type_a iff r2 is at least r1 + 2",
            lambda point: point[1] >= point[0] + 2,
        ),
        make_rule(
            "abs_abs_equal",
            "absolute_comparison",
            "absolute::abs_equal",
            "type_a iff |r1| equals |r2|",
            lambda point: abs(point[0]) == abs(point[1]),
        ),
        make_rule(
            "abs_abs_sum_ge_4",
            "absolute_comparison",
            "absolute::abs_sum_ge_4",
            "type_a iff |r1| + |r2| is at least 4",
            lambda point: abs(point[0]) + abs(point[1]) >= 4,
        ),
        make_rule(
            "abs_abs_diff_ge_2",
            "absolute_comparison",
            "absolute::abs_diff_ge_2",
            "type_a iff ||r1| - |r2|| is at least 2",
            lambda point: abs(abs(point[0]) - abs(point[1])) >= 2,
        ),
        make_rule(
            "abs_abs_diff_le_1",
            "absolute_comparison",
            "absolute::abs_diff_le_1",
            "type_a iff ||r1| - |r2|| is at most 1",
            lambda point: abs(abs(point[0]) - abs(point[1])) <= 1,
        ),
        make_rule(
            "cross_tone_matches_r1_sign",
            "cross_feature_binding",
            "cross_feature_binding::tone_matches_r1_sign",
            "type_a iff tone=warm matches whether r1 is positive",
            lambda point: (point[3] == "warm") == (point[0] > 0),
        ),
        make_rule(
            "cross_tone_matches_r2_sign",
            "cross_feature_binding",
            "cross_feature_binding::tone_matches_r2_sign",
            "type_a iff tone=warm matches whether r2 is positive",
            lambda point: (point[3] == "warm") == (point[1] > 0),
        ),
        make_rule(
            "cross_shape_square_or_sum_ge_2",
            "cross_feature_binding",
            "cross_feature_binding::shape_square_or_sum_ge_2",
            "type_a iff shape is square or r1 + r2 is at least +2",
            lambda point: point[2] == "square" or point[0] + point[1] >= 2,
        ),
        make_rule(
            "cross_shape_circle_xor_tone_warm",
            "cross_feature_binding",
            "cross_feature_binding::shape_circle_xor_tone_warm",
            "type_a iff exactly one of shape=circle and tone=warm is true",
            lambda point: (point[2] == "circle") ^ (point[3] == "warm"),
        ),
        make_rule(
            "gate_tone_warm_then_r1_ge_1_else_r2_ge_1",
            "feature_gate",
            "feature_gate::tone_warm_then_r1_ge_1_else_r2_ge_1",
            "type_a iff tone=warm uses r1>=+1 and tone=cool uses r2>=+1",
            lambda point: point[0] >= 1 if point[3] == "warm" else point[1] >= 1,
        ),
        make_rule(
            "gate_shape_circle_then_sum_positive_else_diff_positive",
            "feature_gate",
            "feature_gate::shape_circle_then_sum_positive_else_diff_positive",
            "type_a iff shape=circle uses r1+r2>0 and otherwise uses r1-r2>0",
            lambda point: (point[0] + point[1] > 0) if point[2] == "circle" else (point[0] - point[1] > 0),
        ),
    ]


def _rule_signature(rule: RuleSpec) -> tuple[bool, ...]:
    return tuple(rule.label(point) for point in DOMAIN)


def _dedupe_rules(rules: Iterable[RuleSpec]) -> tuple[RuleSpec, ...]:
    unique: list[RuleSpec] = []
    seen: set[tuple[bool, ...]] = set()
    for rule in rules:
        signature = _rule_signature(rule)
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(rule)
    return tuple(unique)


PUBLIC_RULES = _dedupe_rules(_public_rules())
PRIVATE_RULES = _dedupe_rules(_private_rules())
ALL_RULES = PUBLIC_RULES + PRIVATE_RULES
RULE_BY_ID = {rule.rule_id: rule for rule in ALL_RULES}


def disagreement_count(initial_rule_id: str, shift_rule_id: str) -> int:
    initial_rule = RULE_BY_ID[initial_rule_id]
    shift_rule = RULE_BY_ID[shift_rule_id]
    return sum(initial_rule.label(point) != shift_rule.label(point) for point in DOMAIN)


def build_pair_families(split_name: str, rules: tuple[RuleSpec, ...], count: int) -> tuple[TransitionFamily, ...]:
    offsets = (5, 7, 3, 1, 9, 11, 13)
    rule_ids = [rule.rule_id for rule in rules]
    for offset in offsets:
        families: list[TransitionFamily] = []
        seen: set[tuple[str, str]] = set()
        for index, initial_rule_id in enumerate(rule_ids):
            shift_rule_id = rule_ids[(index + offset) % len(rule_ids)]
            if initial_rule_id == shift_rule_id:
                continue
            pair_key = (initial_rule_id, shift_rule_id)
            if pair_key in seen:
                continue
            if disagreement_count(initial_rule_id, shift_rule_id) < 8:
                continue
            seen.add(pair_key)
            families.append(
                TransitionFamily(
                    family_id=f"{split_name}::pair::{initial_rule_id}__to__{shift_rule_id}",
                    initial_rule_id=initial_rule_id,
                    shift_rule_id=shift_rule_id,
                )
            )
        if len(families) >= count:
            return tuple(families[:count])
    raise RuntimeError(f"Unable to derive {count} transition families for split {split_name}")


PUBLIC_PAIR_FAMILIES = build_pair_families("public", PUBLIC_RULES, PAIR_FAMILY_COUNT)
PRIVATE_PAIR_FAMILIES = build_pair_families("private", PRIVATE_RULES, PAIR_FAMILY_COUNT)


def families_for_split(split_name: str) -> tuple[TransitionFamily, ...]:
    return PUBLIC_PAIR_FAMILIES if split_name == "public" else PRIVATE_PAIR_FAMILIES


def contexts_for_split(split_name: str) -> tuple[str, str]:
    return PUBLIC_CONTEXTS if split_name == "public" else PRIVATE_CONTEXTS


def cues_for_split(split_name: str) -> tuple[str, str]:
    return PUBLIC_CUES if split_name == "public" else PRIVATE_CUES


def derive_episode_seed(
    split_name: str,
    suite_task_id: str,
    family_id: str,
    variant: int,
    purpose: str,
    private_seed: str | None,
) -> int:
    parts = [split_name, suite_task_id, family_id, str(variant), purpose]
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
    candidates: list[Point],
    count: int,
    predicate: Callable[[list[Point]], bool],
    *,
    attempts: int = 6000,
) -> list[Point]:
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
    point: Point,
    label: str,
    *,
    context: str | None = None,
    cue: str | None = None,
    active_rule_id: str | None = None,
) -> dict[str, object]:
    item: dict[str, object] = {
        "index": index,
        "r1": point[0],
        "r2": point[1],
        "shape": point[2],
        "tone": point[3],
        "label": label,
    }
    if context is not None:
        item["context"] = context
    if cue is not None:
        item["cue"] = cue
    if active_rule_id is not None:
        item["active_rule_id"] = active_rule_id
    return item


def render_case_text(point: Point, *, context: str | None = None, cue: str | None = None) -> str:
    parts: list[str] = []
    if context is not None:
        parts.append(f"context={context}")
    if cue is not None:
        parts.append(f"cue={cue}")
    parts.append(f"shape={point[2]}")
    parts.append(f"tone={point[3]}")
    parts.append(f"r1={fmt_signed(point[0])}, r2={fmt_signed(point[1])}")
    return " | ".join(parts)


def render_labeled_lines(items: list[dict[str, object]]) -> str:
    return "\n".join(
        f"{item['index']}. "
        f"{render_case_text((int(item['r1']), int(item['r2']), str(item['shape']), str(item['tone'])), context=item.get('context'), cue=item.get('cue'))} "
        f"-> {item['label']}"
        for item in items
    )


def render_probe_lines(items: list[dict[str, object]]) -> str:
    return "\n".join(
        f"{item['index']}. "
        f"{render_case_text((int(item['r1']), int(item['r2']), str(item['shape']), str(item['tone'])), context=item.get('context'), cue=item.get('cue'))} "
        "-> ?"
        for item in items
    )


def render_turn_header(episode_id: str, turn_index: int) -> str:
    return f"CogFlex suite task. Episode {episode_id}. Turn {turn_index} of 3."


def render_learn_turn(
    split_name: str,
    episode_id: str,
    suite_task_id: str,
    examples: list[dict[str, object]],
    *,
    surface_variant: int,
) -> str:
    contexts = contexts_for_split(split_name)
    if suite_task_id == "context_binding":
        variants = (
            f"Learn the active rule for context={contexts[0]} from these labeled examples.",
            f"Use these labeled items to identify the rule that applies when context={contexts[0]}.",
            f"Study these labeled cases and infer the current rule for context={contexts[0]}.",
        )
    else:
        variants = (
            "Learn the current classification rule from these labeled examples.",
            "Infer the active rule from these labeled examples.",
            "Use these labeled items to determine the live classification rule.",
        )
    guidance = variants[surface_variant % len(variants)]
    return "\n\n".join(
        [
            render_turn_header(episode_id, 1),
            guidance,
            "Examples:\n" + render_labeled_lines(examples),
        ]
    )


def render_shift_turn(
    episode_id: str,
    split_name: str,
    suite_task_id: str,
    examples: list[dict[str, object]],
    *,
    surface_variant: int,
) -> str:
    contexts = contexts_for_split(split_name)
    if suite_task_id == "explicit_rule_update":
        variants = (
            "The active rule has changed. Update your rule from these labeled examples.",
            "The rule is different now. Revise it using these labeled examples.",
            "A rule shift has occurred. Infer the new rule from these labeled examples.",
        )
    elif suite_task_id == "latent_rule_update":
        variants = (
            "Continue the task using these labeled examples.",
            "Proceed with the task and use these labeled examples to stay calibrated.",
            "Continue from the latest evidence using these labeled examples.",
        )
    elif suite_task_id == "context_binding":
        variants = (
            f"Now learn the active rule for context={contexts[1]} from these labeled examples.",
            f"Switch attention to context={contexts[1]} and infer its rule from these labeled examples.",
            f"These labeled examples define the active rule for context={contexts[1]}.",
        )
    elif suite_task_id == "trial_cued_switch":
        cues = cues_for_split(split_name)
        variants = (
            "From now on, each item includes a cue. "
            f"cue={cues[0]} keeps the original rule. "
            f"cue={cues[1]} uses the alternate rule. "
            "Infer the alternate rule from these labeled examples.",
            "A cue now selects which rule applies. "
            f"cue={cues[0]} means keep the original rule, and cue={cues[1]} means use the alternate rule. "
            "Infer that alternate rule from these labeled examples.",
            "Items now carry a cue. "
            f"When cue={cues[0]}, stay with the initial rule; when cue={cues[1]}, apply the alternate rule. "
            "Use these labeled examples to infer the alternate rule.",
        )
    else:
        raise ValueError(f"Unsupported suite_task_id {suite_task_id}")
    guidance = variants[surface_variant % len(variants)]
    return "\n\n".join(
        [
            render_turn_header(episode_id, 2),
            guidance,
            "Examples:\n" + render_labeled_lines(examples),
        ]
    )


def render_decision_turn(
    episode_id: str,
    suite_task_id: str,
    probes: list[dict[str, object]],
    *,
    surface_variant: int,
) -> str:
    if suite_task_id == "context_binding":
        variants = (
            "Classify each probe with the active rule for that probe's context.",
            "For each probe, use the rule that matches its context.",
            "Choose each label by applying the rule bound to that probe's context.",
        )
    elif suite_task_id == "trial_cued_switch":
        variants = (
            "Classify each probe with the rule selected by that probe's cue.",
            "Use each probe's cue to choose the correct rule before labeling it.",
            "For each probe, apply the rule indicated by its cue.",
        )
    else:
        variants = (
            "Classify each probe with the currently active rule.",
            "Apply the active rule to each probe.",
            "Label each probe using the rule that is active now.",
        )
    guidance = variants[surface_variant % len(variants)]
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
    exclude: set[Point] | None = None,
) -> list[Point]:
    excluded = set() if exclude is None else exclude
    candidates = [point for point in DOMAIN if point not in excluded]
    return pick_points(
        rng,
        candidates,
        count,
        lambda sample: has_label_balance([rule_label(rule, point) for point in sample]),
    )


def build_rule_examples(
    points: list[Point],
    rule: RuleSpec,
    *,
    start_index: int,
    context: str | None = None,
    cue: str | None = None,
) -> list[dict[str, object]]:
    return [
        serialize_case(index, point, rule_label(rule, point), context=context, cue=cue, active_rule_id=rule.rule_id)
        for index, point in enumerate(points, start=start_index)
    ]


def build_transition_episode(
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
        if len(disagreement_points) < FINAL_PROBE_COUNT + SHIFT_EXAMPLE_COUNT:
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
        probe_points = pick_points(rng, remaining_disagreement, FINAL_PROBE_COUNT, lambda sample: True)
        learn_examples = build_rule_examples(initial_points, initial_rule, start_index=1)
        shift_examples = build_rule_examples(shift_points, shift_rule, start_index=1)
        probes = [
            serialize_case(
                index,
                point,
                rule_label(shift_rule, point),
                active_rule_id=shift_rule.rule_id,
            )
            for index, point in enumerate(probe_points, start=1)
        ]
        return learn_examples, shift_examples, probes
    raise RuntimeError(f"Unable to build transition episode for family {family.family_id}")


def build_latent_transition_episode(
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
        agreement_points = [
            point for point in DOMAIN if initial_rule.label(point) == shift_rule.label(point) and point not in used
        ]
        if len(disagreement_points) < FINAL_PROBE_COUNT + 2 or len(agreement_points) < 2:
            continue
        try:
            latent_conflicts = pick_points(
                rng,
                disagreement_points,
                2,
                lambda sample: has_label_balance([rule_label(shift_rule, point) for point in sample], 1, 0),
            )
        except RuntimeError:
            continue
        used.update(latent_conflicts)
        try:
            latent_agreements = pick_points(
                rng,
                [point for point in agreement_points if point not in used],
                2,
                lambda sample: has_label_balance(
                    [rule_label(shift_rule, point) for point in latent_conflicts + sample],
                    1,
                    1,
                ),
            )
        except RuntimeError:
            continue
        used.update(latent_agreements)
        remaining_disagreement = [point for point in disagreement_points if point not in used]
        if len(remaining_disagreement) < FINAL_PROBE_COUNT:
            continue
        probe_points = pick_points(rng, remaining_disagreement, FINAL_PROBE_COUNT, lambda sample: True)
        shift_points = latent_conflicts + latent_agreements
        rng.shuffle(shift_points)
        learn_examples = build_rule_examples(initial_points, initial_rule, start_index=1)
        shift_examples = build_rule_examples(shift_points, shift_rule, start_index=1)
        probes = [
            serialize_case(
                index,
                point,
                rule_label(shift_rule, point),
                active_rule_id=shift_rule.rule_id,
            )
            for index, point in enumerate(probe_points, start=1)
        ]
        return learn_examples, shift_examples, probes
    raise RuntimeError(f"Unable to build latent transition episode for family {family.family_id}")


def build_context_binding_episode(
    split_name: str,
    family: TransitionFamily,
    rng: random.Random,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    initial_rule = RULE_BY_ID[family.initial_rule_id]
    shift_rule = RULE_BY_ID[family.shift_rule_id]
    contexts = contexts_for_split(split_name)
    for _ in range(4000):
        initial_points = pick_balanced_examples(rng, initial_rule, LEARN_EXAMPLE_COUNT)
        used = set(initial_points)
        try:
            shift_points = pick_balanced_examples(rng, shift_rule, SHIFT_EXAMPLE_COUNT, exclude=used)
        except RuntimeError:
            continue
        used.update(shift_points)
        disagreement_points = [
            point for point in DOMAIN if initial_rule.label(point) != shift_rule.label(point) and point not in used
        ]
        if len(disagreement_points) < FINAL_PROBE_COUNT:
            continue
        alpha_points = pick_points(rng, disagreement_points, 2, lambda sample: True)
        used.update(alpha_points)
        remaining = [point for point in disagreement_points if point not in used]
        if len(remaining) < 2:
            continue
        beta_points = pick_points(rng, remaining, 2, lambda sample: True)
        probe_specs = [(contexts[0], point) for point in alpha_points] + [(contexts[1], point) for point in beta_points]
        rng.shuffle(probe_specs)
        learn_examples = build_rule_examples(initial_points, initial_rule, start_index=1, context=contexts[0])
        shift_examples = build_rule_examples(shift_points, shift_rule, start_index=1, context=contexts[1])
        probes = []
        for index, (context, point) in enumerate(probe_specs, start=1):
            active_rule = initial_rule if context == contexts[0] else shift_rule
            probes.append(
                serialize_case(
                    index,
                    point,
                    rule_label(active_rule, point),
                    context=context,
                    active_rule_id=active_rule.rule_id,
                )
            )
        return learn_examples, shift_examples, probes
    raise RuntimeError(f"Unable to build context-binding episode for family {family.family_id}")


def build_trial_cued_switch_episode(
    split_name: str,
    family: TransitionFamily,
    rng: random.Random,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    initial_rule = RULE_BY_ID[family.initial_rule_id]
    shift_rule = RULE_BY_ID[family.shift_rule_id]
    keep_cue, switch_cue = cues_for_split(split_name)
    for _ in range(4000):
        initial_points = pick_balanced_examples(rng, initial_rule, LEARN_EXAMPLE_COUNT)
        used = set(initial_points)
        disagreement_points = [
            point for point in DOMAIN if initial_rule.label(point) != shift_rule.label(point) and point not in used
        ]
        if len(disagreement_points) < FINAL_PROBE_COUNT:
            continue
        try:
            keep_points = pick_points(
                rng,
                [point for point in DOMAIN if point not in used],
                2,
                lambda sample: has_label_balance([rule_label(initial_rule, point) for point in sample], 1, 0),
            )
        except RuntimeError:
            continue
        used.update(keep_points)
        remaining_for_shift = [point for point in DOMAIN if point not in used]
        try:
            switch_points = pick_points(
                rng,
                remaining_for_shift,
                2,
                lambda sample: (
                    sum(point in disagreement_points for point in sample) >= 1
                    and has_label_balance([rule_label(shift_rule, point) for point in sample], 1, 0)
                ),
            )
        except RuntimeError:
            continue
        used.update(switch_points)
        remaining_disagreement = [point for point in disagreement_points if point not in used]
        if len(remaining_disagreement) < FINAL_PROBE_COUNT:
            continue
        keep_probe_points = pick_points(rng, remaining_disagreement, 2, lambda sample: True)
        used.update(keep_probe_points)
        remaining_disagreement = [point for point in remaining_disagreement if point not in used]
        if len(remaining_disagreement) < 2:
            continue
        switch_probe_points = pick_points(rng, remaining_disagreement, 2, lambda sample: True)

        learn_examples = build_rule_examples(initial_points, initial_rule, start_index=1)
        shift_examples = (
            build_rule_examples(keep_points, initial_rule, start_index=1, cue=keep_cue)
            + build_rule_examples(switch_points, shift_rule, start_index=3, cue=switch_cue)
        )
        rng.shuffle(shift_examples)
        for index, item in enumerate(shift_examples, start=1):
            item["index"] = index

        probe_specs = [(keep_cue, point, initial_rule) for point in keep_probe_points] + [
            (switch_cue, point, shift_rule) for point in switch_probe_points
        ]
        rng.shuffle(probe_specs)
        probes = []
        for index, (cue, point, active_rule) in enumerate(probe_specs, start=1):
            probes.append(
                serialize_case(
                    index,
                    point,
                    rule_label(active_rule, point),
                    cue=cue,
                    active_rule_id=active_rule.rule_id,
                )
            )
        return learn_examples, shift_examples, probes
    raise RuntimeError(f"Unable to build trial-cued-switch episode for family {family.family_id}")


def _labels_for_points(rule: RuleSpec, points: Iterable[Point]) -> list[str]:
    return [rule_label(rule, point) for point in points]


def point_from_item(item: dict[str, object]) -> Point:
    return (int(item["r1"]), int(item["r2"]), str(item["shape"]), str(item["tone"]))


def _consistent_rules(examples: list[dict[str, object]]) -> list[RuleSpec]:
    matches: list[RuleSpec] = []
    for rule in ALL_RULES:
        if all(rule_label(rule, point_from_item(item)) == str(item["label"]) for item in examples):
            matches.append(rule)
    return matches


def _majority_vote(predictions: list[tuple[str, ...]]) -> tuple[str, ...]:
    if not predictions:
        raise RuntimeError("No hypothesis predictions available for symbolic baseline")
    labels: list[str] = []
    for index in range(FINAL_PROBE_COUNT):
        counts = Counter(prediction[index] for prediction in predictions)
        labels.append(TYPE_TRUE if counts[TYPE_TRUE] >= counts[TYPE_FALSE] else TYPE_FALSE)
    return tuple(labels)


def score_labels(predictions: Iterable[str], targets: Iterable[str]) -> float:
    pred_tuple = tuple(predictions)
    target_tuple = tuple(targets)
    return sum(pred == target for pred, target in zip(pred_tuple, target_tuple)) / len(target_tuple)


def count_rule_conflicts(rule: RuleSpec, items: list[dict[str, object]]) -> int:
    return sum(rule_label(rule, point_from_item(item)) != str(item["label"]) for item in items)


def case_distance(source: dict[str, object], target: dict[str, object]) -> int:
    distance = abs(int(source["r1"]) - int(target["r1"])) + abs(int(source["r2"]) - int(target["r2"]))
    distance += 2 * (str(source["shape"]) != str(target["shape"]))
    distance += 2 * (str(source["tone"]) != str(target["tone"]))
    distance += 4 * (str(source.get("context", "")) != str(target.get("context", "")))
    distance += 4 * (str(source.get("cue", "")) != str(target.get("cue", "")))
    return distance


def nearest_neighbor_predictions(
    examples: list[dict[str, object]],
    probes: list[dict[str, object]],
) -> tuple[str, ...]:
    predictions: list[str] = []
    for probe in probes:
        ranked = sorted(
            examples,
            key=lambda item: (
                case_distance(item, probe),
                -int(item["index"]),
            ),
        )
        predictions.append(str(ranked[0]["label"]))
    return tuple(predictions)


def symbolic_diagnostics(
    suite_task_id: str,
    learn_examples: list[dict[str, object]],
    shift_examples: list[dict[str, object]],
    probes: list[dict[str, object]],
    *,
    initial_rule: RuleSpec,
    shift_rule: RuleSpec,
    split_name: str,
) -> dict[str, object]:
    targets = tuple(str(item["label"]) for item in probes)
    previous_predictions = tuple(rule_label(initial_rule, point_from_item(item)) for item in probes)
    majority_counts = Counter(str(item["label"]) for item in learn_examples + shift_examples)
    majority_label = TYPE_TRUE if majority_counts[TYPE_TRUE] >= majority_counts[TYPE_FALSE] else TYPE_FALSE
    majority_predictions = (majority_label,) * FINAL_PROBE_COUNT
    nearest_predictions = nearest_neighbor_predictions(learn_examples + shift_examples, probes)

    hypotheses: list[tuple[str, ...]] = []
    learning_hypothesis_size = 0
    cue_agnostic_accuracy: float | None = None

    if suite_task_id in {"explicit_rule_update", "latent_rule_update"}:
        initial_candidates = _consistent_rules(learn_examples)
        shift_candidates = _consistent_rules(shift_examples)
        learning_hypothesis_size = len(shift_candidates)
        hypotheses = [
            tuple(rule_label(shift_candidate, point_from_item(item)) for item in probes)
            for initial_candidate in initial_candidates
            for shift_candidate in shift_candidates
            if initial_candidate.rule_id != shift_candidate.rule_id
        ]
    elif suite_task_id == "context_binding":
        contexts = contexts_for_split(split_name)
        alpha_candidates = _consistent_rules(learn_examples)
        beta_candidates = _consistent_rules(shift_examples)
        learning_hypothesis_size = len(alpha_candidates) * len(beta_candidates)
        hypotheses = []
        for alpha_rule in alpha_candidates:
            for beta_rule in beta_candidates:
                predictions = []
                for item in probes:
                    active_rule = alpha_rule if item.get("context") == contexts[0] else beta_rule
                    predictions.append(rule_label(active_rule, point_from_item(item)))
                hypotheses.append(tuple(predictions))
        one_rule_initial = tuple(rule_label(initial_rule, point_from_item(item)) for item in probes)
        one_rule_shift = tuple(rule_label(shift_rule, point_from_item(item)) for item in probes)
        cue_agnostic_accuracy = max(score_labels(one_rule_initial, targets), score_labels(one_rule_shift, targets))
    elif suite_task_id == "trial_cued_switch":
        keep_cue, switch_cue = cues_for_split(split_name)
        initial_candidates = _consistent_rules(learn_examples)
        shift_candidates = []
        for rule in ALL_RULES:
            if all(
                rule_label(initial_rule if item.get("cue") == keep_cue else rule, point_from_item(item))
                == str(item["label"])
                for item in shift_examples
            ):
                shift_candidates.append(rule)
        learning_hypothesis_size = len(initial_candidates) * len(shift_candidates)
        hypotheses = []
        for initial_candidate in initial_candidates:
            for shift_candidate in shift_candidates:
                predictions = []
                for item in probes:
                    active_rule = initial_candidate if item.get("cue") == keep_cue else shift_candidate
                    predictions.append(rule_label(active_rule, point_from_item(item)))
                hypotheses.append(tuple(predictions))
        one_rule_initial = tuple(rule_label(initial_rule, point_from_item(item)) for item in probes)
        one_rule_shift = tuple(rule_label(shift_rule, point_from_item(item)) for item in probes)
        cue_agnostic_accuracy = max(score_labels(one_rule_initial, targets), score_labels(one_rule_shift, targets))
    else:
        raise ValueError(f"Unsupported suite_task_id {suite_task_id}")

    if not hypotheses:
        raise RuntimeError(f"No symbolic hypotheses for suite task {suite_task_id}")

    symbolic_majority_predictions = _majority_vote(hypotheses)
    symbolic_majority_accuracy = score_labels(symbolic_majority_predictions, targets)
    unanimous_predictions = len(set(hypotheses)) == 1
    previous_rule_accuracy = score_labels(previous_predictions, targets)
    majority_label_accuracy = score_labels(majority_predictions, targets)
    nearest_neighbor_accuracy = score_labels(nearest_predictions, targets)
    adversarial_baseline_accuracy = max(
        previous_rule_accuracy,
        majority_label_accuracy,
        nearest_neighbor_accuracy,
        symbolic_majority_accuracy,
        cue_agnostic_accuracy or 0.0,
    )

    return {
        "previous_rule_accuracy": previous_rule_accuracy,
        "majority_label_accuracy": majority_label_accuracy,
        "nearest_neighbor_accuracy": nearest_neighbor_accuracy,
        "cue_agnostic_accuracy": cue_agnostic_accuracy,
        "symbolic_hypothesis_count": len(hypotheses),
        "learning_hypothesis_size": learning_hypothesis_size,
        "symbolic_majority_accuracy": symbolic_majority_accuracy,
        "symbolic_unanimous_predictions": unanimous_predictions,
        "shift_evidence_conflict_count": count_rule_conflicts(initial_rule, shift_examples),
        "shift_evidence_agreement_count": len(shift_examples) - count_rule_conflicts(initial_rule, shift_examples),
        "adversarial_baseline_accuracy": adversarial_baseline_accuracy,
        "difficulty_bin": "hard" if adversarial_baseline_accuracy <= 0.5 else "medium",
    }


def validate_episode_constraints(answer: dict[str, object]) -> None:
    suite_task_id = str(answer["suite_task_id"])
    diagnostics = answer["generator_diagnostics"]
    previous_rule_accuracy = float(diagnostics["previous_rule_accuracy"])
    cue_agnostic_accuracy = diagnostics["cue_agnostic_accuracy"]
    symbolic_accuracy = float(diagnostics["symbolic_majority_accuracy"])
    adversarial_accuracy = float(diagnostics["adversarial_baseline_accuracy"])
    shift_conflicts = int(diagnostics["shift_evidence_conflict_count"])

    if suite_task_id in {"explicit_rule_update", "latent_rule_update"} and previous_rule_accuracy > 0.25:
        raise ValueError(
            f"episode {answer['episode_id']} violates previous-rule ceiling: {previous_rule_accuracy:.2f}"
        )
    if suite_task_id == "explicit_rule_update" and shift_conflicts < 3:
        raise ValueError(f"episode {answer['episode_id']} lacks strong explicit shift evidence: {shift_conflicts}")
    if suite_task_id == "latent_rule_update" and shift_conflicts not in {1, 2}:
        raise ValueError(f"episode {answer['episode_id']} lacks calibrated latent shift evidence: {shift_conflicts}")
    if suite_task_id in {"context_binding", "trial_cued_switch"}:
        if cue_agnostic_accuracy is None or cue_agnostic_accuracy > 0.5:
            raise ValueError(
                f"episode {answer['episode_id']} violates cue/context ceiling: {cue_agnostic_accuracy!r}"
            )
    if symbolic_accuracy > 0.75:
        raise ValueError(
            f"episode {answer['episode_id']} violates symbolic-majority ceiling: {symbolic_accuracy:.2f}"
        )
    if adversarial_accuracy > 0.85:
        raise ValueError(
            f"episode {answer['episode_id']} violates adversarial-baseline ceiling: {adversarial_accuracy:.2f}"
        )


def episode_record(
    split_name: str,
    episode_id: str,
    suite_task_id: str,
    family: TransitionFamily,
    learn_examples: list[dict[str, object]],
    shift_examples: list[dict[str, object]],
    final_probes: list[dict[str, object]],
    diagnostics: dict[str, object],
    *,
    surface_variant: int,
) -> tuple[dict[str, object], dict[str, object]]:
    initial_rule = RULE_BY_ID[family.initial_rule_id]
    shift_rule = RULE_BY_ID[family.shift_rule_id]
    turns = [
        render_learn_turn(split_name, episode_id, suite_task_id, learn_examples, surface_variant=surface_variant),
        render_shift_turn(episode_id, split_name, suite_task_id, shift_examples, surface_variant=surface_variant),
        render_decision_turn(episode_id, suite_task_id, final_probes, surface_variant=surface_variant),
    ]
    analysis = {
        "faculty_id": FACULTY_ID,
        "suite_task_id": suite_task_id,
        "shift_mode": SHIFT_MODES[suite_task_id],
        "difficulty_bin": str(diagnostics["difficulty_bin"]),
    }
    cue_template_id = "none"
    context_template_id = "none"
    if suite_task_id == "trial_cued_switch":
        cue_template_id = "cue_template::public::celestial" if split_name == "public" else "cue_template::private::metals"
    if suite_task_id == "context_binding":
        context_template_id = (
            "context_template::public::alpha_beta" if split_name == "public" else "context_template::private::mesa_fjord"
        )
    answer = {
        "episode_id": episode_id,
        **analysis,
        "transition_family_id": family.family_id,
        "initial_rule_id": family.initial_rule_id,
        "shift_rule_id": family.shift_rule_id,
        "initial_rule_family_id": initial_rule.family_id,
        "shift_rule_family_id": shift_rule.family_id,
        "initial_rule_template_id": initial_rule.template_id,
        "shift_rule_template_id": shift_rule.template_id,
        "cue_template_id": cue_template_id,
        "context_template_id": context_template_id,
        "surface_template_id": f"surface_template::{surface_variant}",
        "learn_turn_examples": learn_examples,
        "shift_turn_examples": shift_examples,
        "final_probes": final_probes,
        "final_probe_targets": [str(item["label"]) for item in final_probes],
        "turns": turns,
        "generator_diagnostics": diagnostics,
        "split": split_name,
    }
    row = {
        "episode_id": episode_id,
        "inference": {"turns": turns},
        "scoring": {"final_probe_targets": [str(item["label"]) for item in final_probes]},
        "analysis": analysis,
        "split": split_name,
    }
    return row, answer


def build_episode(
    split_name: str,
    episode_id: str,
    suite_task_id: str,
    family: TransitionFamily,
    variant: int,
    base_seed: int,
) -> tuple[dict[str, object], dict[str, object]]:
    initial_rule = RULE_BY_ID[family.initial_rule_id]
    shift_rule = RULE_BY_ID[family.shift_rule_id]
    target_bin = "hard" if variant % 2 == 0 else "medium"
    surface_variant = (base_seed + variant) % 3

    for attempt in range(6000):
        rng = random.Random(base_seed + attempt)
        if suite_task_id == "explicit_rule_update":
            learn_examples, shift_examples, final_probes = build_transition_episode(family, rng)
        elif suite_task_id == "latent_rule_update":
            learn_examples, shift_examples, final_probes = build_latent_transition_episode(family, rng)
        elif suite_task_id == "context_binding":
            learn_examples, shift_examples, final_probes = build_context_binding_episode(split_name, family, rng)
        elif suite_task_id == "trial_cued_switch":
            learn_examples, shift_examples, final_probes = build_trial_cued_switch_episode(split_name, family, rng)
        else:
            raise ValueError(f"Unsupported suite_task_id {suite_task_id}")

        diagnostics = symbolic_diagnostics(
            suite_task_id,
            learn_examples,
            shift_examples,
            final_probes,
            initial_rule=initial_rule,
            shift_rule=shift_rule,
            split_name=split_name,
        )
        if diagnostics["difficulty_bin"] != target_bin:
            continue
        row, answer = episode_record(
            split_name,
            episode_id,
            suite_task_id,
            family,
            learn_examples,
            shift_examples,
            final_probes,
            diagnostics,
            surface_variant=surface_variant,
        )
        try:
            validate_episode_constraints(answer)
        except ValueError:
            continue
        return row, answer
    raise RuntimeError(
        f"Unable to build {suite_task_id} episode for family {family.family_id} with target bin {target_bin}"
    )


def episode_signature(answer: dict[str, object]) -> tuple[object, ...]:
    return (
        answer["suite_task_id"],
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

    public_rule_templates = {
        (str(answer["initial_rule_template_id"]), str(answer["shift_rule_template_id"])) for answer in public_answers
    }
    private_rule_templates = {
        (str(answer["initial_rule_template_id"]), str(answer["shift_rule_template_id"])) for answer in private_answers
    }
    if public_rule_templates & private_rule_templates:
        raise ValueError("public/private rule-template fingerprints overlap")

    public_cues = {str(answer["cue_template_id"]) for answer in public_answers}
    private_cues = {str(answer["cue_template_id"]) for answer in private_answers}
    overlap = sorted((public_cues & private_cues) - {"none"})
    if overlap:
        raise ValueError(f"public/private cue-template fingerprints overlap: {overlap}")

    public_contexts = {str(answer["context_template_id"]) for answer in public_answers}
    private_contexts = {str(answer["context_template_id"]) for answer in private_answers}
    context_overlap = sorted((public_contexts & private_contexts) - {"none"})
    if context_overlap:
        raise ValueError(f"public/private context-template fingerprints overlap: {context_overlap}")


def validate_split_quality(answers: list[dict[str, object]], split_name: str) -> None:
    by_task: dict[str, list[dict[str, object]]] = defaultdict(list)
    symbolic_numerator = 0.0
    adversarial_numerator = 0.0
    denominator = 0
    for answer in answers:
        diagnostics = answer["generator_diagnostics"]
        symbolic_acc = float(diagnostics["symbolic_majority_accuracy"])
        adversarial_acc = float(diagnostics["adversarial_baseline_accuracy"])
        symbolic_numerator += symbolic_acc * FINAL_PROBE_COUNT
        adversarial_numerator += adversarial_acc * FINAL_PROBE_COUNT
        denominator += FINAL_PROBE_COUNT
        by_task[str(answer["suite_task_id"])].append(answer)

    symbolic_overall = symbolic_numerator / denominator
    if symbolic_overall > 0.58:
        raise ValueError(
            f"{split_name} split symbolic-majority micro accuracy is too high: {symbolic_overall:.3f}"
        )
    adversarial_overall = adversarial_numerator / denominator
    if adversarial_overall > 0.65:
        raise ValueError(
            f"{split_name} split adversarial-baseline micro accuracy is too high: {adversarial_overall:.3f}"
        )

    for suite_task_id in SUITE_TASKS:
        episodes = by_task[suite_task_id]
        symbolic_task_acc = (
            sum(float(answer["generator_diagnostics"]["symbolic_majority_accuracy"]) for answer in episodes) / len(episodes)
        )
        if symbolic_task_acc > 0.60:
            raise ValueError(
                f"{split_name} split symbolic-majority task accuracy is too high for {suite_task_id}: {symbolic_task_acc:.3f}"
            )
        adversarial_task_acc = (
            sum(float(answer["generator_diagnostics"]["adversarial_baseline_accuracy"]) for answer in episodes)
            / len(episodes)
        )
        if adversarial_task_acc > 0.65:
            raise ValueError(
                f"{split_name} split adversarial-baseline task accuracy is too high for {suite_task_id}: {adversarial_task_acc:.3f}"
            )

        ambiguous = sum(int(answer["generator_diagnostics"]["learning_hypothesis_size"]) > 1 for answer in episodes)
        if ambiguous < len(episodes) / 2:
            raise ValueError(
                f"{split_name} split lacks calibrated ambiguity for {suite_task_id}: "
                f"{ambiguous}/{len(episodes)} episodes exceed one learning hypothesis"
            )

        unanimous = sum(bool(answer["generator_diagnostics"]["symbolic_unanimous_predictions"]) for answer in episodes)
        if unanimous > len(episodes) / 2:
            raise ValueError(
                f"{split_name} split is too unanimous for {suite_task_id}: "
                f"{unanimous}/{len(episodes)} episodes collapse to one symbolic prediction"
            )


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
    variants_per_family: int,
    variant_start: int = 0,
    private_seed: str | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if split_name == "private" and private_seed is None:
        raise ValueError("private split generation requires a private_seed")

    rows: list[dict[str, object]] = []
    answers: list[dict[str, object]] = []
    episode_counter = 1
    families = families_for_split(split_name)

    for suite_task_id in SUITE_TASKS:
        for variant in range(variant_start, variant_start + variants_per_family):
            for family in families:
                base_seed = derive_episode_seed(
                    split_name,
                    suite_task_id,
                    family.family_id,
                    variant,
                    "episode",
                    private_seed,
                )
                episode_id = f"{episode_counter:04d}"
                row, answer = build_episode(
                    split_name,
                    episode_id,
                    suite_task_id,
                    family,
                    variant,
                    base_seed,
                )
                rows.append(row)
                answers.append(answer)
                episode_counter += 1

    validate_split_quality(answers, split_name)
    return rows, answers


def sanitize_private_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    sanitized_rows: list[dict[str, object]] = []
    for row in rows:
        sanitized_rows.append(
            {
                "episode_id": row["episode_id"],
                "inference": {"turns": list(row["inference"]["turns"])},
                "analysis": {
                    "faculty_id": row["analysis"]["faculty_id"],
                    "suite_task_id": row["analysis"]["suite_task_id"],
                    "shift_mode": row["analysis"]["shift_mode"],
                    "difficulty_bin": row["analysis"]["difficulty_bin"],
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
        "version": "cogflex_suite_v1",
        "split": "private",
        "episodes": sanitized_answers,
    }


def build_private_artifacts(
    manifest_path: Path = PRIVATE_MANIFEST_PATH,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    manifest = load_private_manifest(manifest_path)
    private_rows, private_answers = build_split(
        "private",
        variants_per_family=10,
        private_seed=manifest["private_seed"],
    )
    return private_rows, private_answers, manifest


def dataset_metadata(dataset_id: str, title: str) -> dict[str, object]:
    return {
        "id": dataset_id,
        "title": title,
        "licenses": [{"name": "CC0-1.0"}],
    }


def validate_rows(
    rows: list[dict[str, object]],
    expected_count: int,
    per_task: int,
    require_scoring: bool,
) -> None:
    assert len(rows) == expected_count, (len(rows), expected_count)
    counts = Counter(str(row["analysis"]["suite_task_id"]) for row in rows)
    assert counts == Counter({suite_task_id: per_task for suite_task_id in SUITE_TASKS}), counts
    difficulty_counts = Counter(str(row["analysis"]["difficulty_bin"]) for row in rows)
    assert set(difficulty_counts) == {"hard", "medium"}, difficulty_counts
    for row in rows:
        turns = row["inference"]["turns"]
        assert isinstance(turns, list) and len(turns) == 3, row["episode_id"]
        for turn_index, turn in enumerate(turns, start=1):
            assert turn.startswith(f"CogFlex suite task. Episode {row['episode_id']}. Turn {turn_index} of 3.")
        if require_scoring:
            targets = row["scoring"]["final_probe_targets"]
            assert len(targets) == FINAL_PROBE_COUNT, row["episode_id"]
        else:
            assert "scoring" not in row, row["episode_id"]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    public_rows, public_answers = build_split("public", variants_per_family=2)
    private_rows, private_answers, _manifest = build_private_artifacts()

    for row in public_rows:
        row.pop("split", None)
    private_publish_rows = sanitize_private_rows(private_rows)

    validate_rows(public_rows, expected_count=80, per_task=20, require_scoring=True)
    validate_rows(private_publish_rows, expected_count=400, per_task=100, require_scoring=False)
    validate_split_isolation(public_answers, private_answers)

    write_json(PUBLIC_ROWS_PATH, public_rows)
    write_json(PUBLIC_METADATA_PATH, dataset_metadata(PUBLIC_DATASET_ID, "CogFlex Suite Runtime"))
    write_json(PRIVATE_ROWS_PATH, private_publish_rows)
    write_json(PRIVATE_METADATA_PATH, dataset_metadata(PRIVATE_DATASET_ID, "CogFlex Suite Runtime Private"))
    write_json(PRIVATE_ANSWER_KEY_PATH, private_answer_key_payload(private_answers))

    print(f"Wrote {len(public_rows)} public episodes to {PUBLIC_ROWS_PATH}")
    print(f"Wrote public dataset metadata to {PUBLIC_METADATA_PATH}")
    print(f"Wrote {len(private_publish_rows)} private inference-only episodes to {PRIVATE_ROWS_PATH}")
    print(f"Wrote private dataset metadata to {PRIVATE_METADATA_PATH}")
    print(f"Wrote private answer key to {PRIVATE_ANSWER_KEY_PATH}")


if __name__ == "__main__":
    main()
