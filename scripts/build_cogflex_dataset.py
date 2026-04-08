#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import re
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Final

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROWS_PATH = ROOT / "kaggle/dataset/public/public_leaderboard_rows.json"
PUBLIC_METADATA_PATH = ROOT / "kaggle/dataset/public/dataset-metadata.json"
PUBLIC_QUALITY_REPORT_PATH = ROOT / "kaggle/dataset/public/public_quality_report.json"

PUBLIC_DATASET_ID = "raptorengineer/cogflex-suite-runtime"
PRIVATE_DATASET_ID = "raptorengineer/cogflex-suite-runtime-private"
NOTEBOOK_ID = "raptorengineer/cogflex-suite-notebook"
TASK_NAME = "cogflex_suite_binary"
FACULTY_ID = "executive_functions/cognitive_flexibility"

PUBLIC_ROWS_FILENAME = "public_leaderboard_rows.json"
PRIVATE_ROWS_FILENAME = "private_leaderboard_rows.json"
PRIVATE_ANSWER_KEY_FILENAME = "private_answer_key.json"
PRIVATE_RELEASE_MANIFEST_FILENAME = "private_release_manifest.json"
PRIVATE_QUALITY_REPORT_FILENAME = "private_quality_report.json"
PRIVATE_BUNDLE_ENV_VAR = "COGFLEX_PRIVATE_BUNDLE_DIR"

PUBLIC_BUNDLE_VERSION = "cogflex_public_v1"
PRIVATE_BUNDLE_VERSION = "cogflex_private_bundle_v1"
PRIVATE_QUALITY_REPORT_VERSION = "cogflex_private_quality_v1"

LEARN_EXAMPLE_COUNT = 6
SHIFT_EXAMPLE_COUNT = 6
FINAL_PROBE_COUNT = 8
TURN_COUNT = 3
TRANSITION_FAMILY_COUNT = 15
PUBLIC_VARIANTS_PER_FAMILY = 8
PUBLIC_EPISODES_PER_TASK = 30
PRIVATE_EPISODES_PER_TASK = 120
MAX_SELECTED_TRANSITIONS_PER_FAMILY_ROLE = 3
MAX_SELECTED_TRANSITIONS_PER_FAMILY_PAIR = 1

VALUES = (-4, -3, -2, -1, 1, 2, 3, 4)
SHAPES = ("circle", "triangle", "square")
TONES = ("warm", "cool")
Point = tuple[int, int, str, str]
DOMAIN: list[Point] = [
    (r1, r2, shape, tone)
    for r1 in VALUES
    for r2 in VALUES
    for shape in SHAPES
    for tone in TONES
]

TYPE_TRUE = "type_a"
TYPE_FALSE = "type_b"

SUITE_TASKS: Final[tuple[str, ...]] = (
    "explicit_rule_update",
    "latent_rule_update",
    "context_binding",
    "trial_cued_switch",
)

SHIFT_MODES: Final[dict[str, str]] = {
    "explicit_rule_update": "explicit_instruction",
    "latent_rule_update": "latent_example_change",
    "context_binding": "context_gate",
    "trial_cued_switch": "cue_switching",
}

DISAGREEMENT_BINS: Final[tuple[str, ...]] = ("low", "mid", "high")

FINAL_OUTPUT_INSTRUCTION = (
    "Return exactly 8 outputs in order, one per probe. "
    "Use only type_a or type_b."
)

TURN_HEADER_PREFIX = "CogFlex suite task. Episode "
LINE_RE = re.compile(
    r"^(?P<index>\d+)\.\s+(?P<body>.+?)\s+->\s+(?P<label>type_a|type_b|\?)$"
)
POINT_RE = re.compile(r"^r1=(?P<r1>[+-]\d+),\s*r2=(?P<r2>[+-]\d+)$")
RETRIEVAL_HEADER_RE = re.compile(r"Episode \d+")

PUBLIC_CONTEXT_LEXICONS: Final[tuple[tuple[str, str], ...]] = (
    ("alpha", "beta"),
    ("oak", "reef"),
    ("north", "south"),
    ("ember", "glacier"),
)

PUBLIC_CUE_LEXICONS: Final[tuple[tuple[str, str], ...]] = (
    ("sun", "moon"),
    ("flint", "pearl"),
    ("drift", "pulse"),
    ("keystone", "signal"),
)

LEARN_TEMPLATES_GENERIC: Final[tuple[str, ...]] = (
    "Learn the active classification rule from these labeled examples.",
    "Infer the live rule from these labeled examples.",
    "Use these labeled examples to identify the current rule.",
    "Study these labeled examples and recover the active rule.",
    "Determine the live classifier from these labeled examples.",
    "Infer the present decision rule using these labeled examples.",
    "Read these labeled examples and identify the active mapping.",
    "Recover the active labeling rule from these examples.",
    "Use these labeled cases to determine the current classifier.",
    "Identify the rule that is active in these labeled examples.",
    "Extract the active rule from these supervised examples.",
    "Use the labeled examples below to infer the current rule.",
)

LEARN_TEMPLATES_CONTEXT: Final[tuple[str, ...]] = (
    "Learn the active rule for context={primary} from these labeled examples.",
    "Infer the current rule used when context={primary}.",
    "Use these labeled examples to determine the rule for context={primary}.",
    "Study these examples and recover the classifier tied to context={primary}.",
    "Identify the active rule whenever context={primary}.",
    "Read these labeled items to infer the rule for context={primary}.",
    "Determine the live mapping associated with context={primary}.",
    "Recover the rule that applies under context={primary}.",
    "Use the labeled examples below to infer the rule for context={primary}.",
    "Infer the classifier currently bound to context={primary}.",
    "Identify the decision rule active for context={primary}.",
    "Study the labeled cases and determine the rule for context={primary}.",
)

SHIFT_TEMPLATES_EXPLICIT: Final[tuple[str, ...]] = (
    "The active rule has changed. Infer the replacement rule from these labeled examples.",
    "A rule change occurred. Update to the new rule using these labeled examples.",
    "The classifier is different now. Recover the new rule from these examples.",
    "The live rule has been replaced. Infer the replacement from these labels.",
    "A shift happened. Use these labeled examples to learn the new rule.",
    "Update your rule: the active classifier changed, and these labels show the new behavior.",
    "The task now uses a different rule. Infer it from these labeled examples.",
    "The active mapping switched. Recover the new rule from the evidence below.",
    "A new rule is now active. Use these labeled examples to identify it.",
    "The decision rule changed. Infer the current rule from these labels.",
    "The task switched rules. Use the labeled evidence below to recover the new one.",
    "A rule transition occurred. Determine the replacement rule from these examples.",
)

SHIFT_TEMPLATES_LATENT: Final[tuple[str, ...]] = (
    "Continue the task using these labeled examples.",
    "Proceed with the task and stay calibrated using these labeled examples.",
    "Continue from the latest labeled evidence below.",
    "Use these labeled examples to stay aligned with the task.",
    "Continue the sequence using the labeled evidence that follows.",
    "Proceed using these labeled cases as the latest task evidence.",
    "Continue from the most recent labeled examples.",
    "Use the labeled examples below and continue the task.",
    "Continue after reviewing these newly labeled examples.",
    "Keep going using the labeled evidence below.",
    "Continue the task; the latest labeled examples are below.",
    "Use the latest labeled examples below and continue.",
)

SHIFT_TEMPLATES_CONTEXT: Final[tuple[str, ...]] = (
    "Now learn the active rule for context={secondary} from these labeled examples.",
    "Switch attention to context={secondary} and infer its rule from these labels.",
    "These labeled examples define the active classifier for context={secondary}.",
    "Use these labeled examples to recover the rule for context={secondary}.",
    "Now determine the rule associated with context={secondary}.",
    "Study these labels to infer the classifier bound to context={secondary}.",
    "Use the evidence below to identify the rule for context={secondary}.",
    "Infer the rule that is active when context={secondary}.",
    "These examples reveal the rule used for context={secondary}.",
    "Recover the active rule tied to context={secondary} from these labels.",
    "Determine the classifier that applies under context={secondary}.",
    "Study the labeled items below and identify the rule for context={secondary}.",
)

SHIFT_TEMPLATES_CUED: Final[tuple[str, ...]] = (
    "Each item now includes a cue. cue={keep} keeps the original rule and cue={switch} uses the alternate rule. Infer the alternate rule from these labeled examples.",
    "A cue now selects which rule applies. cue={keep} means keep the initial rule, while cue={switch} means use the alternate rule. Infer that alternate rule from these labels.",
    "Items now carry a cue: cue={keep} keeps the original classifier and cue={switch} selects the alternate one. Use these labeled examples to infer the alternate rule.",
    "A cue gate is now active. When cue={keep}, stay with the original rule; when cue={switch}, apply the alternate rule. Infer the alternate rule from these labels.",
    "The task now depends on a cue. cue={keep} preserves the first rule and cue={switch} activates a second rule. Recover that second rule from these examples.",
    "Each new item is cued. cue={keep} means original rule, cue={switch} means alternate rule. Use the labels below to infer the alternate rule.",
    "Cue-controlled switching is now active: cue={keep} keeps the initial rule and cue={switch} selects an alternate one. Infer the alternate rule here.",
    "The cue determines which rule applies. cue={keep} keeps the previous rule; cue={switch} selects another rule. Use these labels to identify that alternate rule.",
    "A cue selector now routes each item. cue={keep} uses the original rule and cue={switch} uses an alternate rule. Infer the alternate rule below.",
    "Cue-based switching is now part of the task. cue={keep} maps to the original rule and cue={switch} maps to the alternate rule. Recover the alternate rule from these labels.",
    "Use the cue legend for these examples: cue={keep} keeps the original rule, cue={switch} uses the alternate rule. Infer the alternate rule from the evidence below.",
    "Items now include a routing cue. cue={keep} keeps rule one and cue={switch} activates rule two. Infer rule two from the labeled examples.",
)

DECISION_TEMPLATES_GENERIC: Final[tuple[str, ...]] = (
    "Classify each probe with the currently active rule.",
    "Apply the active rule to each probe.",
    "Label each probe using the rule that is active now.",
    "Use the current rule to classify every probe.",
    "Assign a label to each probe using the live classifier.",
    "Apply the current decision rule to the probes below.",
    "Use the active mapping to label each probe.",
    "Classify the probes below with the live rule.",
)

DECISION_TEMPLATES_CONTEXT: Final[tuple[str, ...]] = (
    "Classify each probe with the rule associated with that probe's context.",
    "For each probe, use the classifier bound to its context.",
    "Apply the context-specific rule to every probe.",
    "Use each probe's context to choose the correct rule before labeling it.",
    "Label each probe with the rule that matches its context.",
    "For every probe, select the rule tied to its context and classify it.",
    "Choose the correct context-bound rule for each probe, then label it.",
    "Use the probe context to route to the correct rule before classifying it.",
)

DECISION_TEMPLATES_CUED: Final[tuple[str, ...]] = (
    "Classify each probe with the rule selected by that probe's cue.",
    "Use each probe's cue to choose the correct rule before labeling it.",
    "Apply the cue-selected rule to every probe.",
    "For each probe, route by cue and then classify with the selected rule.",
    "Choose the rule indicated by each cue and label the probe.",
    "Use the cue on each probe to determine which rule applies.",
    "Apply the correct cue-selected classifier to each probe.",
    "Each cue picks a rule; use that rule to classify the probe.",
)

PUBLIC_FAMILY_IDS: Final[tuple[str, ...]] = (
    "axis_threshold",
    "sign",
    "parity",
    "relational",
    "magnitude",
    "numeric_binding",
    "cross_feature_binding",
    "feature_gate",
)


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
    initial_family_id: str
    shift_family_id: str
    disagreement_count: int
    disagreement_bin: str


@dataclass(frozen=True)
class EpisodeCandidate:
    row: dict[str, object]
    answer: dict[str, object]
    diagnostics: dict[str, object]
    suite_task_id: str
    family_id: str
    variant: int
    attack_signal: float


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
            "axis_r1_le_+1",
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
            "axis_r2_le_+1",
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
            "sign_same_sign",
            "sign",
            "sign::same_sign",
            "type_a iff r1 and r2 share the same sign",
            lambda point: (point[0] > 0) == (point[1] > 0),
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
            "parity_same_parity",
            "parity",
            "parity::same_parity",
            "type_a iff r1 and r2 have the same parity",
            lambda point: (point[0] % 2) == (point[1] % 2),
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
            "magnitude_abs_r1_ge_3",
            "magnitude",
            "magnitude::abs_r1_ge_3",
            "type_a iff |r1| is at least 3",
            lambda point: abs(point[0]) >= 3,
        ),
        make_rule(
            "magnitude_abs_r2_ge_3",
            "magnitude",
            "magnitude::abs_r2_ge_3",
            "type_a iff |r2| is at least 3",
            lambda point: abs(point[1]) >= 3,
        ),
        make_rule(
            "magnitude_abs_r1_gt_abs_r2",
            "magnitude",
            "magnitude::abs_r1_gt_abs_r2",
            "type_a iff |r1| is greater than |r2|",
            lambda point: abs(point[0]) > abs(point[1]),
        ),
        make_rule(
            "magnitude_abs_sum_ge_5",
            "magnitude",
            "magnitude::abs_sum_ge_5",
            "type_a iff |r1| + |r2| is at least 5",
            lambda point: abs(point[0]) + abs(point[1]) >= 5,
        ),
        make_rule(
            "numeric_binding_r1_positive_matches_r2_even",
            "numeric_binding",
            "numeric_binding::r1_positive_matches_r2_even",
            "type_a iff r1>0 matches whether r2 is even",
            lambda point: (point[0] > 0) == (point[1] % 2 == 0),
        ),
        make_rule(
            "numeric_binding_r2_positive_matches_r1_even",
            "numeric_binding",
            "numeric_binding::r2_positive_matches_r1_even",
            "type_a iff r2>0 matches whether r1 is even",
            lambda point: (point[1] > 0) == (point[0] % 2 == 0),
        ),
        make_rule(
            "numeric_binding_abs_r1_ge_3_xor_r2_positive",
            "numeric_binding",
            "numeric_binding::abs_r1_ge_3_xor_r2_positive",
            "type_a iff exactly one of |r1|>=3 and r2>0 is true",
            lambda point: (abs(point[0]) >= 3) ^ (point[1] > 0),
        ),
        make_rule(
            "numeric_binding_abs_r2_ge_3_xor_r1_positive",
            "numeric_binding",
            "numeric_binding::abs_r2_ge_3_xor_r1_positive",
            "type_a iff exactly one of |r2|>=3 and r1>0 is true",
            lambda point: (abs(point[1]) >= 3) ^ (point[0] > 0),
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
            "cross_shape_circle_xor_tone_warm",
            "cross_feature_binding",
            "cross_feature_binding::shape_circle_xor_tone_warm",
            "type_a iff exactly one of shape=circle and tone=warm is true",
            lambda point: (point[2] == "circle") ^ (point[3] == "warm"),
        ),
        make_rule(
            "cross_shape_square_or_sum_ge_2",
            "cross_feature_binding",
            "cross_feature_binding::shape_square_or_sum_ge_2",
            "type_a iff shape is square or r1 + r2 is at least +2",
            lambda point: point[2] == "square" or point[0] + point[1] >= 2,
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
        make_rule(
            "gate_shape_triangle_then_r1_even_else_r2_positive",
            "feature_gate",
            "feature_gate::shape_triangle_then_r1_even_else_r2_positive",
            "type_a iff shape=triangle uses even(r1) and otherwise uses r2>0",
            lambda point: (point[0] % 2 == 0) if point[2] == "triangle" else (point[1] > 0),
        ),
    ]


def _rule_signature(rule: RuleSpec) -> tuple[bool, ...]:
    return tuple(rule.label(point) for point in DOMAIN)


def _dedupe_rules(rules: Iterable[RuleSpec]) -> tuple[RuleSpec, ...]:
    seen: set[tuple[bool, ...]] = set()
    unique: list[RuleSpec] = []
    for rule in rules:
        signature = _rule_signature(rule)
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(rule)
    return tuple(unique)


PUBLIC_RULES = _dedupe_rules(_public_rules())
RULE_BY_ID = {rule.rule_id: rule for rule in PUBLIC_RULES}


def disagreement_count(initial_rule_id: str, shift_rule_id: str) -> int:
    initial_rule = RULE_BY_ID[initial_rule_id]
    shift_rule = RULE_BY_ID[shift_rule_id]
    return sum(initial_rule.label(point) != shift_rule.label(point) for point in DOMAIN)


def derive_seed(*parts: object) -> int:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def compute_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def has_label_balance(labels: Iterable[str], *, min_true: int, min_false: int) -> bool:
    counts = Counter(labels)
    return counts[TYPE_TRUE] >= min_true and counts[TYPE_FALSE] >= min_false


def has_exact_label_balance(labels: Iterable[str], *, true_count: int, false_count: int) -> bool:
    counts = Counter(labels)
    return counts[TYPE_TRUE] == true_count and counts[TYPE_FALSE] == false_count


def point_from_item(item: dict[str, object]) -> Point:
    return (int(item["r1"]), int(item["r2"]), str(item["shape"]), str(item["tone"]))


def case_distance(source: dict[str, object], target: dict[str, object]) -> int:
    distance = abs(int(source["r1"]) - int(target["r1"])) + abs(int(source["r2"]) - int(target["r2"]))
    distance += 2 * (str(source["shape"]) != str(target["shape"]))
    distance += 2 * (str(source["tone"]) != str(target["tone"]))
    distance += 4 * (str(source.get("context", "")) != str(target.get("context", "")))
    distance += 4 * (str(source.get("cue", "")) != str(target.get("cue", "")))
    return distance


def pick_points(
    rng: random.Random,
    candidates: list[Point],
    count: int,
    predicate: Callable[[list[Point]], bool],
    *,
    attempts: int = 6000,
) -> list[Point]:
    if len(candidates) < count:
        raise RuntimeError(f"need {count} candidates, found {len(candidates)}")
    for _ in range(attempts):
        sample = rng.sample(candidates, count)
        if predicate(sample):
            rng.shuffle(sample)
            return sample
    raise RuntimeError("unable to sample points that satisfy benchmark constraints")


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


def build_examples(
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
        lambda sample: has_exact_label_balance((rule_label(rule, point) for point in sample), true_count=3, false_count=3)
        if count == 6
        else has_label_balance(
            (rule_label(rule, point) for point in sample),
            min_true=1,
            min_false=1,
        ),
    )


def _consistent_rules(examples: list[dict[str, object]]) -> list[RuleSpec]:
    matches: list[RuleSpec] = []
    for rule in PUBLIC_RULES:
        if all(rule_label(rule, point_from_item(item)) == str(item["label"]) for item in examples):
            matches.append(rule)
    return matches


def infer_context_terms(
    learn_examples: list[dict[str, object]],
    shift_examples: list[dict[str, object]],
    probes: list[dict[str, object]],
) -> tuple[str, str]:
    values: list[str] = []
    for item in learn_examples + shift_examples + probes:
        context = item.get("context")
        if context is None:
            continue
        normalized = str(context)
        if normalized not in values:
            values.append(normalized)
    if not values:
        return ("", "")
    if len(values) == 1:
        return (values[0], values[0])
    primary = str(learn_examples[0].get("context", values[0]))
    secondary = next((value for value in values if value != primary), values[1])
    return primary, secondary


def infer_cue_terms(
    learn_examples: list[dict[str, object]],
    shift_examples: list[dict[str, object]],
    probes: list[dict[str, object]],
) -> tuple[str, str]:
    values: list[str] = []
    for item in shift_examples + probes:
        cue = item.get("cue")
        if cue is None:
            continue
        normalized = str(cue)
        if normalized not in values:
            values.append(normalized)
    if not values:
        return ("", "")
    if len(values) == 1:
        return (values[0], values[0])
    initial_candidates = _consistent_rules(learn_examples)
    cue_to_examples: dict[str, list[dict[str, object]]] = {}
    for item in shift_examples:
        cue_to_examples.setdefault(str(item["cue"]), []).append(item)
    for cue in values:
        examples = cue_to_examples.get(cue, [])
        if examples and any(
            all(rule_label(rule, point_from_item(item)) == str(item["label"]) for item in examples)
            for rule in initial_candidates
        ):
            switch = next((value for value in values if value != cue), values[1])
            return cue, switch
    return values[0], values[1]


def _majority_vote(predictions: list[tuple[str, ...]]) -> tuple[str, ...]:
    labels: list[str] = []
    for index in range(FINAL_PROBE_COUNT):
        counts = Counter(prediction[index] for prediction in predictions)
        labels.append(TYPE_TRUE if counts[TYPE_TRUE] >= counts[TYPE_FALSE] else TYPE_FALSE)
    return tuple(labels)


def score_labels(predictions: Iterable[str], targets: Iterable[str]) -> float:
    pred_tuple = tuple(predictions)
    target_tuple = tuple(targets)
    return sum(pred == target for pred, target in zip(pred_tuple, target_tuple)) / len(target_tuple)


def normalized_turn_text(turn: str) -> str:
    return RETRIEVAL_HEADER_RE.sub("Episode XXXX", turn)


def token_signature(turns: list[str]) -> set[str]:
    text = " ".join(normalized_turn_text(turn) for turn in turns)
    return {token for token in re.split(r"[^a-z0-9_+\-]+", text.lower()) if token}


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def render_case_text(
    point: Point,
    *,
    context: str | None = None,
    cue: str | None = None,
    attribute_variant: int,
) -> str:
    chunks = [
        f"shape={point[2]}",
        f"tone={point[3]}",
        f"r1={fmt_signed(point[0])}, r2={fmt_signed(point[1])}",
    ]
    if context is not None:
        chunks.append(f"context={context}")
    if cue is not None:
        chunks.append(f"cue={cue}")

    ordered = list(chunks)
    rotation = attribute_variant % len(ordered)
    ordered = ordered[rotation:] + ordered[:rotation]
    if len(ordered) > 3 and attribute_variant % 2 == 1:
        ordered[0], ordered[1] = ordered[1], ordered[0]
    return " | ".join(ordered)


def render_labeled_lines(items: list[dict[str, object]], *, attribute_variant: int) -> str:
    return "\n".join(
        f"{item['index']}. "
        f"{render_case_text(point_from_item(item), context=item.get('context'), cue=item.get('cue'), attribute_variant=attribute_variant + int(item['index']))} "
        f"-> {item['label']}"
        for item in items
    )


def render_probe_lines(items: list[dict[str, object]], *, attribute_variant: int) -> str:
    return "\n".join(
        f"{item['index']}. "
        f"{render_case_text(point_from_item(item), context=item.get('context'), cue=item.get('cue'), attribute_variant=attribute_variant + int(item['index']))} "
        "-> ?"
        for item in items
    )


def render_turn_header(episode_id: str, turn_index: int) -> str:
    return f"{TURN_HEADER_PREFIX}{episode_id}. Turn {turn_index} of {TURN_COUNT}."


def select_template(templates: tuple[str, ...], variant: int, **fmt: str) -> str:
    return templates[variant % len(templates)].format(**fmt)


def render_learn_turn(
    episode_id: str,
    suite_task_id: str,
    examples: list[dict[str, object]],
    *,
    template_variant: int,
    attribute_variant: int,
    context_terms: tuple[str, str],
) -> str:
    if suite_task_id == "context_binding":
        guidance = select_template(
            LEARN_TEMPLATES_CONTEXT,
            template_variant,
            primary=context_terms[0],
        )
    else:
        guidance = select_template(LEARN_TEMPLATES_GENERIC, template_variant)
    return "\n\n".join(
        [
            render_turn_header(episode_id, 1),
            guidance,
            "Examples:\n" + render_labeled_lines(examples, attribute_variant=attribute_variant),
        ]
    )


def render_shift_turn(
    episode_id: str,
    suite_task_id: str,
    examples: list[dict[str, object]],
    *,
    template_variant: int,
    attribute_variant: int,
    context_terms: tuple[str, str],
    cue_terms: tuple[str, str],
) -> str:
    if suite_task_id == "explicit_rule_update":
        guidance = select_template(SHIFT_TEMPLATES_EXPLICIT, template_variant)
    elif suite_task_id == "latent_rule_update":
        guidance = select_template(SHIFT_TEMPLATES_LATENT, template_variant)
    elif suite_task_id == "context_binding":
        guidance = select_template(
            SHIFT_TEMPLATES_CONTEXT,
            template_variant,
            secondary=context_terms[1],
        )
    elif suite_task_id == "trial_cued_switch":
        guidance = select_template(
            SHIFT_TEMPLATES_CUED,
            template_variant,
            keep=cue_terms[0],
            switch=cue_terms[1],
        )
    else:
        raise ValueError(f"unsupported suite task {suite_task_id}")
    return "\n\n".join(
        [
            render_turn_header(episode_id, 2),
            guidance,
            "Examples:\n" + render_labeled_lines(examples, attribute_variant=attribute_variant),
        ]
    )


def render_decision_turn(
    episode_id: str,
    suite_task_id: str,
    probes: list[dict[str, object]],
    *,
    template_variant: int,
    attribute_variant: int,
) -> str:
    if suite_task_id == "context_binding":
        guidance = select_template(DECISION_TEMPLATES_CONTEXT, template_variant)
    elif suite_task_id == "trial_cued_switch":
        guidance = select_template(DECISION_TEMPLATES_CUED, template_variant)
    else:
        guidance = select_template(DECISION_TEMPLATES_GENERIC, template_variant)
    return "\n\n".join(
        [
            render_turn_header(episode_id, 3),
            guidance,
            "Probes:\n" + render_probe_lines(probes, attribute_variant=attribute_variant),
            FINAL_OUTPUT_INSTRUCTION,
        ]
    )


def parse_case_line(line: str) -> dict[str, object] | None:
    match = LINE_RE.match(line.strip())
    if match is None:
        return None
    body = match.group("body")
    item: dict[str, object] = {
        "index": int(match.group("index")),
        "label": match.group("label"),
    }
    point: tuple[int, int] | None = None
    for chunk in (part.strip() for part in body.split("|")):
        point_match = POINT_RE.match(chunk)
        if point_match is not None:
            point = (int(point_match.group("r1")), int(point_match.group("r2")))
            continue
        if "=" not in chunk:
            raise ValueError(f"malformed chunk: {chunk!r}")
        key, value = chunk.split("=", 1)
        item[key.strip()] = value.strip()
    if point is None:
        raise ValueError(f"missing coordinates in line: {line!r}")
    item["r1"] = point[0]
    item["r2"] = point[1]
    return item


def learn_only_max_probe_accuracy(
    learn_examples: list[dict[str, object]],
    probes: list[dict[str, object]],
    targets: tuple[str, ...],
) -> float:
    return max(
        score_labels((rule_label(rule, point_from_item(item)) for item in probes), targets)
        for rule in _consistent_rules(learn_examples)
    )


def _parse_examples(turn: str) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for line in turn.splitlines():
        parsed = parse_case_line(line)
        if parsed is None or parsed["label"] == "?":
            continue
        items.append(parsed)
    return items


def _parse_probes(turn: str) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for line in turn.splitlines():
        parsed = parse_case_line(line)
        if parsed is None or parsed["label"] != "?":
            continue
        items.append(parsed)
    return items


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


def dsl_hypothesis_predictions(
    suite_task_id: str,
    learn_examples: list[dict[str, object]],
    shift_examples: list[dict[str, object]],
    probes: list[dict[str, object]],
    *,
    context_terms: tuple[str, str],
    cue_terms: tuple[str, str],
) -> list[tuple[str, ...]]:
    hypotheses: list[tuple[str, ...]] = []
    if suite_task_id in {"explicit_rule_update", "latent_rule_update"}:
        initial_candidates = _consistent_rules(learn_examples)
        shift_candidates = _consistent_rules(shift_examples)
        hypotheses = [
            tuple(rule_label(shift_rule, point_from_item(item)) for item in probes)
            for initial_rule in initial_candidates
            for shift_rule in shift_candidates
            if initial_rule.rule_id != shift_rule.rule_id
        ]
    elif suite_task_id == "context_binding":
        alpha_candidates = _consistent_rules(learn_examples)
        beta_candidates = _consistent_rules(shift_examples)
        for alpha_rule in alpha_candidates:
            for beta_rule in beta_candidates:
                predictions: list[str] = []
                for item in probes:
                    active_rule = alpha_rule if item.get("context") == context_terms[0] else beta_rule
                    predictions.append(rule_label(active_rule, point_from_item(item)))
                hypotheses.append(tuple(predictions))
    elif suite_task_id == "trial_cued_switch":
        initial_candidates = _consistent_rules(learn_examples)
        keep_cue, switch_cue = cue_terms
        for initial_rule in initial_candidates:
            for shift_rule in PUBLIC_RULES:
                if not all(
                    rule_label(initial_rule if item.get("cue") == keep_cue else shift_rule, point_from_item(item))
                    == str(item["label"])
                    for item in shift_examples
                ):
                    continue
                predictions = []
                for item in probes:
                    active_rule = initial_rule if item.get("cue") == keep_cue else shift_rule
                    predictions.append(rule_label(active_rule, point_from_item(item)))
                hypotheses.append(tuple(predictions))
    else:
        raise ValueError(f"unsupported suite task {suite_task_id}")

    if not hypotheses:
        raise RuntimeError(f"no DSL hypotheses for suite task {suite_task_id}")
    return hypotheses


def dsl_search_predictions(
    suite_task_id: str,
    learn_examples: list[dict[str, object]],
    shift_examples: list[dict[str, object]],
    probes: list[dict[str, object]],
    *,
    context_terms: tuple[str, str],
    cue_terms: tuple[str, str],
) -> tuple[tuple[str, ...], int]:
    hypotheses = dsl_hypothesis_predictions(
        suite_task_id,
        learn_examples,
        shift_examples,
        probes,
        context_terms=context_terms,
        cue_terms=cue_terms,
    )
    return _majority_vote(hypotheses), len(hypotheses)


def symbolic_diagnostics(
    suite_task_id: str,
    learn_examples: list[dict[str, object]],
    shift_examples: list[dict[str, object]],
    probes: list[dict[str, object]],
    *,
    initial_rule: RuleSpec,
    shift_rule: RuleSpec,
    context_terms: tuple[str, str],
    cue_terms: tuple[str, str],
) -> dict[str, object]:
    targets = tuple(str(item["label"]) for item in probes)
    previous_predictions = tuple(rule_label(initial_rule, point_from_item(item)) for item in probes)
    majority_counts = Counter(str(item["label"]) for item in learn_examples + shift_examples)
    majority_label = TYPE_TRUE if majority_counts[TYPE_TRUE] >= majority_counts[TYPE_FALSE] else TYPE_FALSE
    majority_predictions = (majority_label,) * FINAL_PROBE_COUNT
    nearest_predictions = nearest_neighbor_predictions(learn_examples + shift_examples, probes)
    learn_only_accuracy = learn_only_max_probe_accuracy(learn_examples, probes, targets)

    cue_agnostic_accuracy: float | None = None
    if suite_task_id == "context_binding":
        alpha_predictions = tuple(rule_label(initial_rule, point_from_item(item)) for item in probes)
        beta_predictions = tuple(rule_label(shift_rule, point_from_item(item)) for item in probes)
        cue_agnostic_accuracy = max(score_labels(alpha_predictions, targets), score_labels(beta_predictions, targets))
    elif suite_task_id == "trial_cued_switch":
        initial_predictions = tuple(rule_label(initial_rule, point_from_item(item)) for item in probes)
        shift_predictions = tuple(rule_label(shift_rule, point_from_item(item)) for item in probes)
        cue_agnostic_accuracy = max(score_labels(initial_predictions, targets), score_labels(shift_predictions, targets))

    dsl_hypotheses = dsl_hypothesis_predictions(
        suite_task_id,
        learn_examples,
        shift_examples,
        probes,
        context_terms=context_terms,
        cue_terms=cue_terms,
    )
    dsl_predictions = _majority_vote(dsl_hypotheses)
    dsl_hypothesis_count = len(dsl_hypotheses)
    turn2_required_probe_count = sum(
        previous_prediction != target for previous_prediction, target in zip(previous_predictions, targets)
    )

    return {
        "previous_rule_accuracy": score_labels(previous_predictions, targets),
        "learn_only_max_probe_accuracy": learn_only_accuracy,
        "majority_label_accuracy": score_labels(majority_predictions, targets),
        "nearest_neighbor_accuracy": score_labels(nearest_predictions, targets),
        "cue_agnostic_accuracy": cue_agnostic_accuracy,
        "dsl_search_accuracy": score_labels(dsl_predictions, targets),
        "dsl_hypothesis_count": dsl_hypothesis_count,
        "post_shift_prediction_set_size": len(set(dsl_hypotheses)),
        "turn2_required_probe_count": turn2_required_probe_count,
    }


def public_retrieval_scores(answers: list[dict[str, object]]) -> dict[str, float]:
    task_buckets: dict[str, list[dict[str, object]]] = defaultdict(list)
    for answer in answers:
        task_buckets[str(answer["suite_task_id"])].append(answer)

    scores: dict[str, float] = {}
    for suite_task_id, items in task_buckets.items():
        signatures = {str(item["episode_id"]): token_signature(list(item["turns"])) for item in items}
        for answer in items:
            episode_id = str(answer["episode_id"])
            current_signature = signatures[episode_id]
            current_targets = tuple(str(label) for label in answer["final_probe_targets"])
            best_score = -1.0
            best_predictions = current_targets
            for other in items:
                other_id = str(other["episode_id"])
                if other_id == episode_id:
                    continue
                similarity = jaccard_similarity(current_signature, signatures[other_id])
                if similarity > best_score:
                    best_score = similarity
                    memory_bank = list(other["learn_turn_examples"]) + list(other["shift_turn_examples"])
                    best_predictions = nearest_neighbor_predictions(memory_bank, list(answer["final_probes"]))
            scores[episode_id] = score_labels(best_predictions, current_targets)
    return scores


def attack_limits_for_task(suite_task_id: str) -> dict[str, float]:
    limits = {
        "learn_only_max_probe_accuracy": 0.625,
        "majority_label_accuracy": 0.625,
        "nearest_neighbor_accuracy": 0.625,
        "dsl_search_accuracy": 0.625,
        "public_retrieval_accuracy": 1.0,
    }
    if suite_task_id in {"explicit_rule_update", "latent_rule_update"}:
        limits["learn_only_max_probe_accuracy"] = 0.50
        limits["previous_rule_accuracy"] = 0.25
    if suite_task_id in {"context_binding", "trial_cued_switch"}:
        limits["cue_agnostic_accuracy"] = 0.50
    return limits


def validate_label_shortcut_constraints(answer: dict[str, object]) -> None:
    episode_id = str(answer["episode_id"])
    if not has_exact_label_balance(
        (str(item["label"]) for item in answer["learn_turn_examples"]),
        true_count=3,
        false_count=3,
    ):
        raise ValueError(f"episode {episode_id} learn examples must stay label-balanced")
    if not has_exact_label_balance(
        (str(item["label"]) for item in answer["shift_turn_examples"]),
        true_count=3,
        false_count=3,
    ):
        raise ValueError(f"episode {episode_id} shift examples must stay label-balanced")
    if not has_exact_label_balance(
        (str(label) for label in answer["final_probe_targets"]),
        true_count=4,
        false_count=4,
    ):
        raise ValueError(f"episode {episode_id} final probes must stay label-balanced")


def count_rule_conflicts(rule: RuleSpec, items: list[dict[str, object]]) -> int:
    return sum(rule_label(rule, point_from_item(item)) != str(item["label"]) for item in items)


def validate_episode_constraints(answer: dict[str, object]) -> None:
    suite_task_id = str(answer["suite_task_id"])
    diagnostics = answer["generator_diagnostics"]
    validate_label_shortcut_constraints(answer)
    limits = attack_limits_for_task(suite_task_id)
    for metric, ceiling in limits.items():
        value = diagnostics.get(metric)
        if value is None:
            continue
        if float(value) > ceiling:
            raise ValueError(f"episode {answer['episode_id']} violates {metric}: {value:.3f} > {ceiling:.3f}")

    shift_conflicts = int(diagnostics["shift_evidence_conflict_count"])
    if suite_task_id == "explicit_rule_update" and shift_conflicts < 4:
        raise ValueError(f"episode {answer['episode_id']} lacks strong explicit shift evidence: {shift_conflicts}")
    if suite_task_id == "latent_rule_update" and shift_conflicts != 2:
        raise ValueError(f"episode {answer['episode_id']} has invalid latent conflict count: {shift_conflicts}")
    prediction_set_ceiling = 4 if suite_task_id in {"explicit_rule_update", "latent_rule_update"} else 8
    if int(diagnostics["post_shift_prediction_set_size"]) > prediction_set_ceiling:
        raise ValueError(
            f"episode {answer['episode_id']} leaves turn-3 predictions too ambiguous after turn 2"
        )
    required_probe_count = 8 if suite_task_id in {"explicit_rule_update", "latent_rule_update"} else 4
    if int(diagnostics["turn2_required_probe_count"]) != required_probe_count:
        raise ValueError(
            f"episode {answer['episode_id']} has invalid turn2_required_probe_count: "
            f"{diagnostics['turn2_required_probe_count']}"
        )
    if suite_task_id in {"context_binding", "trial_cued_switch"} and int(diagnostics["diagnostic_probe_count"]) != 8:
        raise ValueError(
            f"episode {answer['episode_id']} lacks enough diagnostic probes: {diagnostics['diagnostic_probe_count']}"
        )


def candidate_attack_signal(diagnostics: dict[str, object], suite_task_id: str) -> float:
    finite_metrics = [
        float(diagnostics["learn_only_max_probe_accuracy"]),
        float(diagnostics["majority_label_accuracy"]),
        float(diagnostics["nearest_neighbor_accuracy"]),
        float(diagnostics["dsl_search_accuracy"]),
    ]
    if suite_task_id in {"explicit_rule_update", "latent_rule_update"}:
        finite_metrics.append(float(diagnostics["previous_rule_accuracy"]))
    if suite_task_id in {"context_binding", "trial_cued_switch"}:
        finite_metrics.append(float(diagnostics["cue_agnostic_accuracy"]))
    return max(finite_metrics)


def rule_pair_candidates() -> list[tuple[str, str, int]]:
    pairs: list[tuple[str, str, int]] = []
    for initial_rule in PUBLIC_RULES:
        for shift_rule in PUBLIC_RULES:
            if initial_rule.rule_id == shift_rule.rule_id:
                continue
            if initial_rule.family_id == shift_rule.family_id:
                continue
            disagreements = disagreement_count(initial_rule.rule_id, shift_rule.rule_id)
            if disagreements < 40:
                continue
            pairs.append((initial_rule.rule_id, shift_rule.rule_id, disagreements))
    return pairs


def disagreement_thresholds(pairs: list[tuple[str, str, int]]) -> tuple[int, int]:
    values = sorted(disagreements for _initial, _shift, disagreements in pairs)
    low_index = len(values) // 3
    high_index = (2 * len(values)) // 3
    return values[low_index], values[high_index]


def disagreement_bin(disagreements: int, *, low_cutoff: int, high_cutoff: int) -> str:
    if disagreements < low_cutoff:
        return "low"
    if disagreements < high_cutoff:
        return "mid"
    return "high"


def build_transition_families() -> tuple[TransitionFamily, ...]:
    pairs = rule_pair_candidates()
    low_cutoff, high_cutoff = disagreement_thresholds(pairs)
    families: list[TransitionFamily] = []
    for initial_rule_id, shift_rule_id, disagreements in pairs:
        initial_rule = RULE_BY_ID[initial_rule_id]
        shift_rule = RULE_BY_ID[shift_rule_id]
        families.append(
            TransitionFamily(
                family_id=f"public::pair::{initial_rule_id}__to__{shift_rule_id}",
                initial_rule_id=initial_rule_id,
                shift_rule_id=shift_rule_id,
                initial_family_id=initial_rule.family_id,
                shift_family_id=shift_rule.family_id,
                disagreement_count=disagreements,
                disagreement_bin=disagreement_bin(disagreements, low_cutoff=low_cutoff, high_cutoff=high_cutoff),
            )
        )
    return tuple(
        sorted(
            families,
            key=lambda item: (
                item.initial_family_id,
                item.shift_family_id,
                DISAGREEMENT_BINS.index(item.disagreement_bin),
                -item.disagreement_count,
                item.initial_rule_id,
                item.shift_rule_id,
            ),
        )
    )

ALL_TRANSITION_FAMILIES = build_transition_families()


def family_by_id() -> dict[str, TransitionFamily]:
    return {family.family_id: family for family in ALL_TRANSITION_FAMILIES}


TRANSITION_FAMILY_BY_ID = family_by_id()


def select_context_terms(variant_seed: int) -> tuple[str, str]:
    return PUBLIC_CONTEXT_LEXICONS[variant_seed % len(PUBLIC_CONTEXT_LEXICONS)]


def select_cue_terms(variant_seed: int) -> tuple[str, str]:
    return PUBLIC_CUE_LEXICONS[variant_seed % len(PUBLIC_CUE_LEXICONS)]


def build_transition_episode(
    family: TransitionFamily,
    rng: random.Random,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    initial_rule = RULE_BY_ID[family.initial_rule_id]
    shift_rule = RULE_BY_ID[family.shift_rule_id]
    for _ in range(6000):
        learn_points = pick_balanced_examples(rng, initial_rule, LEARN_EXAMPLE_COUNT)
        used = set(learn_points)
        disagreement_points = [
            point for point in DOMAIN if initial_rule.label(point) != shift_rule.label(point) and point not in used
        ]
        if len(disagreement_points) < SHIFT_EXAMPLE_COUNT + FINAL_PROBE_COUNT:
            continue
        shift_points = pick_points(
            rng,
            [point for point in DOMAIN if point not in used],
            SHIFT_EXAMPLE_COUNT,
            lambda sample: (
                sum(point in disagreement_points for point in sample) >= 4
                and has_exact_label_balance(
                    (rule_label(shift_rule, point) for point in sample),
                    true_count=3,
                    false_count=3,
                )
            ),
        )
        used.update(shift_points)
        remaining_disagreement = [point for point in disagreement_points if point not in used]
        if len(remaining_disagreement) < FINAL_PROBE_COUNT:
            continue
        probe_points = pick_points(
            rng,
            remaining_disagreement,
            FINAL_PROBE_COUNT,
            lambda sample: has_exact_label_balance(
                (rule_label(shift_rule, point) for point in sample),
                true_count=4,
                false_count=4,
            ),
        )
        learn_examples = build_examples(learn_points, initial_rule, start_index=1)
        shift_examples = build_examples(shift_points, shift_rule, start_index=1)
        probes = [
            serialize_case(index, point, rule_label(shift_rule, point), active_rule_id=shift_rule.rule_id)
            for index, point in enumerate(probe_points, start=1)
        ]
        return learn_examples, shift_examples, probes, {
            "shift_evidence_conflict_count": count_rule_conflicts(initial_rule, shift_examples),
            "diagnostic_probe_count": len(probes),
        }
    raise RuntimeError(f"unable to build explicit transition episode for {family.family_id}")


def build_latent_transition_episode(
    family: TransitionFamily,
    rng: random.Random,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    initial_rule = RULE_BY_ID[family.initial_rule_id]
    shift_rule = RULE_BY_ID[family.shift_rule_id]
    for _ in range(6000):
        learn_points = pick_balanced_examples(rng, initial_rule, LEARN_EXAMPLE_COUNT)
        used = set(learn_points)
        disagreement_points = [
            point for point in DOMAIN if initial_rule.label(point) != shift_rule.label(point) and point not in used
        ]
        agreement_points = [
            point for point in DOMAIN if initial_rule.label(point) == shift_rule.label(point) and point not in used
        ]
        if len(disagreement_points) < FINAL_PROBE_COUNT + 2 or len(agreement_points) < 4:
            continue
        conflict_points = pick_points(
            rng,
            disagreement_points,
            2,
            lambda sample: has_exact_label_balance(
                (rule_label(shift_rule, point) for point in sample),
                true_count=1,
                false_count=1,
            ),
        )
        used.update(conflict_points)
        agreement_points = [point for point in agreement_points if point not in used]
        agreement_sample = pick_points(
            rng,
            agreement_points,
            4,
            lambda sample: has_exact_label_balance(
                (rule_label(shift_rule, point) for point in conflict_points + sample),
                true_count=3,
                false_count=3,
            ),
        )
        used.update(agreement_sample)
        remaining_disagreement = [point for point in disagreement_points if point not in used]
        if len(remaining_disagreement) < FINAL_PROBE_COUNT:
            continue
        probe_points = pick_points(
            rng,
            remaining_disagreement,
            FINAL_PROBE_COUNT,
            lambda sample: has_exact_label_balance(
                (rule_label(shift_rule, point) for point in sample),
                true_count=4,
                false_count=4,
            ),
        )
        shift_points = list(conflict_points + agreement_sample)
        rng.shuffle(shift_points)
        learn_examples = build_examples(learn_points, initial_rule, start_index=1)
        shift_examples = build_examples(shift_points, shift_rule, start_index=1)
        probes = [
            serialize_case(index, point, rule_label(shift_rule, point), active_rule_id=shift_rule.rule_id)
            for index, point in enumerate(probe_points, start=1)
        ]
        return learn_examples, shift_examples, probes, {
            "shift_evidence_conflict_count": count_rule_conflicts(initial_rule, shift_examples),
            "diagnostic_probe_count": len(probes),
        }
    raise RuntimeError(f"unable to build latent transition episode for {family.family_id}")


def build_context_binding_episode(
    family: TransitionFamily,
    rng: random.Random,
    context_terms: tuple[str, str],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    initial_rule = RULE_BY_ID[family.initial_rule_id]
    shift_rule = RULE_BY_ID[family.shift_rule_id]
    for _ in range(6000):
        alpha_points = pick_balanced_examples(rng, initial_rule, LEARN_EXAMPLE_COUNT)
        used = set(alpha_points)
        beta_points = pick_balanced_examples(rng, shift_rule, SHIFT_EXAMPLE_COUNT, exclude=used)
        used.update(beta_points)
        disagreement_points = [
            point for point in DOMAIN if initial_rule.label(point) != shift_rule.label(point) and point not in used
        ]
        if len(disagreement_points) < FINAL_PROBE_COUNT:
            continue
        alpha_probe_points = pick_points(
            rng,
            disagreement_points,
            4,
            lambda sample: has_exact_label_balance(
                (rule_label(initial_rule, point) for point in sample),
                true_count=2,
                false_count=2,
            ),
        )
        used.update(alpha_probe_points)
        remaining = [point for point in disagreement_points if point not in used]
        if len(remaining) < 4:
            continue
        beta_probe_points = pick_points(
            rng,
            remaining,
            4,
            lambda sample: has_exact_label_balance(
                (rule_label(shift_rule, point) for point in sample),
                true_count=2,
                false_count=2,
            ),
        )
        probe_specs = (
            [(context_terms[0], point, initial_rule) for point in alpha_probe_points]
            + [(context_terms[1], point, shift_rule) for point in beta_probe_points]
        )
        rng.shuffle(probe_specs)
        probes = [
            serialize_case(index, point, rule_label(rule, point), context=context, active_rule_id=rule.rule_id)
            for index, (context, point, rule) in enumerate(probe_specs, start=1)
        ]
        learn_examples = build_examples(alpha_points, initial_rule, start_index=1, context=context_terms[0])
        shift_examples = build_examples(beta_points, shift_rule, start_index=1, context=context_terms[1])
        return learn_examples, shift_examples, probes, {
            "shift_evidence_conflict_count": count_rule_conflicts(initial_rule, shift_examples),
            "diagnostic_probe_count": sum(
                rule_label(initial_rule if item["context"] == context_terms[0] else shift_rule, point_from_item(item))
                != rule_label(shift_rule if item["context"] == context_terms[0] else initial_rule, point_from_item(item))
                for item in probes
            ),
        }
    raise RuntimeError(f"unable to build context-binding episode for {family.family_id}")


def build_trial_cued_switch_episode(
    family: TransitionFamily,
    rng: random.Random,
    cue_terms: tuple[str, str],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    initial_rule = RULE_BY_ID[family.initial_rule_id]
    shift_rule = RULE_BY_ID[family.shift_rule_id]
    keep_cue, switch_cue = cue_terms
    for _ in range(6000):
        learn_points = pick_balanced_examples(rng, initial_rule, LEARN_EXAMPLE_COUNT)
        used = set(learn_points)
        disagreement_points = [
            point for point in DOMAIN if initial_rule.label(point) != shift_rule.label(point) and point not in used
        ]
        if len(disagreement_points) < FINAL_PROBE_COUNT + 2:
            continue
        keep_points = pick_points(
            rng,
            [point for point in DOMAIN if point not in used],
            3,
            lambda sample: has_label_balance((rule_label(initial_rule, point) for point in sample), min_true=1, min_false=1),
        )
        used.update(keep_points)
        switch_points = pick_points(
            rng,
            [point for point in DOMAIN if point not in used],
            3,
            lambda sample: (
                sum(point in disagreement_points for point in sample) >= 2
                and has_exact_label_balance(
                    [rule_label(initial_rule, point) for point in keep_points]
                    + [rule_label(shift_rule, point) for point in sample],
                    true_count=3,
                    false_count=3,
                )
            ),
        )
        used.update(switch_points)
        remaining_disagreement = [point for point in disagreement_points if point not in used]
        if len(remaining_disagreement) < FINAL_PROBE_COUNT:
            continue
        keep_probe_points = pick_points(
            rng,
            remaining_disagreement,
            4,
            lambda sample: has_exact_label_balance(
                (rule_label(initial_rule, point) for point in sample),
                true_count=2,
                false_count=2,
            ),
        )
        used.update(keep_probe_points)
        remaining_disagreement = [point for point in remaining_disagreement if point not in used]
        if len(remaining_disagreement) < 4:
            continue
        switch_probe_points = pick_points(
            rng,
            remaining_disagreement,
            4,
            lambda sample: has_exact_label_balance(
                (rule_label(shift_rule, point) for point in sample),
                true_count=2,
                false_count=2,
            ),
        )
        probe_specs = (
            [(keep_cue, point, initial_rule) for point in keep_probe_points]
            + [(switch_cue, point, shift_rule) for point in switch_probe_points]
        )
        rng.shuffle(probe_specs)
        probes = [
            serialize_case(index, point, rule_label(rule, point), cue=cue, active_rule_id=rule.rule_id)
            for index, (cue, point, rule) in enumerate(probe_specs, start=1)
        ]
        shift_examples = build_examples(keep_points, initial_rule, start_index=1, cue=keep_cue) + build_examples(
            switch_points,
            shift_rule,
            start_index=4,
            cue=switch_cue,
        )
        rng.shuffle(shift_examples)
        for index, item in enumerate(shift_examples, start=1):
            item["index"] = index
        return build_examples(learn_points, initial_rule, start_index=1), shift_examples, probes, {
            "shift_evidence_conflict_count": count_rule_conflicts(initial_rule, shift_examples),
            "diagnostic_probe_count": len(probes),
        }
    raise RuntimeError(f"unable to build trial-cued switch episode for {family.family_id}")


def build_episode_candidate(
    episode_id: str,
    suite_task_id: str,
    family: TransitionFamily,
    variant: int,
) -> EpisodeCandidate:
    initial_rule = RULE_BY_ID[family.initial_rule_id]
    shift_rule = RULE_BY_ID[family.shift_rule_id]

    for attempt in range(400):
        seed = derive_seed("public", suite_task_id, family.family_id, variant, attempt)
        rng = random.Random(seed)
        context_terms = select_context_terms(seed)
        cue_terms = select_cue_terms(seed // 7)

        if suite_task_id == "explicit_rule_update":
            learn_examples, shift_examples, probes, extra = build_transition_episode(family, rng)
        elif suite_task_id == "latent_rule_update":
            learn_examples, shift_examples, probes, extra = build_latent_transition_episode(family, rng)
        elif suite_task_id == "context_binding":
            learn_examples, shift_examples, probes, extra = build_context_binding_episode(family, rng, context_terms)
        elif suite_task_id == "trial_cued_switch":
            learn_examples, shift_examples, probes, extra = build_trial_cued_switch_episode(family, rng, cue_terms)
        else:
            raise ValueError(f"unsupported suite task {suite_task_id}")

        surface_variant = seed % 12
        decision_variant = (seed // 11) % 8
        attribute_variant = (seed // 17) % 7
        turns = [
            render_learn_turn(
                episode_id,
                suite_task_id,
                learn_examples,
                template_variant=surface_variant,
                attribute_variant=attribute_variant,
                context_terms=context_terms,
            ),
            render_shift_turn(
                episode_id,
                suite_task_id,
                shift_examples,
                template_variant=surface_variant + 3,
                attribute_variant=attribute_variant + 1,
                context_terms=context_terms,
                cue_terms=cue_terms,
            ),
            render_decision_turn(
                episode_id,
                suite_task_id,
                probes,
                template_variant=decision_variant,
                attribute_variant=attribute_variant + 2,
            ),
        ]

        diagnostics = symbolic_diagnostics(
            suite_task_id,
            learn_examples,
            shift_examples,
            probes,
            initial_rule=initial_rule,
            shift_rule=shift_rule,
            context_terms=context_terms,
            cue_terms=cue_terms,
        )
        diagnostics.update(extra)
        diagnostics.update(
            {
                "public_retrieval_accuracy": None,
                "transition_disagreement_count": family.disagreement_count,
                "transition_disagreement_bin": family.disagreement_bin,
            }
        )

        analysis = {
            "faculty_id": FACULTY_ID,
            "suite_task_id": suite_task_id,
            "shift_mode": SHIFT_MODES[suite_task_id],
            "difficulty_bin": "pending",
        }
        answer = {
            "episode_id": episode_id,
            "suite_task_id": suite_task_id,
            "shift_mode": SHIFT_MODES[suite_task_id],
            "difficulty_bin": "pending",
            "transition_family_id": family.family_id,
            "initial_rule_id": family.initial_rule_id,
            "shift_rule_id": family.shift_rule_id,
            "initial_rule_family_id": family.initial_family_id,
            "shift_rule_family_id": family.shift_family_id,
            "initial_rule_template_id": initial_rule.template_id,
            "shift_rule_template_id": shift_rule.template_id,
            "cue_lexicon_id": f"public_cue_lexicon::{PUBLIC_CUE_LEXICONS.index(cue_terms)}",
            "context_lexicon_id": f"public_context_lexicon::{PUBLIC_CONTEXT_LEXICONS.index(context_terms)}",
            "surface_template_id": f"surface_template::{surface_variant % 12}",
            "learn_turn_examples": learn_examples,
            "shift_turn_examples": shift_examples,
            "final_probes": probes,
            "final_probe_targets": [str(item["label"]) for item in probes],
            "turns": turns,
            "generator_diagnostics": diagnostics,
        }
        row = {
            "episode_id": episode_id,
            "inference": {"turns": turns},
            "scoring": {"final_probe_targets": [str(item["label"]) for item in probes]},
            "analysis": analysis,
        }
        try:
            validate_episode_constraints(answer)
        except ValueError:
            continue
        return EpisodeCandidate(
            row=row,
            answer=answer,
            diagnostics=diagnostics,
            suite_task_id=suite_task_id,
            family_id=family.family_id,
            variant=variant,
            attack_signal=candidate_attack_signal(diagnostics, suite_task_id),
        )

    raise RuntimeError(f"unable to build valid episode candidate for {suite_task_id}/{family.family_id}/{variant}")


def candidate_sort_key(candidate: EpisodeCandidate) -> tuple[float, int, str]:
    return (
        candidate.attack_signal,
        -int(candidate.diagnostics["transition_disagreement_count"]),
        str(candidate.answer["surface_template_id"]),
    )


def pick_family_candidates(candidates: list[EpisodeCandidate]) -> list[EpisodeCandidate]:
    if len(candidates) < 2:
        raise RuntimeError("need at least two candidates to select a family pair")
    shortlist = candidates[: min(len(candidates), 6)]
    signatures = [token_signature(list(candidate.answer["turns"])) for candidate in shortlist]
    best_pair: tuple[int, int] | None = None
    best_key: tuple[float, float, int, int, int] | None = None
    for left_index in range(len(shortlist)):
        for right_index in range(left_index + 1, len(shortlist)):
            left = shortlist[left_index]
            right = shortlist[right_index]
            similarity = jaccard_similarity(signatures[left_index], signatures[right_index])
            same_surface = int(left.answer["surface_template_id"] == right.answer["surface_template_id"])
            same_lexicon = int(left.answer["cue_lexicon_id"] == right.answer["cue_lexicon_id"]) + int(
                left.answer["context_lexicon_id"] == right.answer["context_lexicon_id"]
            )
            key = (
                max(left.attack_signal, right.attack_signal),
                similarity,
                same_surface,
                same_lexicon,
                left_index + right_index,
            )
            if best_key is None or key < best_key:
                best_key = key
                best_pair = (left_index, right_index)
    if best_pair is None:
        raise RuntimeError("unable to select a diverse candidate pair")
    return [shortlist[best_pair[0]], shortlist[best_pair[1]]]


def ordered_family_buckets() -> dict[tuple[str, str, str], list[TransitionFamily]]:
    buckets: dict[tuple[str, str, str], list[TransitionFamily]] = defaultdict(list)
    for family in ALL_TRANSITION_FAMILIES:
        buckets[(family.initial_family_id, family.shift_family_id, family.disagreement_bin)].append(family)
    return buckets


def build_family_task_candidates(
    suite_task_id: str,
    family: TransitionFamily,
    *,
    provisional_counter_start: int,
) -> list[EpisodeCandidate]:
    candidates: list[EpisodeCandidate] = []
    provisional_counter = provisional_counter_start
    for variant in range(PUBLIC_VARIANTS_PER_FAMILY):
        episode_id = f"cand-{provisional_counter:05d}"
        try:
            candidate = build_episode_candidate(episode_id, suite_task_id, family, variant)
        except RuntimeError:
            provisional_counter += 1
            continue
        candidates.append(candidate)
        provisional_counter += 1
    candidates.sort(key=candidate_sort_key)
    return candidates


def build_candidate_pool() -> tuple[tuple[TransitionFamily, ...], dict[str, dict[str, list[EpisodeCandidate]]]]:
    buckets = ordered_family_buckets()
    ordered_keys = sorted(
        buckets,
        key=lambda key: (
            key[0],
            key[1],
            DISAGREEMENT_BINS.index(key[2]),
        ),
    )
    selected: list[TransitionFamily] = []
    selected_pool: dict[str, dict[str, list[EpisodeCandidate]]] = {suite_task_id: {} for suite_task_id in SUITE_TASKS}
    initial_usage: Counter[str] = Counter()
    shift_usage: Counter[str] = Counter()
    initial_family_usage: Counter[str] = Counter()
    shift_family_usage: Counter[str] = Counter()
    family_pair_usage: Counter[tuple[str, str]] = Counter()
    family_cache: dict[str, dict[str, list[EpisodeCandidate]]] = {}
    provisional_counter = 1

    def ensure_family_pool(family: TransitionFamily) -> dict[str, list[EpisodeCandidate]]:
        nonlocal provisional_counter
        if family.family_id not in family_cache:
            task_pool: dict[str, list[EpisodeCandidate]] = {}
            for suite_task_id in SUITE_TASKS:
                task_pool[suite_task_id] = build_family_task_candidates(
                    suite_task_id,
                    family,
                    provisional_counter_start=provisional_counter,
                )
                provisional_counter += PUBLIC_VARIANTS_PER_FAMILY
            family_cache[family.family_id] = task_pool
        return family_cache[family.family_id]

    def try_select(family: TransitionFamily) -> bool:
        if initial_usage[family.initial_rule_id] >= 2:
            return False
        if shift_usage[family.shift_rule_id] >= 2:
            return False
        if initial_family_usage[family.initial_family_id] >= MAX_SELECTED_TRANSITIONS_PER_FAMILY_ROLE:
            return False
        if shift_family_usage[family.shift_family_id] >= MAX_SELECTED_TRANSITIONS_PER_FAMILY_ROLE:
            return False
        family_pair = (family.initial_family_id, family.shift_family_id)
        if family_pair_usage[family_pair] >= MAX_SELECTED_TRANSITIONS_PER_FAMILY_PAIR:
            return False
        task_pool = ensure_family_pool(family)
        if any(len(task_pool[suite_task_id]) < 2 for suite_task_id in SUITE_TASKS):
            return False
        selected.append(family)
        initial_usage[family.initial_rule_id] += 1
        shift_usage[family.shift_rule_id] += 1
        initial_family_usage[family.initial_family_id] += 1
        shift_family_usage[family.shift_family_id] += 1
        family_pair_usage[family_pair] += 1
        for suite_task_id in SUITE_TASKS:
            selected_pool[suite_task_id][family.family_id] = task_pool[suite_task_id]
        return True

    while len(selected) < TRANSITION_FAMILY_COUNT:
        progress = False
        for key in ordered_keys:
            bucket = buckets[key]
            while bucket:
                family = bucket.pop(0)
                if not try_select(family):
                    continue
                progress = True
                break
            if len(selected) >= TRANSITION_FAMILY_COUNT:
                break
        if not progress:
            break

    if len(selected) != TRANSITION_FAMILY_COUNT:
        raise RuntimeError(f"unable to select {TRANSITION_FAMILY_COUNT} supported transition families")
    return tuple(selected), selected_pool


def assign_difficulty_bins(answers: list[dict[str, object]], rows: list[dict[str, object]]) -> None:
    by_task: dict[str, list[tuple[dict[str, object], dict[str, object]]]] = defaultdict(list)
    for row, answer in zip(rows, answers):
        by_task[str(answer["suite_task_id"])].append((row, answer))
    for suite_task_id in SUITE_TASKS:
        pairs = by_task[suite_task_id]
        pairs.sort(
            key=lambda pair: (
                candidate_attack_signal(pair[1]["generator_diagnostics"], suite_task_id),
                str(pair[1]["transition_family_id"]),
                str(pair[1]["episode_id"]),
            )
        )
        hard_count = len(pairs) // 2
        for index, (row, answer) in enumerate(pairs):
            difficulty_bin = "hard" if index < hard_count else "medium"
            row["analysis"]["difficulty_bin"] = difficulty_bin
            answer["difficulty_bin"] = difficulty_bin


def apply_retrieval_scores(answers: list[dict[str, object]]) -> None:
    retrieval_scores = public_retrieval_scores(answers)
    for answer in answers:
        answer["generator_diagnostics"]["public_retrieval_accuracy"] = retrieval_scores[str(answer["episode_id"])]


def select_public_split() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    selected_families, pool = build_candidate_pool()
    selected_rows: list[dict[str, object]] = []
    selected_answers: list[dict[str, object]] = []
    episode_counter = 1

    for suite_task_id in SUITE_TASKS:
        for family in selected_families:
            family_candidates = pool[suite_task_id][family.family_id]
            chosen = pick_family_candidates(family_candidates)
            if len(chosen) != 2:
                raise RuntimeError(f"expected two candidates for {suite_task_id}/{family.family_id}")
            for candidate in chosen:
                episode_id = f"{episode_counter:04d}"
                row = json.loads(json.dumps(candidate.row))
                answer = json.loads(json.dumps(candidate.answer))
                row["episode_id"] = episode_id
                answer["episode_id"] = episode_id
                row["inference"]["turns"] = [turn.replace(candidate.row["episode_id"], episode_id) for turn in row["inference"]["turns"]]
                answer["turns"] = [turn.replace(candidate.answer["episode_id"], episode_id) for turn in answer["turns"]]
                selected_rows.append(row)
                selected_answers.append(answer)
                episode_counter += 1

    apply_retrieval_scores(selected_answers)
    assign_difficulty_bins(selected_answers, selected_rows)
    validate_public_split(selected_rows, selected_answers)
    return selected_rows, selected_answers


def episode_signature(answer: dict[str, object]) -> tuple[object, ...]:
    return (
        answer["suite_task_id"],
        tuple(answer["turns"]),
        tuple(answer["final_probe_targets"]),
    )


def validate_answer_uniqueness(answers: list[dict[str, object]]) -> None:
    seen: dict[tuple[object, ...], str] = {}
    for answer in answers:
        signature = episode_signature(answer)
        previous = seen.get(signature)
        if previous is not None:
            raise ValueError(
                f"public split contains duplicate semantic episode content: {previous} and {answer['episode_id']}"
            )
        seen[signature] = str(answer["episode_id"])


def validate_public_split(rows: list[dict[str, object]], answers: list[dict[str, object]]) -> None:
    if len(rows) != len(answers):
        raise ValueError("rows and answers length mismatch")
    if len(rows) != len(SUITE_TASKS) * PUBLIC_EPISODES_PER_TASK:
        raise ValueError(f"public split expected {len(SUITE_TASKS) * PUBLIC_EPISODES_PER_TASK} rows, found {len(rows)}")
    validate_answer_uniqueness(answers)
    task_counts = Counter(str(row["analysis"]["suite_task_id"]) for row in rows)
    if task_counts != Counter({suite_task_id: PUBLIC_EPISODES_PER_TASK for suite_task_id in SUITE_TASKS}):
        raise ValueError(f"public split task counts mismatch: {task_counts}")
    difficulty_counts = Counter(str(row["analysis"]["difficulty_bin"]) for row in rows)
    if difficulty_counts != Counter({"hard": len(rows) // 2, "medium": len(rows) // 2}):
        raise ValueError(f"public split difficulty counts mismatch: {difficulty_counts}")

    per_task_difficulty = {
        suite_task_id: Counter(
            str(row["analysis"]["difficulty_bin"])
            for row in rows
            if row["analysis"]["suite_task_id"] == suite_task_id
        )
        for suite_task_id in SUITE_TASKS
    }
    for suite_task_id, counts in per_task_difficulty.items():
        if counts != Counter({"hard": PUBLIC_EPISODES_PER_TASK // 2, "medium": PUBLIC_EPISODES_PER_TASK // 2}):
            raise ValueError(f"public split per-task difficulty counts mismatch for {suite_task_id}: {counts}")

    for answer in answers:
        validate_episode_constraints(answer)
        retrieval_score = answer["generator_diagnostics"]["public_retrieval_accuracy"]
        retrieval_limit = attack_limits_for_task(str(answer["suite_task_id"]))["public_retrieval_accuracy"]
        if retrieval_score is None or float(retrieval_score) > retrieval_limit:
            raise ValueError(
                f"episode {answer['episode_id']} violates retrieval ceiling: {retrieval_score!r}"
            )


def summarize_attacks(answers: list[dict[str, object]]) -> dict[str, object]:
    task_buckets: dict[str, list[dict[str, object]]] = defaultdict(list)
    for answer in answers:
        task_buckets[str(answer["suite_task_id"])].append(answer)

    metrics = (
        "previous_rule_accuracy",
        "learn_only_max_probe_accuracy",
        "majority_label_accuracy",
        "nearest_neighbor_accuracy",
        "cue_agnostic_accuracy",
        "dsl_search_accuracy",
        "public_retrieval_accuracy",
    )

    def summarize_metric(metric: str) -> dict[str, object]:
        per_task: dict[str, float | None] = {}
        values: list[float] = []
        for suite_task_id in SUITE_TASKS:
            metric_values = [
                float(answer["generator_diagnostics"][metric])
                for answer in task_buckets[suite_task_id]
                if answer["generator_diagnostics"][metric] is not None
            ]
            per_task[suite_task_id] = round(sum(metric_values) / len(metric_values), 4) if metric_values else None
            values.extend(metric_values)
        return {
            "micro_accuracy": round(sum(values) / len(values), 4) if values else None,
            "per_task_accuracy": per_task,
        }

    return {metric: summarize_metric(metric) for metric in metrics}


def public_quality_report(rows: list[dict[str, object]], answers: list[dict[str, object]]) -> dict[str, object]:
    family_counts = Counter(str(answer["transition_family_id"]) for answer in answers)
    family_pair_counts = Counter(
        f"{answer['initial_rule_family_id']}->{answer['shift_rule_family_id']}" for answer in answers
    )
    initial_family_selection_counts = Counter(str(answer["initial_rule_family_id"]) for answer in answers)
    shift_family_selection_counts = Counter(str(answer["shift_rule_family_id"]) for answer in answers)
    disagreement_bin_counts = Counter(str(answer["generator_diagnostics"]["transition_disagreement_bin"]) for answer in answers)
    cue_lexicons = sorted({str(answer["cue_lexicon_id"]) for answer in answers})
    context_lexicons = sorted({str(answer["context_lexicon_id"]) for answer in answers})
    rule_family_counts = Counter(rule.family_id for rule in PUBLIC_RULES)
    turn2_required_counts = Counter(int(answer["generator_diagnostics"]["turn2_required_probe_count"]) for answer in answers)
    post_shift_prediction_set_sizes = Counter(
        int(answer["generator_diagnostics"]["post_shift_prediction_set_size"]) for answer in answers
    )
    learn_only_values = [
        float(answer["generator_diagnostics"]["learn_only_max_probe_accuracy"])
        for answer in answers
    ]
    return {
        "version": PUBLIC_BUNDLE_VERSION,
        "task_name": TASK_NAME,
        "split": "public",
        "row_count": len(rows),
        "episodes_per_task": PUBLIC_EPISODES_PER_TASK,
        "rule_inventory_count": len(PUBLIC_RULES),
        "rule_family_count": len(rule_family_counts),
        "rule_family_rule_counts": dict(sorted(rule_family_counts.items())),
        "transition_family_count": len(family_counts),
        "transition_family_usage": dict(sorted(family_counts.items())),
        "selected_initial_family_count": len(initial_family_selection_counts),
        "selected_initial_family_usage": dict(sorted(initial_family_selection_counts.items())),
        "selected_shift_family_count": len(shift_family_selection_counts),
        "selected_shift_family_usage": dict(sorted(shift_family_selection_counts.items())),
        "transition_pair_pattern_count": len(family_pair_counts),
        "transition_pair_usage": dict(sorted(family_pair_counts.items())),
        "disagreement_bin_counts": dict(sorted(disagreement_bin_counts.items())),
        "difficulty_bin_counts": dict(
            sorted(Counter(str(row["analysis"]["difficulty_bin"]) for row in rows).items())
        ),
        "cue_lexicon_ids": cue_lexicons,
        "context_lexicon_ids": context_lexicons,
        "switching_diagnostics": {
            "learn_only_max_probe_accuracy": round(sum(learn_only_values) / len(learn_only_values), 4),
            "turn2_required_probe_count_distribution": {
                str(key): value for key, value in sorted(turn2_required_counts.items())
            },
            "post_shift_prediction_set_size_distribution": {
                str(key): value for key, value in sorted(post_shift_prediction_set_sizes.items())
            },
        },
        "attack_suite": summarize_attacks(answers),
    }


def dataset_metadata(dataset_id: str, title: str) -> dict[str, object]:
    return {
        "id": dataset_id,
        "title": title,
        "licenses": [{"name": "CC0-1.0"}],
    }


def build_public_artifacts() -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    rows, answers = select_public_split()
    report = public_quality_report(rows, answers)
    return rows, answers, report


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    public_rows, _public_answers, report = build_public_artifacts()
    write_json(PUBLIC_ROWS_PATH, public_rows)
    write_json(PUBLIC_METADATA_PATH, dataset_metadata(PUBLIC_DATASET_ID, "CogFlex Suite Runtime"))
    write_json(PUBLIC_QUALITY_REPORT_PATH, report)
    print(f"Wrote {len(public_rows)} public episodes to {PUBLIC_ROWS_PATH}")
    print(f"Wrote public dataset metadata to {PUBLIC_METADATA_PATH}")
    print(f"Wrote public quality report to {PUBLIC_QUALITY_REPORT_PATH}")


if __name__ == "__main__":
    main()
