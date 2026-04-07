#!/usr/bin/env python3

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROWS_PATH = ROOT / "kaggle/dataset/public/public_leaderboard_rows.json"
PRIVATE_ROWS_PATH = ROOT / "kaggle/dataset/private/private_leaderboard_rows.json"
ANSWER_KEY_PATH = ROOT / "kaggle/dataset/audit_key.json"
PRIVATE_METADATA_PATH = ROOT / "kaggle/dataset/private/dataset-metadata.json"
PRIVATE_DATASET_ID = "raptorengineer/ruleshift-runtime-private"

VALUES = (-3, -2, -1, 1, 2, 3)
DOMAIN = [(r1, r2) for r1 in VALUES for r2 in VALUES]
LABEL_TRUE = "zark"
LABEL_FALSE = "blim"
TYPE_TRUE = "type_a"
TYPE_FALSE = "type_b"
GROUPS = ("simple", "exception", "distractor", "hard")
MEMOS = ("archived", "reviewed", "queue-ok", "duplicate-check", "carry-forward")
OUTPUT_INSTRUCTION = (
    "Return exactly 4 outputs in order, one per probe. "
    "Use only type_a or type_b. Map zark to type_a and blim to type_b."
)


def fmt_signed(value: int) -> str:
    return f"{value:+d}"


@dataclass(frozen=True)
class RuleSpec:
    rule_id: str
    description: str
    shortcut_text: str
    predicate: Callable[[int, int], bool]
    true_reason: Callable[[int, int], str]
    false_reason: Callable[[int, int], str]

    def label(self, point: tuple[int, int]) -> bool:
        return self.predicate(*point)

    def explain(self, point: tuple[int, int]) -> str:
        return (
            self.true_reason(*point)
            if self.label(point)
            else self.false_reason(*point)
        )


def make_rule(
    rule_id: str,
    description: str,
    shortcut_text: str,
    predicate: Callable[[int, int], bool],
    true_reason: Callable[[int, int], str],
    false_reason: Callable[[int, int], str],
) -> RuleSpec:
    return RuleSpec(
        rule_id=rule_id,
        description=description,
        shortcut_text=shortcut_text,
        predicate=predicate,
        true_reason=true_reason,
        false_reason=false_reason,
    )


RULES: tuple[RuleSpec, ...] = (
    make_rule(
        "r1_gt_r2",
        "type_a iff r1 is greater than r2",
        "guess from the sign of r1 instead of comparing both markers",
        lambda r1, r2: r1 > r2,
        lambda r1, r2: f"{fmt_signed(r1)} is greater than {fmt_signed(r2)}.",
        lambda r1, r2: f"{fmt_signed(r1)} is not greater than {fmt_signed(r2)}.",
    ),
    make_rule(
        "r1_lt_r2",
        "type_a iff r1 is less than r2",
        "guess from the sign of r2 instead of comparing both markers",
        lambda r1, r2: r1 < r2,
        lambda r1, r2: f"{fmt_signed(r1)} is less than {fmt_signed(r2)}.",
        lambda r1, r2: f"{fmt_signed(r1)} is not less than {fmt_signed(r2)}.",
    ),
    make_rule(
        "sum_positive",
        "type_a iff r1 + r2 is positive",
        "focus on one marker instead of the combined total",
        lambda r1, r2: r1 + r2 > 0,
        lambda r1, r2: f"{fmt_signed(r1)} + {fmt_signed(r2)} = {r1 + r2:+d}, which is positive.",
        lambda r1, r2: f"{fmt_signed(r1)} + {fmt_signed(r2)} = {r1 + r2:+d}, which is not positive.",
    ),
    make_rule(
        "sum_negative",
        "type_a iff r1 + r2 is negative",
        "focus on one marker instead of the combined total",
        lambda r1, r2: r1 + r2 < 0,
        lambda r1, r2: f"{fmt_signed(r1)} + {fmt_signed(r2)} = {r1 + r2:+d}, which is negative.",
        lambda r1, r2: f"{fmt_signed(r1)} + {fmt_signed(r2)} = {r1 + r2:+d}, which is not negative.",
    ),
    make_rule(
        "abs_r1_gt_abs_r2",
        "type_a iff |r1| is greater than |r2|",
        "look at raw signs instead of comparing magnitudes",
        lambda r1, r2: abs(r1) > abs(r2),
        lambda r1, r2: f"|{fmt_signed(r1)}| = {abs(r1)} is greater than |{fmt_signed(r2)}| = {abs(r2)}.",
        lambda r1, r2: f"|{fmt_signed(r1)}| = {abs(r1)} is not greater than |{fmt_signed(r2)}| = {abs(r2)}.",
    ),
    make_rule(
        "abs_r1_lt_abs_r2",
        "type_a iff |r1| is less than |r2|",
        "look at raw signs instead of comparing magnitudes",
        lambda r1, r2: abs(r1) < abs(r2),
        lambda r1, r2: f"|{fmt_signed(r1)}| = {abs(r1)} is less than |{fmt_signed(r2)}| = {abs(r2)}.",
        lambda r1, r2: f"|{fmt_signed(r1)}| = {abs(r1)} is not less than |{fmt_signed(r2)}| = {abs(r2)}.",
    ),
    make_rule(
        "abs_equal",
        "type_a iff |r1| equals |r2|",
        "treat sign match as sufficient instead of matching magnitudes",
        lambda r1, r2: abs(r1) == abs(r2),
        lambda r1, r2: f"|{fmt_signed(r1)}| and |{fmt_signed(r2)}| are both {abs(r1)}.",
        lambda r1, r2: f"|{fmt_signed(r1)}| = {abs(r1)} and |{fmt_signed(r2)}| = {abs(r2)}, so they differ.",
    ),
    make_rule(
        "r1_positive",
        "type_a iff r1 is positive",
        "overweight the second marker even though only the first matters",
        lambda r1, r2: r1 > 0,
        lambda r1, r2: f"r1 is {fmt_signed(r1)}, which is positive.",
        lambda r1, r2: f"r1 is {fmt_signed(r1)}, which is not positive.",
    ),
    make_rule(
        "r2_positive",
        "type_a iff r2 is positive",
        "overweight the first marker even though only the second matters",
        lambda r1, r2: r2 > 0,
        lambda r1, r2: f"r2 is {fmt_signed(r2)}, which is positive.",
        lambda r1, r2: f"r2 is {fmt_signed(r2)}, which is not positive.",
    ),
    make_rule(
        "r1_negative",
        "type_a iff r1 is negative",
        "overweight the second marker even though only the first matters",
        lambda r1, r2: r1 < 0,
        lambda r1, r2: f"r1 is {fmt_signed(r1)}, which is negative.",
        lambda r1, r2: f"r1 is {fmt_signed(r1)}, which is not negative.",
    ),
    make_rule(
        "r2_negative",
        "type_a iff r2 is negative",
        "overweight the first marker even though only the second matters",
        lambda r1, r2: r2 < 0,
        lambda r1, r2: f"r2 is {fmt_signed(r2)}, which is negative.",
        lambda r1, r2: f"r2 is {fmt_signed(r2)}, which is not negative.",
    ),
    make_rule(
        "both_positive",
        "type_a iff both markers are positive",
        "treat any positive marker as sufficient",
        lambda r1, r2: r1 > 0 and r2 > 0,
        lambda r1, r2: f"both r1={fmt_signed(r1)} and r2={fmt_signed(r2)} are positive.",
        lambda r1, r2: f"at least one of r1={fmt_signed(r1)} or r2={fmt_signed(r2)} is not positive.",
    ),
    make_rule(
        "both_negative",
        "type_a iff both markers are negative",
        "treat any negative marker as sufficient",
        lambda r1, r2: r1 < 0 and r2 < 0,
        lambda r1, r2: f"both r1={fmt_signed(r1)} and r2={fmt_signed(r2)} are negative.",
        lambda r1, r2: f"at least one of r1={fmt_signed(r1)} or r2={fmt_signed(r2)} is not negative.",
    ),
    make_rule(
        "same_sign",
        "type_a iff r1 and r2 have the same sign",
        "track only magnitude and ignore sign agreement",
        lambda r1, r2: (r1 > 0) == (r2 > 0),
        lambda r1, r2: f"r1={fmt_signed(r1)} and r2={fmt_signed(r2)} share the same sign.",
        lambda r1, r2: f"r1={fmt_signed(r1)} and r2={fmt_signed(r2)} do not share the same sign.",
    ),
    make_rule(
        "opposite_sign",
        "type_a iff r1 and r2 have different signs",
        "track only magnitude and ignore sign mismatch",
        lambda r1, r2: (r1 > 0) != (r2 > 0),
        lambda r1, r2: f"r1={fmt_signed(r1)} and r2={fmt_signed(r2)} have different signs.",
        lambda r1, r2: f"r1={fmt_signed(r1)} and r2={fmt_signed(r2)} have the same sign.",
    ),
    make_rule(
        "sum_at_least_two",
        "type_a iff r1 + r2 is at least +2",
        "use a looser positive-sum shortcut and miss the threshold",
        lambda r1, r2: r1 + r2 >= 2,
        lambda r1, r2: f"{fmt_signed(r1)} + {fmt_signed(r2)} = {r1 + r2:+d}, which meets the +2 threshold.",
        lambda r1, r2: f"{fmt_signed(r1)} + {fmt_signed(r2)} = {r1 + r2:+d}, which does not meet the +2 threshold.",
    ),
    make_rule(
        "sum_at_most_minus_two",
        "type_a iff r1 + r2 is at most -2",
        "use a looser negative-sum shortcut and miss the threshold",
        lambda r1, r2: r1 + r2 <= -2,
        lambda r1, r2: f"{fmt_signed(r1)} + {fmt_signed(r2)} = {r1 + r2:+d}, which meets the -2 threshold.",
        lambda r1, r2: f"{fmt_signed(r1)} + {fmt_signed(r2)} = {r1 + r2:+d}, which does not meet the -2 threshold.",
    ),
    make_rule(
        "diff_at_least_two",
        "type_a iff r1 - r2 is at least +2",
        "compare direction but miss the distance threshold",
        lambda r1, r2: r1 - r2 >= 2,
        lambda r1, r2: f"{fmt_signed(r1)} - {fmt_signed(r2)} = {r1 - r2:+d}, which meets the +2 threshold.",
        lambda r1, r2: f"{fmt_signed(r1)} - {fmt_signed(r2)} = {r1 - r2:+d}, which does not meet the +2 threshold.",
    ),
    make_rule(
        "diff_at_most_minus_two",
        "type_a iff r1 - r2 is at most -2",
        "compare direction but miss the distance threshold",
        lambda r1, r2: r1 - r2 <= -2,
        lambda r1, r2: f"{fmt_signed(r1)} - {fmt_signed(r2)} = {r1 - r2:+d}, which meets the -2 threshold.",
        lambda r1, r2: f"{fmt_signed(r1)} - {fmt_signed(r2)} = {r1 - r2:+d}, which does not meet the -2 threshold.",
    ),
    make_rule(
        "both_large_abs",
        "type_a iff both markers have absolute value at least 2",
        "use one large marker as a shortcut instead of checking both",
        lambda r1, r2: abs(r1) >= 2 and abs(r2) >= 2,
        lambda r1, r2: f"both |{fmt_signed(r1)}|={abs(r1)} and |{fmt_signed(r2)}|={abs(r2)} are at least 2.",
        lambda r1, r2: f"at least one of |{fmt_signed(r1)}|={abs(r1)} or |{fmt_signed(r2)}|={abs(r2)} is below 2.",
    ),
)

RULE_BY_ID = {rule.rule_id: rule for rule in RULES}


def format_line(
    idx: int,
    point: tuple[int, int],
    label: str,
    group_id: str,
    memo: str | None,
) -> str:
    r1, r2 = point
    rendered_label = label
    if group_id == "distractor":
        return (
            f"{idx}. memo={memo} | r1={fmt_signed(r1)} | "
            f"r2={fmt_signed(r2)} -> {rendered_label}"
        )
    return f"{idx}. r1={fmt_signed(r1)}, r2={fmt_signed(r2)} -> {rendered_label}"


def render_prompt(
    episode_id: str,
    group_id: str,
    examples: list[tuple[int, int]],
    probes: list[tuple[int, int]],
    rule: RuleSpec,
    memo_cycle: list[str],
) -> str:
    example_lines = [
        format_line(i + 1, point, LABEL_TRUE if rule.label(point) else LABEL_FALSE, group_id, memo_cycle[i])
        for i, point in enumerate(examples)
    ]
    probe_lines = [
        format_line(i + 6, point, "?", group_id, memo_cycle[i + 5])
        for i, point in enumerate(probes)
    ]
    return "\n\n".join(
        [
            f"RuleShift classification task. Episode {episode_id}.",
            "Examples:\n" + "\n".join(example_lines),
            "Probes:\n" + "\n".join(probe_lines),
            OUTPUT_INSTRUCTION,
        ]
    )


def labeled_points(rule: RuleSpec, points: list[tuple[int, int]]) -> list[tuple[tuple[int, int], bool]]:
    return [(point, rule.label(point)) for point in points]


def consistent_rules(points: list[tuple[tuple[int, int], bool]]) -> list[RuleSpec]:
    survivors: list[RuleSpec] = []
    for rule in RULES:
        if all(rule.label(point) == label for point, label in points):
            survivors.append(rule)
    return survivors


def score_probe_set(points: list[tuple[int, int]], candidates: list[RuleSpec]) -> int:
    score = 0
    for point in points:
        values = {rule.label(point) for rule in candidates}
        score += len(values)
    return score


def choose_shortcut(
    actual_rule: RuleSpec,
    labeled_example_points: list[tuple[tuple[int, int], bool]],
) -> RuleSpec:
    ranked: list[tuple[int, int, str, RuleSpec]] = []
    for rule in RULES:
        if rule.rule_id == actual_rule.rule_id:
            continue
        matches = sum(
            rule.label(point) == label
            for point, label in labeled_example_points
        )
        first_four_matches = sum(
            rule.label(point) == label
            for point, label in labeled_example_points[:4]
        )
        ranked.append((matches, first_four_matches, rule.rule_id, rule))
    ranked.sort(reverse=True)
    return ranked[0][3]


def balanced_points(
    rng: random.Random,
    points: list[tuple[int, int]],
    rule: RuleSpec,
    count: int,
    min_positive: int | None = None,
    min_negative: int | None = None,
) -> list[tuple[int, int]]:
    positives = [point for point in points if rule.label(point)]
    negatives = [point for point in points if not rule.label(point)]
    pos_need = count // 2 if min_positive is None else min_positive
    neg_need = count - pos_need if min_negative is None else min_negative
    if pos_need + neg_need > count:
        raise ValueError("Requested label minimums exceed requested point count")
    rng.shuffle(positives)
    rng.shuffle(negatives)
    if len(positives) < pos_need or len(negatives) < neg_need:
        raise ValueError(f"Not enough balanced points for {rule.rule_id}")
    chosen = positives[:pos_need] + negatives[:neg_need]
    remaining = [point for point in points if point not in chosen]
    rng.shuffle(remaining)
    chosen.extend(remaining[: count - len(chosen)])
    rng.shuffle(chosen)
    return chosen


def find_simple_episode(rule: RuleSpec, rng: random.Random) -> tuple[list[tuple[int, int]], list[tuple[int, int]], RuleSpec]:
    for _ in range(20_000):
        examples = balanced_points(rng, DOMAIN[:], rule, 5)
        example_labels = labeled_points(rule, examples)
        survivors = consistent_rules(example_labels)
        if [item.rule_id for item in survivors] != [rule.rule_id]:
            continue
        remaining = [point for point in DOMAIN if point not in examples]
        probes = balanced_points(rng, remaining, rule, 4)
        shortcut = choose_shortcut(rule, example_labels)
        return (examples, probes, shortcut)
    raise RuntimeError(f"Failed to find simple episode for {rule.rule_id}")


def find_exception_episode(rule: RuleSpec, rng: random.Random) -> tuple[list[tuple[int, int]], list[tuple[int, int]], RuleSpec]:
    alternatives = [alt for alt in RULES if alt.rule_id != rule.rule_id]
    rng.shuffle(alternatives)
    for alt in alternatives:
        overlap = [point for point in DOMAIN if rule.label(point) == alt.label(point)]
        divergence = [point for point in DOMAIN if rule.label(point) != alt.label(point)]
        for _ in range(4_000):
            try:
                first_four = balanced_points(
                    rng,
                    overlap[:],
                    rule,
                    4,
                    min_positive=1,
                    min_negative=1,
                )
            except ValueError:
                continue
            first_four_labels = labeled_points(rule, first_four)
            first_four_survivors = {item.rule_id for item in consistent_rules(first_four_labels)}
            if rule.rule_id not in first_four_survivors or alt.rule_id not in first_four_survivors:
                continue
            divergence_points = divergence[:]
            rng.shuffle(divergence_points)
            for exception_point in divergence_points:
                examples = first_four + [exception_point]
                example_labels = labeled_points(rule, examples)
                survivors = consistent_rules(example_labels)
                if [item.rule_id for item in survivors] != [rule.rule_id]:
                    continue
                disagreement_points = [
                    point
                    for point in DOMAIN
                    if point not in examples and rule.label(point) != alt.label(point)
                ]
                same_points = [
                    point
                    for point in DOMAIN
                    if point not in examples and rule.label(point) == alt.label(point)
                ]
                if len(disagreement_points) < 2 or len(same_points) < 2:
                    continue
                rng.shuffle(disagreement_points)
                rng.shuffle(same_points)
                probes = disagreement_points[:2] + same_points[:2]
                rng.shuffle(probes)
                return (examples, probes, alt)
    raise RuntimeError(f"Failed to find exception episode for {rule.rule_id}")


def find_distractor_episode(rule: RuleSpec, rng: random.Random) -> tuple[list[tuple[int, int]], list[tuple[int, int]], RuleSpec]:
    examples, probes, shortcut = find_simple_episode(rule, rng)
    return (examples, probes, shortcut)


def find_hard_episode(rule: RuleSpec, rng: random.Random) -> tuple[list[tuple[int, int]], list[tuple[int, int]], RuleSpec]:
    for _ in range(40_000):
        examples = balanced_points(rng, DOMAIN[:], rule, 5)
        first_four_labels = labeled_points(rule, examples[:4])
        first_four_survivors = consistent_rules(first_four_labels)
        if len(first_four_survivors) < 3:
            continue
        example_labels = labeled_points(rule, examples)
        survivors = consistent_rules(example_labels)
        if [item.rule_id for item in survivors] != [rule.rule_id]:
            continue
        remaining = [point for point in DOMAIN if point not in examples]
        probe_pool = [point for point in remaining if rule.label(point)] + [point for point in remaining if not rule.label(point)]
        best: tuple[int, list[tuple[int, int]]] | None = None
        for _ in range(200):
            probes = balanced_points(rng, probe_pool[:], rule, 4)
            score = score_probe_set(probes, first_four_survivors)
            if best is None or score > best[0]:
                best = (score, probes)
        if best is None:
            continue
        shortcut = choose_shortcut(rule, first_four_labels)
        return (examples, best[1], shortcut)
    raise RuntimeError(f"Failed to find hard episode for {rule.rule_id}")


GROUP_BUILDERS = {
    "simple": find_simple_episode,
    "exception": find_exception_episode,
    "distractor": find_distractor_episode,
    "hard": find_hard_episode,
}


def probe_targets(rule: RuleSpec, probes: list[tuple[int, int]]) -> list[str]:
    return [TYPE_TRUE if rule.label(point) else TYPE_FALSE for point in probes]


def memo_cycle_for_episode(rng: random.Random) -> list[str]:
    cycle = list(MEMOS) * 2
    rng.shuffle(cycle)
    return cycle[:9]


def episode_record(
    episode_id: str,
    group_id: str,
    rule: RuleSpec,
    examples: list[tuple[int, int]],
    probes: list[tuple[int, int]],
    shortcut: RuleSpec | None,
    memo_cycle: list[str],
) -> tuple[dict[str, object], dict[str, object]]:
    prompt = render_prompt(episode_id, group_id, examples, probes, rule, memo_cycle)
    shortcut_payload: dict[str, object]
    if group_id == "distractor":
        shortcut_payload = {
            "shortcut_type": "metadata_noise",
            "shortcut_text": "treat memo tags as if they carried label information",
        }
    else:
        assert shortcut is not None
        shortcut_payload = {
            "shortcut_type": "rule_shortcut",
            "shortcut_rule_id": shortcut.rule_id,
            "shortcut_text": shortcut.description,
        }
    row = {
        "episode_id": episode_id,
        "inference": {
            "prompt": prompt,
        },
        "scoring": {
            "probe_targets": probe_targets(rule, probes),
        },
        "analysis": {
            "group_id": group_id,
            "rule_id": rule.rule_id,
            "shortcut_type": shortcut_payload["shortcut_type"],
            "shortcut_rule_id": shortcut_payload.get("shortcut_rule_id"),
        },
    }
    answer = {
        "episode_id": episode_id,
        "group_id": group_id,
        "rule_id": rule.rule_id,
        "rule_description": rule.description,
        **shortcut_payload,
        "examples": [
            {
                "index": idx + 1,
                "r1": point[0],
                "r2": point[1],
                "label": TYPE_TRUE if rule.label(point) else TYPE_FALSE,
            }
            for idx, point in enumerate(examples)
        ],
        "probes": [
            {
                "index": idx + 6,
                "r1": point[0],
                "r2": point[1],
                "label": TYPE_TRUE if rule.label(point) else TYPE_FALSE,
                "justification": rule.explain(point),
            }
            for idx, point in enumerate(probes)
        ],
    }
    return (row, answer)


def episode_signature(answer: dict[str, object]) -> tuple[object, ...]:
    return (
        answer["group_id"],
        answer["rule_id"],
        tuple(
            (example["r1"], example["r2"], example["label"])
            for example in answer["examples"]
        ),
        tuple(
            (probe["r1"], probe["r2"], probe["label"])
            for probe in answer["probes"]
        ),
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


def validate_split_isolation(
    public_answers: list[dict[str, object]],
    private_answers: list[dict[str, object]],
) -> None:
    validate_answer_uniqueness(public_answers, "public")
    validate_answer_uniqueness(private_answers, "private")

    public_signatures = {
        episode_signature(answer): str(answer["episode_id"])
        for answer in public_answers
    }
    for answer in private_answers:
        signature = episode_signature(answer)
        public_episode_id = public_signatures.get(signature)
        if public_episode_id is None:
            continue
        raise ValueError(
            "public/private split isolation violated: "
            f"public episode {public_episode_id} overlaps private episode {answer['episode_id']}"
        )


def build_split(
    split_name: str,
    variants_per_rule: int,
    variant_start: int = 0,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    answers: list[dict[str, object]] = []
    episode_counter = 1
    for group_index, group_id in enumerate(GROUPS):
        for variant in range(variant_start, variant_start + variants_per_rule):
            for rule_index, rule in enumerate(RULES):
                seed = 10_000 * group_index + 1_000 * variant + rule_index
                rng = random.Random(seed)
                examples, probes, shortcut = GROUP_BUILDERS[group_id](rule, rng)
                memo_cycle = memo_cycle_for_episode(random.Random(seed + 7))
                episode_id = f"{episode_counter:04d}"
                row, answer = episode_record(
                    episode_id,
                    group_id,
                    rule,
                    examples,
                    probes,
                    shortcut,
                    memo_cycle,
                )
                row["split"] = split_name
                answer["split"] = split_name
                rows.append(row)
                answers.append(answer)
                episode_counter += 1
    return (rows, answers)


def validate_rows(rows: list[dict[str, object]], expected_count: int, per_group: int) -> None:
    assert len(rows) == expected_count, (len(rows), expected_count)
    counts = Counter(str(row["analysis"]["group_id"]) for row in rows)
    assert counts == Counter({group: per_group for group in GROUPS}), counts
    for row in rows:
        prompt = str(row["inference"]["prompt"])
        parts = prompt.split("\n\n")
        assert len(parts) == 4, row["episode_id"]
        assert parts[0] == f"RuleShift classification task. Episode {row['episode_id']}.", row["episode_id"]
        assert parts[1].startswith("Examples:\n"), row["episode_id"]
        assert parts[2].startswith("Probes:\n"), row["episode_id"]
        assert parts[3] == OUTPUT_INSTRUCTION, row["episode_id"]
        assert len(row["scoring"]["probe_targets"]) == 4, row["episode_id"]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def dataset_metadata(dataset_id: str, title: str) -> dict[str, object]:
    return {
        "id": dataset_id,
        "title": title,
        "licenses": [{"name": "CC0-1.0"}],
    }


def main() -> None:
    public_rows, public_answers = build_split("public", variants_per_rule=1)
    private_rows, private_answers = build_split(
        "private",
        variants_per_rule=5,
        variant_start=1,
    )

    for row in public_rows:
        row.pop("split")
    for row in private_rows:
        row.pop("split")

    validate_rows(public_rows, expected_count=80, per_group=20)
    validate_rows(private_rows, expected_count=400, per_group=100)
    validate_split_isolation(public_answers, private_answers)

    write_json(PUBLIC_ROWS_PATH, public_rows)
    write_json(PRIVATE_ROWS_PATH, private_rows)
    write_json(
        PRIVATE_METADATA_PATH,
        dataset_metadata(PRIVATE_DATASET_ID, "RuleShift Runtime Private"),
    )
    write_json(
        ANSWER_KEY_PATH,
        {
            "version": "scoped_single_turn",
            "public": public_answers,
            "private": private_answers,
        },
    )

    print(f"Wrote {len(public_rows)} public episodes to {PUBLIC_ROWS_PATH}")
    print(f"Wrote {len(private_rows)} private episodes to {PRIVATE_ROWS_PATH}")
    print(f"Wrote private metadata to {PRIVATE_METADATA_PATH}")
    print(f"Wrote answer key to {ANSWER_KEY_PATH}")


if __name__ == "__main__":
    main()
