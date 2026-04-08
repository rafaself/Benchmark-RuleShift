#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Final

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.build_cogflex_dataset import (  # noqa: E402
    FACULTY_ID,
    FINAL_OUTPUT_INSTRUCTION,
    FINAL_PROBE_COUNT,
    LEARN_EXAMPLE_COUNT,
    PRIVATE_ANSWER_KEY_FILENAME,
    PRIVATE_BUNDLE_ENV_VAR,
    PRIVATE_BUNDLE_VERSION,
    PRIVATE_EPISODES_PER_TASK,
    PRIVATE_QUALITY_REPORT_FILENAME,
    PRIVATE_QUALITY_REPORT_VERSION,
    PRIVATE_RELEASE_MANIFEST_FILENAME,
    PRIVATE_ROWS_FILENAME,
    PUBLIC_CONTEXT_LEXICONS,
    PUBLIC_CUE_LEXICONS,
    PUBLIC_EPISODES_PER_TASK,
    PUBLIC_FAMILY_IDS,
    PUBLIC_QUALITY_REPORT_PATH,
    PUBLIC_RULES,
    PUBLIC_ROWS_PATH,
    SHIFT_EXAMPLE_COUNT,
    SHIFT_MODES,
    SUITE_TASKS,
    TRANSITION_FAMILY_COUNT,
    TURN_COUNT,
    build_public_artifacts,
    compute_sha256,
    dsl_hypothesis_predictions,
    has_exact_label_balance,
    infer_context_terms,
    infer_cue_terms,
    learn_only_max_probe_accuracy,
    normalized_turn_text,
    parse_case_line,
)

EXPECTED_COUNTS: Final[dict[str, int]] = {
    "public": len(SUITE_TASKS) * PUBLIC_EPISODES_PER_TASK,
    "private": len(SUITE_TASKS) * PRIVATE_EPISODES_PER_TASK,
}

EXPECTED_TASK_COUNTS: Final[dict[str, int]] = {
    "public": PUBLIC_EPISODES_PER_TASK,
    "private": PRIVATE_EPISODES_PER_TASK,
}

PUBLIC_ATTACK_CEILINGS: Final[dict[str, float]] = {
    "majority_label_accuracy": 0.50,
    "previous_rule_accuracy": 0.25,
    "cue_agnostic_accuracy": 0.50,
}

PUBLIC_ATTACK_PER_TASK_CEILINGS: Final[dict[str, float]] = {
    "majority_label_accuracy": 0.50,
    "cue_agnostic_accuracy": 0.50,
}

MIN_PUBLIC_TRANSITION_PAIR_PATTERN_COUNT: Final[int] = TRANSITION_FAMILY_COUNT
MIN_PUBLIC_SELECTED_INITIAL_FAMILY_COUNT: Final[int] = 5
MIN_PUBLIC_SELECTED_SHIFT_FAMILY_COUNT: Final[int] = 6
PRIVATE_NEAR_DUPLICATE_OVERLAP_THRESHOLD: Final[float] = 0.95


def switching_prediction_set_ceiling(suite_task_id: str) -> int:
    return 4 if suite_task_id in {"explicit_rule_update", "latent_rule_update"} else 8


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


def normalize_labels(values: object) -> tuple[str, ...] | None:
    if not isinstance(values, (list, tuple)):
        return None
    normalized = tuple(str(value).strip().lower() for value in values)
    allowed = {"type_a", "type_b"}
    if len(normalized) != FINAL_PROBE_COUNT or any(label not in allowed for label in normalized):
        return None
    return normalized


def semantic_signature(row: dict[str, object]) -> tuple[object, ...]:
    targets = normalize_labels(row["scoring"]["final_probe_targets"])
    if targets is None:
        raise RuntimeError(f"row {row.get('episode_id')} has invalid final_probe_targets")
    normalized_turns = tuple(normalized_turn_text(turn) for turn in row["inference"]["turns"])
    return (row["analysis"]["suite_task_id"], normalized_turns, targets)


def _scored_episode_components(
    row: dict[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    learn_examples = _parse_examples(row["inference"]["turns"][0])
    shift_examples = _parse_examples(row["inference"]["turns"][1])
    probes = _parse_probes(row["inference"]["turns"][2])
    targets = normalize_labels(row["scoring"]["final_probe_targets"])
    if targets is None:
        raise RuntimeError(f"row {row.get('episode_id')} has invalid final_probe_targets")
    scored_probes: list[dict[str, object]] = []
    for probe, target in zip(probes, targets):
        scored_probe = dict(probe)
        scored_probe["target"] = target
        scored_probes.append(scored_probe)
    return learn_examples, shift_examples, scored_probes


def _normalized_nominal_mapping(
    phase_items: list[tuple[str, list[dict[str, object]]]],
    key: str,
    prefix: str,
) -> dict[str, str]:
    value_signatures: dict[str, list[tuple[object, ...]]] = {}
    for phase, items in phase_items:
        for item in items:
            raw_value = str(item.get(key, "")).strip()
            if not raw_value:
                continue
            value_signatures.setdefault(raw_value, []).append(
                (
                    phase,
                    int(item["r1"]),
                    int(item["r2"]),
                    str(item["shape"]),
                    str(item["tone"]),
                    str(item.get("label", item.get("target", ""))),
                )
            )
    ordered_values = sorted(
        value_signatures.items(),
        key=lambda item: (tuple(sorted(item[1])), item[0]),
    )
    return {value: f"{prefix}{index}" for index, (value, _signature) in enumerate(ordered_values)}


def _canonical_structural_case(
    item: dict[str, object],
    *,
    context_map: dict[str, str],
    cue_map: dict[str, str],
    target_key: str,
) -> tuple[object, ...]:
    return (
        int(item["r1"]),
        int(item["r2"]),
        str(item["shape"]),
        str(item["tone"]),
        context_map.get(str(item.get("context", "")).strip(), ""),
        cue_map.get(str(item.get("cue", "")).strip(), ""),
        str(item[target_key]),
    )


def structural_signature(row: dict[str, object]) -> tuple[object, ...]:
    learn_examples, shift_examples, scored_probes = _scored_episode_components(row)
    phase_items = [
        ("learn", learn_examples),
        ("shift", shift_examples),
        ("probe", scored_probes),
    ]
    context_map = _normalized_nominal_mapping(phase_items, "context", "context_")
    cue_map = _normalized_nominal_mapping(phase_items, "cue", "cue_")
    return (
        str(row["analysis"]["suite_task_id"]),
        tuple(
            sorted(
                _canonical_structural_case(
                    item,
                    context_map=context_map,
                    cue_map=cue_map,
                    target_key="label",
                )
                for item in learn_examples
            )
        ),
        tuple(
            sorted(
                _canonical_structural_case(
                    item,
                    context_map=context_map,
                    cue_map=cue_map,
                    target_key="label",
                )
                for item in shift_examples
            )
        ),
        tuple(
            sorted(
                _canonical_structural_case(
                    item,
                    context_map=context_map,
                    cue_map=cue_map,
                    target_key="target",
                )
                for item in scored_probes
            )
        ),
    )


def structural_case_counter(row: dict[str, object]) -> Counter[tuple[object, ...]]:
    learn_examples, shift_examples, scored_probes = _scored_episode_components(row)
    phase_items = [
        ("learn", learn_examples),
        ("shift", shift_examples),
        ("probe", scored_probes),
    ]
    context_map = _normalized_nominal_mapping(phase_items, "context", "context_")
    cue_map = _normalized_nominal_mapping(phase_items, "cue", "cue_")
    counter: Counter[tuple[object, ...]] = Counter()
    for phase, items, target_key in (
        ("learn", learn_examples, "label"),
        ("shift", shift_examples, "label"),
        ("probe", scored_probes, "target"),
    ):
        for item in items:
            counter[
                (
                    phase,
                    *_canonical_structural_case(
                        item,
                        context_map=context_map,
                        cue_map=cue_map,
                        target_key=target_key,
                    ),
                )
            ] += 1
    return counter


def structural_overlap_score(left: dict[str, object], right: dict[str, object]) -> float:
    left_counter = structural_case_counter(left)
    right_counter = structural_case_counter(right)
    total = max(sum(left_counter.values()), sum(right_counter.values()))
    if total == 0:
        return 0.0
    return sum((left_counter & right_counter).values()) / total


def verify_split_isolation(
    public_rows: list[dict[str, object]],
    private_rows: list[dict[str, object]],
) -> dict[str, object]:
    public_semantic = {semantic_signature(row): str(row["episode_id"]) for row in public_rows}
    public_structural: dict[tuple[object, ...], str] = {}
    public_by_task: dict[str, list[dict[str, object]]] = {}
    for row in public_rows:
        public_structural[structural_signature(row)] = str(row["episode_id"])
        public_by_task.setdefault(str(row["analysis"]["suite_task_id"]), []).append(row)

    exact_overlap: list[tuple[str, str]] = []
    structural_overlap: list[tuple[str, str]] = []
    near_duplicate_overlap: list[tuple[str, str, float]] = []

    for row in private_rows:
        private_episode_id = str(row["episode_id"])
        exact_public_id = public_semantic.get(semantic_signature(row))
        if exact_public_id is not None:
            exact_overlap.append((exact_public_id, private_episode_id))
            continue

        structural_public_id = public_structural.get(structural_signature(row))
        if structural_public_id is not None:
            structural_overlap.append((structural_public_id, private_episode_id))
            continue

        suite_task_id = str(row["analysis"]["suite_task_id"])
        best_score = -1.0
        best_public_id = ""
        for public_row in public_by_task[suite_task_id]:
            score = structural_overlap_score(public_row, row)
            if score > best_score:
                best_score = score
                best_public_id = str(public_row["episode_id"])
        if best_score >= PRIVATE_NEAR_DUPLICATE_OVERLAP_THRESHOLD:
            near_duplicate_overlap.append((best_public_id, private_episode_id, round(best_score, 4)))

    if exact_overlap:
        raise RuntimeError(f"public/private semantic overlap detected: {exact_overlap}")
    if structural_overlap:
        raise RuntimeError(f"public/private structural overlap detected: {structural_overlap}")
    if near_duplicate_overlap:
        raise RuntimeError(f"public/private near-duplicate overlap detected: {near_duplicate_overlap}")

    return {
        "semantic_overlap_count": 0,
        "structural_overlap_count": 0,
        "near_duplicate_overlap_count": 0,
    }


def switching_diagnostics_for_row(row: dict[str, object]) -> dict[str, object]:
    learn_examples = _parse_examples(row["inference"]["turns"][0])
    shift_examples = _parse_examples(row["inference"]["turns"][1])
    probes = _parse_probes(row["inference"]["turns"][2])
    targets = normalize_labels(row["scoring"]["final_probe_targets"])
    if targets is None:
        raise RuntimeError(f"row {row.get('episode_id')} has invalid final_probe_targets")
    suite_task_id = str(row["analysis"]["suite_task_id"])
    context_terms = infer_context_terms(learn_examples, shift_examples, probes)
    cue_terms = infer_cue_terms(learn_examples, shift_examples, probes)
    hypotheses = dsl_hypothesis_predictions(
        suite_task_id,
        learn_examples,
        shift_examples,
        probes,
        context_terms=context_terms,
        cue_terms=cue_terms,
    )
    return {
        "learn_only_max_probe_accuracy": learn_only_max_probe_accuracy(learn_examples, probes, targets),
        "post_shift_prediction_set_size": len(set(hypotheses)),
    }


def verify_switching_requirements(rows: list[dict[str, object]]) -> dict[str, object]:
    learn_only_values: list[float] = []
    prediction_set_sizes: Counter[int] = Counter()
    for row in rows:
        diagnostics = switching_diagnostics_for_row(row)
        suite_task_id = str(row["analysis"]["suite_task_id"])
        learn_only_ceiling = 0.50 if suite_task_id in {"explicit_rule_update", "latent_rule_update"} else 0.625
        learn_only_values.append(float(diagnostics["learn_only_max_probe_accuracy"]))
        prediction_set_size = int(diagnostics["post_shift_prediction_set_size"])
        prediction_set_sizes[prediction_set_size] += 1
        if float(diagnostics["learn_only_max_probe_accuracy"]) > learn_only_ceiling:
            raise RuntimeError(f"row {row['episode_id']} is too solvable from turn 1 alone")
        if prediction_set_size > switching_prediction_set_ceiling(suite_task_id):
            raise RuntimeError(f"row {row['episode_id']} leaves turn-3 predictions too ambiguous after turn 2")
    return {
        "learn_only_max_probe_accuracy": round(sum(learn_only_values) / len(learn_only_values), 4),
        "post_shift_prediction_set_size_distribution": dict(sorted(prediction_set_sizes.items())),
    }


def verify_schema(rows: list[dict[str, object]], split: str) -> dict[str, object]:
    expected_task_count = EXPECTED_TASK_COUNTS[split]
    expected_row_count = EXPECTED_COUNTS[split]
    if len(rows) != expected_row_count:
        raise RuntimeError(f"{split} split expected {expected_row_count} rows, found {len(rows)}")

    task_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()

    for row in rows:
        expected_keys = ["analysis", "episode_id", "inference", "scoring"] if split == "public" else [
            "analysis",
            "episode_id",
            "inference",
        ]
        if sorted(row.keys()) != expected_keys:
            raise RuntimeError(f"row {row.get('episode_id')} has inconsistent top-level keys")

        analysis = row.get("analysis")
        inference = row.get("inference")
        episode_id = row.get("episode_id")
        if not isinstance(analysis, dict) or not isinstance(inference, dict):
            raise RuntimeError(f"row {episode_id} is missing required objects")
        if sorted(analysis.keys()) != ["difficulty_bin", "faculty_id", "shift_mode", "suite_task_id"]:
            raise RuntimeError(f"row {episode_id} has inconsistent analysis keys")
        if analysis["faculty_id"] != FACULTY_ID:
            raise RuntimeError(f"row {episode_id} has unsupported faculty_id {analysis['faculty_id']!r}")
        suite_task_id = str(analysis["suite_task_id"])
        if suite_task_id not in SUITE_TASKS:
            raise RuntimeError(f"row {episode_id} has unsupported suite_task_id {suite_task_id!r}")
        if analysis["shift_mode"] != SHIFT_MODES[suite_task_id]:
            raise RuntimeError(f"row {episode_id} has mismatched shift_mode")
        if analysis["difficulty_bin"] not in {"hard", "medium"}:
            raise RuntimeError(f"row {episode_id} has unsupported difficulty_bin {analysis['difficulty_bin']!r}")

        turns = inference.get("turns")
        if not isinstance(turns, list) or len(turns) != TURN_COUNT:
            raise RuntimeError(f"row {episode_id} must expose exactly {TURN_COUNT} turns")
        for turn_index, turn in enumerate(turns, start=1):
            prefix = f"CogFlex suite task. Episode {episode_id}. Turn {turn_index} of {TURN_COUNT}."
            if not isinstance(turn, str) or not turn.startswith(prefix):
                raise RuntimeError(f"row {episode_id} has malformed turn {turn_index}")

        learn_examples = _parse_examples(turns[0])
        shift_examples = _parse_examples(turns[1])
        probes = _parse_probes(turns[2])
        if len(learn_examples) != LEARN_EXAMPLE_COUNT:
            raise RuntimeError(f"row {episode_id} learn turn must contain {LEARN_EXAMPLE_COUNT} examples")
        if len(shift_examples) != SHIFT_EXAMPLE_COUNT:
            raise RuntimeError(f"row {episode_id} shift turn must contain {SHIFT_EXAMPLE_COUNT} examples")
        if len(probes) != FINAL_PROBE_COUNT:
            raise RuntimeError(f"row {episode_id} decision turn must contain {FINAL_PROBE_COUNT} probes")
        if not has_exact_label_balance((str(item["label"]) for item in learn_examples), true_count=3, false_count=3):
            raise RuntimeError(f"row {episode_id} learn turn must stay label-balanced")
        if not has_exact_label_balance((str(item["label"]) for item in shift_examples), true_count=3, false_count=3):
            raise RuntimeError(f"row {episode_id} shift turn must stay label-balanced")
        if FINAL_OUTPUT_INSTRUCTION not in turns[2]:
            raise RuntimeError(f"row {episode_id} decision turn must use the fixed output instruction")
        for item in learn_examples + shift_examples + probes:
            for key in ("shape", "tone"):
                if key not in item or not str(item[key]).strip():
                    raise RuntimeError(f"row {episode_id} is missing required stimulus attribute {key!r}")

        if split == "public":
            scoring = row.get("scoring")
            if not isinstance(scoring, dict) or sorted(scoring.keys()) != ["final_probe_targets"]:
                raise RuntimeError(f"row {episode_id} has inconsistent scoring keys")
            targets = normalize_labels(scoring["final_probe_targets"])
            if targets is None:
                raise RuntimeError(f"row {episode_id} has invalid final_probe_targets")
            if not has_exact_label_balance(targets, true_count=4, false_count=4):
                raise RuntimeError(f"row {episode_id} final probes must stay label-balanced")
            label_counts.update(targets)
        elif "scoring" in row:
            raise RuntimeError(f"row {episode_id} leaks scoring fields in the private split")

        task_counts[suite_task_id] += 1
        difficulty_counts[str(analysis["difficulty_bin"])] += 1

    if task_counts != Counter({suite_task_id: expected_task_count for suite_task_id in SUITE_TASKS}):
        raise RuntimeError(f"{split} split task counts mismatch: {task_counts}")

    return {
        "row_count": len(rows),
        "suite_task_counts": dict(sorted(task_counts.items())),
        "difficulty_bin_counts": dict(sorted(difficulty_counts.items())),
        "label_counts": dict(sorted(label_counts.items())) if label_counts else {},
    }


def load_rows(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError(f"expected a JSON list at {path}")
    return payload


def public_quality_report_payload() -> dict[str, object]:
    payload = json.loads(PUBLIC_QUALITY_REPORT_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("public quality report must be a JSON object")
    return payload


def verify_public_attack_suite(payload: dict[str, object]) -> dict[str, object]:
    attack_suite = payload.get("attack_suite")
    if not isinstance(attack_suite, dict):
        raise RuntimeError("public quality report must include attack_suite")
    for metric, ceiling in PUBLIC_ATTACK_CEILINGS.items():
        summary = attack_suite.get(metric)
        if not isinstance(summary, dict):
            raise RuntimeError(f"public quality report must include {metric}")
        micro_accuracy = summary.get("micro_accuracy")
        if not isinstance(micro_accuracy, (int, float)) or float(micro_accuracy) > ceiling:
            raise RuntimeError(f"public quality report {metric} micro_accuracy exceeds ceiling")
        per_task = summary.get("per_task_accuracy")
        if not isinstance(per_task, dict):
            raise RuntimeError(f"public quality report {metric} must include per_task_accuracy")
        per_task_ceiling = PUBLIC_ATTACK_PER_TASK_CEILINGS.get(metric)
        for suite_task_id, value in per_task.items():
            if value is None:
                continue
            if suite_task_id not in SUITE_TASKS:
                raise RuntimeError(f"public quality report {metric} has unknown task {suite_task_id!r}")
            if per_task_ceiling is None:
                continue
            if not isinstance(value, (int, float)) or float(value) > per_task_ceiling:
                raise RuntimeError(f"public quality report {metric} per-task accuracy exceeds ceiling")
    return attack_suite


def verify_public_diversity_summary(payload: dict[str, object]) -> dict[str, object]:
    if payload.get("rule_inventory_count") != len(PUBLIC_RULES):
        raise RuntimeError("public quality report rule_inventory_count mismatch")
    if payload.get("rule_family_count") != len(PUBLIC_FAMILY_IDS):
        raise RuntimeError("public quality report rule_family_count mismatch")
    family_counts = payload.get("rule_family_rule_counts")
    expected_family_counts = Counter(rule.family_id for rule in PUBLIC_RULES)
    if family_counts != dict(sorted(expected_family_counts.items())):
        raise RuntimeError("public quality report rule_family_rule_counts mismatch")
    if payload.get("transition_family_count") != TRANSITION_FAMILY_COUNT:
        raise RuntimeError("public quality report transition_family_count mismatch")
    pair_pattern_count = payload.get("transition_pair_pattern_count")
    if not isinstance(pair_pattern_count, int) or pair_pattern_count < MIN_PUBLIC_TRANSITION_PAIR_PATTERN_COUNT:
        raise RuntimeError("public quality report transition_pair_pattern_count is too low")
    pair_usage = payload.get("transition_pair_usage")
    if not isinstance(pair_usage, dict) or len(pair_usage) != pair_pattern_count:
        raise RuntimeError("public quality report transition_pair_usage mismatch")
    initial_family_count = payload.get("selected_initial_family_count")
    if not isinstance(initial_family_count, int) or initial_family_count < MIN_PUBLIC_SELECTED_INITIAL_FAMILY_COUNT:
        raise RuntimeError("public quality report selected_initial_family_count is too low")
    initial_family_usage = payload.get("selected_initial_family_usage")
    if not isinstance(initial_family_usage, dict) or len(initial_family_usage) != initial_family_count:
        raise RuntimeError("public quality report selected_initial_family_usage mismatch")
    shift_family_count = payload.get("selected_shift_family_count")
    if not isinstance(shift_family_count, int) or shift_family_count < MIN_PUBLIC_SELECTED_SHIFT_FAMILY_COUNT:
        raise RuntimeError("public quality report selected_shift_family_count is too low")
    shift_family_usage = payload.get("selected_shift_family_usage")
    if not isinstance(shift_family_usage, dict) or len(shift_family_usage) != shift_family_count:
        raise RuntimeError("public quality report selected_shift_family_usage mismatch")
    return {
        "rule_inventory_count": payload["rule_inventory_count"],
        "rule_family_count": payload["rule_family_count"],
        "selected_initial_family_count": initial_family_count,
        "selected_shift_family_count": shift_family_count,
        "transition_pair_pattern_count": pair_pattern_count,
    }


def verify_public_switching_summary(payload: dict[str, object]) -> dict[str, object]:
    switching = payload.get("switching_diagnostics")
    if not isinstance(switching, dict):
        raise RuntimeError("public quality report must include switching_diagnostics")
    learn_only_value = switching.get("learn_only_max_probe_accuracy")
    if not isinstance(learn_only_value, (int, float)) or float(learn_only_value) > 0.625:
        raise RuntimeError("public quality report learn_only_max_probe_accuracy exceeds ceiling")
    set_sizes = switching.get("post_shift_prediction_set_size_distribution")
    if not isinstance(set_sizes, dict):
        raise RuntimeError("public quality report must include post_shift_prediction_set_size_distribution")
    normalized_set_sizes = {int(key): int(value) for key, value in set_sizes.items()}
    if sum(normalized_set_sizes.values()) != EXPECTED_COUNTS["public"] or max(normalized_set_sizes, default=0) > 8:
        raise RuntimeError("public quality report post_shift_prediction_set_size_distribution mismatch")
    turn2_distribution = switching.get("turn2_required_probe_count_distribution")
    if not isinstance(turn2_distribution, dict):
        raise RuntimeError("public quality report must include turn2_required_probe_count_distribution")
    return {
        "learn_only_max_probe_accuracy": round(float(learn_only_value), 4),
        "post_shift_prediction_set_size_distribution": dict(sorted(normalized_set_sizes.items())),
    }


def verify_public_split() -> None:
    rows = load_rows(PUBLIC_ROWS_PATH)
    schema_summary = verify_schema(rows, "public")
    expected_rows, _answers, report = build_public_artifacts()
    if rows != expected_rows:
        raise RuntimeError("public split rows are not reproducible from the generator")
    tracked_report = public_quality_report_payload()
    if tracked_report != report:
        raise RuntimeError("public quality report is not reproducible from the generator")
    attack_suite = verify_public_attack_suite(tracked_report)
    diversity_summary = verify_public_diversity_summary(tracked_report)
    switching_summary = verify_public_switching_summary(tracked_report)
    row_switching_summary = verify_switching_requirements(rows)
    print(
        json.dumps(
            {
                "split": "public",
                "rows_path": str(PUBLIC_ROWS_PATH),
                "quality_report_path": str(PUBLIC_QUALITY_REPORT_PATH),
                **schema_summary,
                "attack_suite": attack_suite,
                **diversity_summary,
                "switching_diagnostics": switching_summary,
                "row_switching_diagnostics": row_switching_summary,
                "transition_family_count": tracked_report["transition_family_count"],
                "disagreement_bin_counts": tracked_report["disagreement_bin_counts"],
            },
            indent=2,
        )
    )


def private_bundle_paths(bundle_dir: Path) -> dict[str, Path]:
    return {
        "rows": bundle_dir / PRIVATE_ROWS_FILENAME,
        "answer_key": bundle_dir / PRIVATE_ANSWER_KEY_FILENAME,
        "manifest": bundle_dir / PRIVATE_RELEASE_MANIFEST_FILENAME,
        "quality": bundle_dir / PRIVATE_QUALITY_REPORT_FILENAME,
    }


def resolve_private_bundle_dir(explicit_path: str | None) -> Path:
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
    else:
        env_value = Path.cwd().joinpath("")  # placeholder for type narrowing
        env_raw = None
        import os

        env_raw = os.environ.get(PRIVATE_BUNDLE_ENV_VAR)
        if not env_raw:
            raise RuntimeError(
                f"private verification requires --private-bundle-dir or {PRIVATE_BUNDLE_ENV_VAR}"
            )
        path = Path(env_raw).expanduser().resolve()
    if not path.exists() or not path.is_dir():
        raise RuntimeError(f"private bundle directory does not exist: {path}")
    return path


def load_private_answer_key(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("private answer key payload must be a JSON object")
    if payload.get("version") != "cogflex_private_answer_key_v1":
        raise RuntimeError("private answer key has an unsupported version")
    if payload.get("split") != "private":
        raise RuntimeError("private answer key must declare split='private'")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list):
        raise RuntimeError("private answer key must expose an episodes list")
    return payload


def verify_private_answer_key(
    payload: dict[str, object],
    private_rows: list[dict[str, object]],
) -> tuple[dict[str, object], dict[str, tuple[str, ...]]]:
    rows_by_id = {str(row["episode_id"]): row for row in private_rows}
    episode_targets: dict[str, tuple[str, ...]] = {}
    label_counts: Counter[str] = Counter()
    episodes = payload["episodes"]

    for answer in episodes:
        if not isinstance(answer, dict):
            raise RuntimeError("private answer key episodes must be JSON objects")
        episode_id = str(answer.get("episode_id"))
        if episode_id in episode_targets:
            raise RuntimeError(f"private answer key duplicates episode_id {episode_id}")
        row = rows_by_id.get(episode_id)
        if row is None:
            raise RuntimeError(f"private answer key contains unknown episode_id {episode_id}")
        targets = normalize_labels(answer.get("final_probe_targets"))
        if targets is None:
            raise RuntimeError(f"private answer key episode {episode_id} has invalid final_probe_targets")
        for key in ("faculty_id", "suite_task_id", "shift_mode", "difficulty_bin"):
            if answer.get(key) != row["analysis"][key]:
                raise RuntimeError(f"private answer key {key} mismatch for episode {episode_id}")
        turns = answer.get("turns")
        if turns != row["inference"]["turns"]:
            raise RuntimeError(f"private answer key turns mismatch for episode {episode_id}")
        episode_targets[episode_id] = targets
        label_counts.update(targets)

    missing_ids = set(rows_by_id) - set(episode_targets)
    if missing_ids:
        raise RuntimeError(f"private answer key is missing episode_ids: {sorted(missing_ids)}")

    return {
        "answer_key_episode_count": len(episode_targets),
        "answer_key_label_counts": dict(sorted(label_counts.items())),
    }, episode_targets


def attach_private_scoring(
    private_rows: list[dict[str, object]],
    answer_key: dict[str, object],
) -> list[dict[str, object]]:
    _summary, episode_targets = verify_private_answer_key(answer_key, private_rows)
    scored_rows: list[dict[str, object]] = []
    for row in private_rows:
        episode_id = str(row["episode_id"])
        scored_rows.append(
            {
                "episode_id": row["episode_id"],
                "inference": {"turns": list(row["inference"]["turns"])},
                "analysis": dict(row["analysis"]),
                "scoring": {"final_probe_targets": list(episode_targets[episode_id])},
            }
        )
    return scored_rows


def verify_manifest(path: Path, bundle_paths: dict[str, Path]) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("private release manifest must be a JSON object")
    if payload.get("version") != PRIVATE_BUNDLE_VERSION:
        raise RuntimeError("private release manifest has an unsupported version")
    if payload.get("split") != "private":
        raise RuntimeError("private release manifest must declare split='private'")
    if payload.get("row_count") != EXPECTED_COUNTS["private"]:
        raise RuntimeError("private release manifest has an invalid row_count")
    if payload.get("episodes_per_task") != PRIVATE_EPISODES_PER_TASK:
        raise RuntimeError("private release manifest has an invalid episodes_per_task")
    digests = payload.get("sha256")
    if not isinstance(digests, dict):
        raise RuntimeError("private release manifest must expose sha256 digests")
    for key, file_path in bundle_paths.items():
        filename = file_path.name
        if filename == PRIVATE_RELEASE_MANIFEST_FILENAME:
            continue
        declared = digests.get(filename)
        if not isinstance(declared, str) or len(declared) != 64:
            raise RuntimeError(f"private release manifest is missing digest for {filename}")
        actual = compute_sha256(file_path)
        if actual != declared:
            raise RuntimeError(f"private release manifest digest mismatch for {filename}")

    lexicons = payload.get("lexicons")
    if not isinstance(lexicons, dict):
        raise RuntimeError("private release manifest must include lexicons")
    cue_terms = lexicons.get("cue_terms")
    context_terms = lexicons.get("context_terms")
    if not isinstance(cue_terms, list) or not isinstance(context_terms, list):
        raise RuntimeError("private release manifest lexicons must include cue_terms and context_terms lists")

    public_terms = {
        term
        for lexicon in PUBLIC_CUE_LEXICONS + PUBLIC_CONTEXT_LEXICONS
        for term in lexicon
    }
    overlap = sorted({str(term) for term in cue_terms + context_terms if str(term) in public_terms})
    if overlap:
        raise RuntimeError(f"private bundle lexicon overlap with public split: {overlap}")

    return payload


def verify_quality_report(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("private quality report must be a JSON object")
    if payload.get("version") != PRIVATE_QUALITY_REPORT_VERSION:
        raise RuntimeError("private quality report has an unsupported version")
    if payload.get("split") != "private":
        raise RuntimeError("private quality report must declare split='private'")
    if payload.get("row_count") != EXPECTED_COUNTS["private"]:
        raise RuntimeError("private quality report row_count mismatch")
    if payload.get("episodes_per_task") != PRIVATE_EPISODES_PER_TASK:
        raise RuntimeError("private quality report episodes_per_task mismatch")
    difficulty_counts = payload.get("difficulty_bin_counts")
    if difficulty_counts != {"hard": EXPECTED_COUNTS["private"] // 2, "medium": EXPECTED_COUNTS["private"] // 2}:
        raise RuntimeError("private quality report difficulty_bin_counts mismatch")

    calibration_summary = payload.get("calibration_summary")
    if not isinstance(calibration_summary, dict):
        raise RuntimeError("private quality report must include calibration_summary")
    models = calibration_summary.get("models")
    if not isinstance(models, list) or len(models) != 3:
        raise RuntimeError("private quality report calibration_summary must include exactly 3 models")
    for model in models:
        if not isinstance(model, dict):
            raise RuntimeError("private quality report model entries must be objects")
        for key in ("name", "macro_accuracy", "micro_accuracy"):
            if key not in model:
                raise RuntimeError(f"private quality report model is missing {key}")

    attack_suite = payload.get("attack_suite")
    if not isinstance(attack_suite, dict) or not attack_suite:
        raise RuntimeError("private quality report must include attack_suite")
    semantic = payload.get("semantic_isolation_summary")
    if not isinstance(semantic, dict):
        raise RuntimeError("private quality report must include semantic_isolation_summary")
    return payload


def verify_private_bundle(bundle_dir: Path) -> None:
    bundle_paths = private_bundle_paths(bundle_dir)
    missing = [str(path) for path in bundle_paths.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"private bundle is missing required files: {missing}")

    private_rows = load_rows(bundle_paths["rows"])
    schema_summary = verify_schema(private_rows, "private")
    answer_key = load_private_answer_key(bundle_paths["answer_key"])
    answer_key_summary, _targets = verify_private_answer_key(answer_key, private_rows)
    scored_rows = attach_private_scoring(private_rows, answer_key)
    manifest = verify_manifest(bundle_paths["manifest"], bundle_paths)
    quality_report = verify_quality_report(bundle_paths["quality"])

    public_rows = load_rows(PUBLIC_ROWS_PATH)
    isolation_summary = verify_split_isolation(public_rows, scored_rows)
    switching_summary = verify_switching_requirements(scored_rows)

    print(
        json.dumps(
            {
                "split": "private",
                "bundle_dir": str(bundle_dir),
                **schema_summary,
                **answer_key_summary,
                "manifest_version": manifest["version"],
                "quality_report_version": quality_report["version"],
                "calibration_models": [model["name"] for model in quality_report["calibration_summary"]["models"]],
                **isolation_summary,
                "switching_diagnostics": switching_summary,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the CogFlex benchmark release artifacts.")
    parser.add_argument("--split", choices=("public", "private"), default="public")
    parser.add_argument("--private-bundle-dir", default=None)
    args = parser.parse_args()
    if args.split == "public":
        verify_public_split()
    else:
        verify_private_bundle(resolve_private_bundle_dir(args.private_bundle_dir))


if __name__ == "__main__":
    main()
