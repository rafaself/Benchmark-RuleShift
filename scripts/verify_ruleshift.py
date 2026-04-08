#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Final

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.build_ruleshift_dataset import (  # noqa: E402
    FACULTY_ID,
    FINAL_OUTPUT_INSTRUCTION,
    FINAL_PROBE_COUNT,
    PRIVATE_ANSWER_KEY_PATH,
    PRIVATE_MANIFEST_PATH,
    PRIVATE_ROWS_PATH,
    PUBLIC_ROWS_PATH,
    SHIFT_MODES,
    SUITE_TASKS,
    build_private_artifacts,
    build_split,
    load_private_manifest,
    private_answer_key_payload,
    sanitize_private_rows,
    validate_split_isolation,
)

EXPECTED_COUNTS: Final[dict[str, int]] = {"public": 80, "private": 400}
EXPECTED_TASK_COUNTS: Final[dict[str, int]] = {"public": 20, "private": 100}
TURN_COUNT: Final[int] = 3
LINE_RE = re.compile(
    r"^(?P<index>\d+)\.\s+(?P<body>.+?)\s+->\s+(?P<label>type_a|type_b|\?)$"
)
POINT_RE = re.compile(r"^r1=(?P<r1>[+-]\d+),\s*r2=(?P<r2>[+-]\d+)$")


def detect_legacy_rows(rows: list[dict[str, object]]) -> str | None:
    if not rows:
        return None
    row = rows[0]
    inference = row.get("inference")
    analysis = row.get("analysis")
    if not isinstance(inference, dict) or not isinstance(analysis, dict):
        return None
    if "prompt" in inference or "group_id" in analysis or "initial_rule_id" in analysis:
        return (
            "detected legacy RuleShift artifacts. "
            "Regenerate benchmark assets with scripts/build_ruleshift_dataset.py."
        )
    return None


def dataset_path(split: str) -> Path:
    return PUBLIC_ROWS_PATH if split == "public" else PRIVATE_ROWS_PATH


def load_rows(split: str) -> list[dict[str, object]]:
    rows = json.loads(dataset_path(split).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise RuntimeError(f"{split} dataset payload must be a list")
    return rows


def load_private_answer_key(path: Path | None = None) -> dict[str, object]:
    path = PRIVATE_ANSWER_KEY_PATH if path is None else path
    if not path.exists():
        raise RuntimeError(
            f"private split requires an answer key at {path}. "
            "Run scripts/build_ruleshift_dataset.py after creating the private split manifest."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("private answer key payload must be a JSON object")
    return payload


def normalize_labels(values: object) -> tuple[str, ...] | None:
    if not isinstance(values, (list, tuple)):
        return None
    normalized = tuple(str(value).strip().lower() for value in values)
    allowed = {"type_a", "type_b"}
    if len(normalized) != FINAL_PROBE_COUNT or any(label not in allowed for label in normalized):
        return None
    return normalized


def score_episode(predictions: object, targets: object) -> tuple[int, int]:
    norm_targets = normalize_labels(targets)
    if norm_targets is None:
        raise ValueError("targets must contain exactly four valid labels")
    norm_predictions = normalize_labels(predictions)
    if norm_predictions is None:
        return (0, FINAL_PROBE_COUNT)
    return (sum(pred == target for pred, target in zip(norm_predictions, norm_targets)), FINAL_PROBE_COUNT)


def parse_case_line(line: str) -> dict[str, object] | None:
    match = LINE_RE.match(line.strip())
    if match is None:
        return None
    body = match.group("body")
    attributes: dict[str, object] = {"index": int(match.group("index")), "label": match.group("label")}
    point: tuple[int, int] | None = None
    for chunk in (part.strip() for part in body.split("|")):
        point_match = POINT_RE.match(chunk)
        if point_match is not None:
            point = (int(point_match.group("r1")), int(point_match.group("r2")))
            continue
        if "=" not in chunk:
            raise RuntimeError(f"Malformed attribute chunk: {chunk!r}")
        key, value = chunk.split("=", 1)
        attributes[key.strip()] = value.strip()
    if point is None:
        raise RuntimeError(f"Missing coordinates in line: {line!r}")
    attributes["r1"] = point[0]
    attributes["r2"] = point[1]
    return attributes


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


def verify_schema(rows: list[dict[str, object]], split: str) -> dict[str, object]:
    expected_task_count = EXPECTED_TASK_COUNTS[split]
    if len(rows) != EXPECTED_COUNTS[split]:
        raise RuntimeError(f"{split} split expected {EXPECTED_COUNTS[split]} rows, found {len(rows)}")

    task_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()

    for row in rows:
        expected_row_keys = ["analysis", "episode_id", "inference", "scoring"]
        if split == "private":
            expected_row_keys = ["analysis", "episode_id", "inference"]
        if sorted(row.keys()) != expected_row_keys:
            raise RuntimeError(f"row {row.get('episode_id')} has inconsistent top-level keys")

        episode_id = row["episode_id"]
        analysis = row["analysis"]
        inference = row["inference"]
        expected_analysis_keys = ["difficulty_bin", "faculty_id", "shift_mode", "suite_task_id"]
        if not isinstance(analysis, dict) or sorted(analysis.keys()) != expected_analysis_keys:
            raise RuntimeError(f"row {episode_id} has inconsistent analysis keys")
        if analysis["faculty_id"] != FACULTY_ID:
            raise RuntimeError(f"row {episode_id} exposes unsupported faculty_id {analysis['faculty_id']!r}")
        suite_task_id = str(analysis["suite_task_id"])
        if suite_task_id not in SUITE_TASKS:
            raise RuntimeError(f"row {episode_id} exposes unsupported suite_task_id {suite_task_id!r}")
        if analysis["shift_mode"] != SHIFT_MODES[suite_task_id]:
            raise RuntimeError(f"row {episode_id} has inconsistent shift_mode for {suite_task_id!r}")
        if analysis["difficulty_bin"] not in {"hard", "medium"}:
            raise RuntimeError(f"row {episode_id} exposes unsupported difficulty_bin {analysis['difficulty_bin']!r}")

        if not isinstance(inference, dict) or sorted(inference.keys()) != ["turns"]:
            raise RuntimeError(f"row {episode_id} has inconsistent inference keys")
        turns = inference["turns"]
        if not isinstance(turns, list) or len(turns) != TURN_COUNT:
            raise RuntimeError(f"row {episode_id} must expose exactly {TURN_COUNT} turns")
        for turn_index, turn in enumerate(turns, start=1):
            prefix = f"CogFlex suite task. Episode {episode_id}. Turn {turn_index} of 3."
            if not isinstance(turn, str) or not turn.startswith(prefix):
                raise RuntimeError(f"row {episode_id} has malformed turn {turn_index}")
        if FINAL_OUTPUT_INSTRUCTION not in turns[2]:
            raise RuntimeError(f"row {episode_id} decision turn has inconsistent output instruction")

        learn_examples = _parse_examples(turns[0])
        shift_examples = _parse_examples(turns[1])
        probes = _parse_probes(turns[2])
        if len(learn_examples) != 4 or len(shift_examples) != 4 or len(probes) != FINAL_PROBE_COUNT:
            raise RuntimeError(f"row {episode_id} has malformed example/probe counts")
        for item in learn_examples + shift_examples + probes:
            for key in ("shape", "tone"):
                if key not in item or not str(item[key]).strip():
                    raise RuntimeError(f"row {episode_id} is missing required stimulus attribute {key!r}")

        if split == "public":
            scoring = row["scoring"]
            if not isinstance(scoring, dict) or sorted(scoring.keys()) != ["final_probe_targets"]:
                raise RuntimeError(f"row {episode_id} has inconsistent scoring keys")
            targets = normalize_labels(scoring["final_probe_targets"])
            if targets is None:
                raise RuntimeError(f"row {episode_id} has invalid final_probe_targets")
            label_counts.update(targets)
        elif "scoring" in row:
            raise RuntimeError(f"row {episode_id} leaks scoring fields in the private split")

        task_counts[suite_task_id] += 1
        difficulty_counts[str(analysis["difficulty_bin"])] += 1

    if set(task_counts) != set(SUITE_TASKS):
        raise RuntimeError(f"{split} split suite tasks mismatch: {sorted(task_counts)}")
    mismatched_counts = {task: count for task, count in task_counts.items() if count != expected_task_count}
    if mismatched_counts:
        raise RuntimeError(f"{split} split task counts mismatch: {mismatched_counts}")

    return {
        "row_count": len(rows),
        "suite_task_counts": dict(sorted(task_counts.items())),
        "difficulty_bin_counts": dict(sorted(difficulty_counts.items())),
        "label_counts": dict(sorted(label_counts.items())) if label_counts else {},
    }


def verify_private_answer_key(
    payload: dict[str, object],
    private_rows: list[dict[str, object]],
) -> tuple[dict[str, object], dict[str, tuple[str, ...]], list[dict[str, object]]]:
    if payload.get("version") != "cogflex_suite_v1":
        raise RuntimeError("private answer key has an unsupported version")
    if payload.get("split") != "private":
        raise RuntimeError("private answer key must declare split='private'")

    episodes = payload.get("episodes")
    if not isinstance(episodes, list):
        raise RuntimeError("private answer key must expose an episodes list")

    rows_by_id = {str(row["episode_id"]): row for row in private_rows}
    episode_targets: dict[str, tuple[str, ...]] = {}
    label_counts: Counter[str] = Counter()
    answers: list[dict[str, object]] = []

    for answer in episodes:
        if not isinstance(answer, dict):
            raise RuntimeError("private answer key episodes must be objects")
        episode_id = str(answer.get("episode_id"))
        if episode_id in episode_targets:
            raise RuntimeError(f"private answer key duplicates episode_id {episode_id}")
        row = rows_by_id.get(episode_id)
        if row is None:
            raise RuntimeError(f"private answer key contains unknown episode_id {episode_id}")

        targets = normalize_labels(answer.get("final_probe_targets"))
        if targets is None:
            raise RuntimeError(f"private answer key episode {episode_id} has invalid final_probe_targets")

        required = (
            "faculty_id",
            "suite_task_id",
            "shift_mode",
            "difficulty_bin",
            "transition_family_id",
            "initial_rule_id",
            "shift_rule_id",
            "initial_rule_family_id",
            "shift_rule_family_id",
            "initial_rule_template_id",
            "shift_rule_template_id",
            "cue_template_id",
            "context_template_id",
            "surface_template_id",
            "learn_turn_examples",
            "shift_turn_examples",
            "final_probes",
            "turns",
            "generator_diagnostics",
        )
        for key in required:
            if key not in answer:
                raise RuntimeError(f"private answer key episode {episode_id} is missing {key}")

        if answer["turns"] != row["inference"]["turns"]:
            raise RuntimeError(f"private answer key turns mismatch for episode {episode_id}")
        for key in ("faculty_id", "suite_task_id", "shift_mode", "difficulty_bin"):
            if answer[key] != row["analysis"][key]:
                raise RuntimeError(f"private answer key {key} mismatch for episode {episode_id}")

        answers.append(answer)
        episode_targets[episode_id] = targets
        label_counts.update(targets)

    missing_ids = set(rows_by_id) - set(episode_targets)
    if missing_ids:
        raise RuntimeError(f"private answer key is missing episode_ids: {sorted(missing_ids)}")

    return (
        {
            "answer_key_episode_count": len(episodes),
            "answer_key_label_counts": dict(sorted(label_counts.items())),
        },
        episode_targets,
        answers,
    )


def attach_private_scoring(
    private_rows: list[dict[str, object]],
    payload: dict[str, object],
) -> list[dict[str, object]]:
    _summary, episode_targets, _answers = verify_private_answer_key(payload, private_rows)
    enriched_rows: list[dict[str, object]] = []
    for row in private_rows:
        episode_id = str(row["episode_id"])
        enriched_rows.append(
            {
                "episode_id": row["episode_id"],
                "inference": {"turns": list(row["inference"]["turns"])},
                "scoring": {"final_probe_targets": episode_targets[episode_id]},
                "analysis": dict(row["analysis"]),
            }
        )
    return enriched_rows


def semantic_signature(row: dict[str, object]) -> tuple[object, ...]:
    targets = normalize_labels(row["scoring"]["final_probe_targets"])
    if targets is None:
        raise RuntimeError(f"row {row.get('episode_id')} has invalid final_probe_targets")
    return (row["analysis"]["suite_task_id"], tuple(row["inference"]["turns"]), targets)


def verify_split_isolation(
    public_rows: list[dict[str, object]],
    private_rows: list[dict[str, object]],
) -> None:
    public_signatures = {semantic_signature(row): str(row["episode_id"]) for row in public_rows}
    for row in private_rows:
        signature = semantic_signature(row)
        public_episode_id = public_signatures.get(signature)
        if public_episode_id is not None:
            raise RuntimeError(
                "public/private split isolation violated: "
                f"public episode {public_episode_id} overlaps private episode {row['episode_id']}"
            )


def summarize_generator_diagnostics(answers: list[dict[str, object]]) -> dict[str, object]:
    task_buckets: dict[str, list[dict[str, object]]] = defaultdict(list)
    label_counts: Counter[str] = Counter()
    for answer in answers:
        task_buckets[str(answer["suite_task_id"])].append(answer)
        label_counts.update(answer["final_probe_targets"])

    def summarize_metric(metric: str) -> dict[str, object]:
        per_task: dict[str, float | None] = {}
        weighted_total = 0.0
        weighted_denominator = 0
        for suite_task_id in SUITE_TASKS:
            values = [
                answer["generator_diagnostics"][metric]
                for answer in task_buckets[suite_task_id]
                if answer["generator_diagnostics"][metric] is not None
            ]
            if not values:
                per_task[suite_task_id] = None
                continue
            mean_value = sum(float(value) for value in values) / len(values)
            per_task[suite_task_id] = round(mean_value, 4)
            weighted_total += sum(float(value) for value in values)
            weighted_denominator += len(values)
        micro = None if weighted_denominator == 0 else round(weighted_total / weighted_denominator, 4)
        return {"micro_accuracy": micro, "per_task_accuracy": per_task}

    unanimous_counts = {
        suite_task_id: sum(
            bool(answer["generator_diagnostics"]["symbolic_unanimous_predictions"])
            for answer in task_buckets[suite_task_id]
        )
        for suite_task_id in SUITE_TASKS
    }
    learning_ambiguity_counts = {
        suite_task_id: sum(
            int(answer["generator_diagnostics"]["learning_hypothesis_size"]) > 1
            for answer in task_buckets[suite_task_id]
        )
        for suite_task_id in SUITE_TASKS
    }

    return {
        "label_counts": dict(sorted(label_counts.items())),
        "difficulty_bin_counts": dict(
            sorted(Counter(str(answer["difficulty_bin"]) for answer in answers).items())
        ),
        "previous_rule_baseline": summarize_metric("previous_rule_accuracy"),
        "majority_label_baseline": summarize_metric("majority_label_accuracy"),
        "nearest_neighbor_baseline": summarize_metric("nearest_neighbor_accuracy"),
        "cue_agnostic_baseline": summarize_metric("cue_agnostic_accuracy"),
        "symbolic_baseline": summarize_metric("symbolic_majority_accuracy"),
        "adversarial_baseline": summarize_metric("adversarial_baseline_accuracy"),
        "shift_conflict_counts": {
            suite_task_id: dict(
                sorted(
                    Counter(
                        int(answer["generator_diagnostics"]["shift_evidence_conflict_count"])
                        for answer in task_buckets[suite_task_id]
                    ).items()
                )
            )
            for suite_task_id in SUITE_TASKS
        },
        "learning_ambiguity_counts": learning_ambiguity_counts,
        "symbolic_unanimous_counts": unanimous_counts,
    }


def verify_split(split: str) -> None:
    rows = load_rows(split)
    if split == "private":
        legacy_message = detect_legacy_rows(rows)
        if legacy_message is not None:
            raise RuntimeError(legacy_message)

    schema_summary = verify_schema(rows, split)

    if split == "public":
        expected_rows, public_answers = build_split("public", variants_per_family=2)
        for row in expected_rows:
            row.pop("split", None)
        if rows != expected_rows:
            raise RuntimeError("public split rows are not reproducible from the generator")
        scored_rows = rows
        answer_key_summary: dict[str, object] = {}
        answers = public_answers
        manifest = None
    else:
        answer_key = load_private_answer_key()
        extra_summary, _episode_targets, answers = verify_private_answer_key(answer_key, rows)
        answer_key_summary = extra_summary
        scored_rows = attach_private_scoring(rows, answer_key)
        manifest = load_private_manifest(PRIVATE_MANIFEST_PATH)
        regenerated_rows, regenerated_answers, _ = build_private_artifacts(PRIVATE_MANIFEST_PATH)
        expected_private_rows = sanitize_private_rows(regenerated_rows)
        expected_answer_key = private_answer_key_payload(regenerated_answers)
        if rows != expected_private_rows:
            raise RuntimeError("private split rows are not reproducible from the private split manifest")
        if answer_key != expected_answer_key:
            raise RuntimeError("private split answer key is not reproducible from the private split manifest")

    public_regenerated_rows, public_answers = build_split("public", variants_per_family=2)
    for row in public_regenerated_rows:
        row.pop("split", None)
    if split == "private":
        verify_split_isolation(public_regenerated_rows, scored_rows)
        validate_split_isolation(public_answers, answers)
    else:
        if PRIVATE_ROWS_PATH.exists() and PRIVATE_ANSWER_KEY_PATH.exists():
            private_rows = load_rows("private")
            private_payload = load_private_answer_key()
            private_scored_rows = attach_private_scoring(private_rows, private_payload)
            _summary, _targets, private_answers = verify_private_answer_key(private_payload, private_rows)
            verify_split_isolation(scored_rows, private_scored_rows)
            validate_split_isolation(answers, private_answers)

    diagnostics_summary = summarize_generator_diagnostics(answers)

    print(
        json.dumps(
            {
                "split": split,
                "rows_path": str(dataset_path(split)),
                **schema_summary,
                **answer_key_summary,
                **(
                    {
                        "answer_key_path": str(PRIVATE_ANSWER_KEY_PATH),
                        "manifest_path": str(PRIVATE_MANIFEST_PATH),
                        "manifest_loaded": manifest is not None,
                    }
                    if split == "private"
                    else {}
                ),
                "baseline_ceilings": {
                    "symbolic_micro_accuracy_max": 0.58,
                    "symbolic_task_accuracy_max": 0.60,
                    "adversarial_micro_accuracy_max": 0.65,
                    "adversarial_task_accuracy_max": 0.65,
                    "explicit_or_latent_previous_rule_max": 0.25,
                    "context_or_cued_one_rule_max": 0.50,
                },
                **diagnostics_summary,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the deterministic CogFlex suite evaluation path.")
    parser.add_argument("--split", choices=("public", "private"), default="public")
    args = parser.parse_args()
    verify_split(args.split)


if __name__ == "__main__":
    main()
