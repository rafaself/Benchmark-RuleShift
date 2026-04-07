#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Final

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.build_ruleshift_dataset import (  # noqa: E402
    FACULTY_ID,
    FINAL_OUTPUT_INSTRUCTION,
    FINAL_PROBE_COUNT,
    GROUPS,
    PRIVATE_ANSWER_KEY_PATH,
    PRIVATE_MANIFEST_PATH,
    PRIVATE_ROWS_PATH,
    PUBLIC_ROWS_PATH,
    RULE_BY_ID,
    TYPE_FALSE,
    TYPE_TRUE,
    build_private_artifacts,
    load_private_manifest,
    private_answer_key_payload,
    sanitize_private_rows,
    validate_episode_constraints,
)

EXPECTED_COUNTS: Final[dict[str, int]] = {"public": 80, "private": 400}
EXPECTED_GROUP_COUNTS: Final[dict[str, int]] = {"public": 20, "private": 100}
ALLOWED_LABELS: Final[set[str]] = {TYPE_TRUE, TYPE_FALSE}
TURN_COUNT: Final[int] = 3
TURN_HEADERS: Final[tuple[str, str, str]] = (
    "Learn the current classification rule",
    None,
    "Probes:\n",
)

EXAMPLE_LINE_RE = re.compile(
    r"^(?P<index>\d+)\.\s+"
    r"(?:(?:context)=(?P<context>[a-z]+)\s+\|\s+)?"
    r"r1=(?P<r1>[+-]\d+),\s+r2=(?P<r2>[+-]\d+)\s+->\s+(?P<label>type_a|type_b)$"
)
PROBE_LINE_RE = re.compile(
    r"^(?P<index>\d+)\.\s+"
    r"(?:(?:context)=(?P<context>[a-z]+)\s+\|\s+)?"
    r"r1=(?P<r1>[+-]\d+),\s+r2=(?P<r2>[+-]\d+)\s+->\s+\?$"
)


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
    if len(normalized) != FINAL_PROBE_COUNT or any(label not in ALLOWED_LABELS for label in normalized):
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


def _parse_example_lines(turn: str) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for line in turn.splitlines():
        match = EXAMPLE_LINE_RE.match(line.strip())
        if match is None:
            continue
        items.append(
            {
                "index": int(match.group("index")),
                "r1": int(match.group("r1")),
                "r2": int(match.group("r2")),
                "context": match.group("context"),
                "label": match.group("label"),
            }
        )
    return items


def _parse_probe_lines(turn: str) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for line in turn.splitlines():
        match = PROBE_LINE_RE.match(line.strip())
        if match is None:
            continue
        items.append(
            {
                "index": int(match.group("index")),
                "r1": int(match.group("r1")),
                "r2": int(match.group("r2")),
                "context": match.group("context"),
            }
        )
    return items


def _row_to_answer_like(row: dict[str, object], targets: tuple[str, ...]) -> dict[str, object]:
    analysis = row["analysis"]
    turns = row["inference"]["turns"]
    initial_rule = RULE_BY_ID[str(analysis["initial_rule_id"])]
    probe_specs = _parse_probe_lines(turns[2])
    final_probes = []
    for probe, label in zip(probe_specs, targets):
        point = (int(probe["r1"]), int(probe["r2"]))
        item = {
            "index": probe["index"],
            "r1": point[0],
            "r2": point[1],
            "label": label,
            "previous_rule_label": TYPE_TRUE if initial_rule.label(point) else TYPE_FALSE,
        }
        if probe["context"] is not None:
            item["context"] = probe["context"]
        final_probes.append(item)
    return {
        "episode_id": row["episode_id"],
        "group_id": analysis["group_id"],
        "initial_rule_id": analysis["initial_rule_id"],
        "shift_rule_id": analysis["shift_rule_id"],
        "final_probe_targets": targets,
        "final_probes": final_probes,
    }


def verify_schema(rows: list[dict[str, object]], split: str) -> dict[str, dict[str, int] | int]:
    expected_group_count = EXPECTED_GROUP_COUNTS[split]
    if len(rows) != EXPECTED_COUNTS[split]:
        raise RuntimeError(f"{split} split expected {EXPECTED_COUNTS[split]} rows, found {len(rows)}")

    group_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    shift_mode_counts: Counter[str] = Counter()

    for row in rows:
        expected_row_keys = ["analysis", "episode_id", "inference", "scoring"]
        if split == "private":
            expected_row_keys = ["analysis", "episode_id", "inference"]
        if sorted(row.keys()) != expected_row_keys:
            raise RuntimeError(f"row {row.get('episode_id')} has inconsistent top-level keys")

        episode_id = row["episode_id"]
        analysis = row["analysis"]
        inference = row["inference"]
        expected_analysis_keys = [
            "faculty_id",
            "group_id",
            "initial_rule_id",
            "shift_mode",
            "shift_rule_id",
            "transition_family_id",
        ]
        if not isinstance(analysis, dict) or sorted(analysis.keys()) != expected_analysis_keys:
            raise RuntimeError(f"row {episode_id} has inconsistent analysis keys")
        if analysis["faculty_id"] != FACULTY_ID:
            raise RuntimeError(f"row {episode_id} exposes unsupported faculty_id {analysis['faculty_id']!r}")
        if not isinstance(inference, dict) or sorted(inference.keys()) != ["turns"]:
            raise RuntimeError(f"row {episode_id} has inconsistent inference keys")

        turns = inference["turns"]
        if not isinstance(turns, list) or len(turns) != TURN_COUNT:
            raise RuntimeError(f"row {episode_id} must expose exactly {TURN_COUNT} turns")
        for turn_index, turn in enumerate(turns, start=1):
            if not isinstance(turn, str) or not turn.startswith(
                f"RuleShift cognitive flexibility task. Episode {episode_id}. Turn {turn_index} of 3."
            ):
                raise RuntimeError(f"row {episode_id} has malformed turn {turn_index}")
        if FINAL_OUTPUT_INSTRUCTION not in turns[2]:
            raise RuntimeError(f"row {episode_id} decision turn has inconsistent output instruction")

        learn_examples = _parse_example_lines(turns[0])
        shift_examples = _parse_example_lines(turns[1])
        probes = _parse_probe_lines(turns[2])
        if len(learn_examples) != 4 or len(shift_examples) != 4 or len(probes) != FINAL_PROBE_COUNT:
            raise RuntimeError(f"row {episode_id} has malformed example/probe counts")

        if split == "public":
            scoring = row["scoring"]
            if not isinstance(scoring, dict) or sorted(scoring.keys()) != ["final_probe_targets"]:
                raise RuntimeError(f"row {episode_id} has inconsistent scoring keys")
            targets = normalize_labels(scoring["final_probe_targets"])
            if targets is None:
                raise RuntimeError(f"row {episode_id} has invalid final_probe_targets")
            validate_episode_constraints(row, _row_to_answer_like(row, targets))
        elif "scoring" in row:
            raise RuntimeError(f"row {episode_id} leaks scoring fields in the private split")

        group_id = str(analysis["group_id"])
        if group_id not in GROUPS:
            raise RuntimeError(f"row {episode_id} exposes unsupported group {group_id!r}")

        group_counts[group_id] += 1
        family_counts[str(analysis["transition_family_id"])] += 1
        shift_mode_counts[str(analysis["shift_mode"])] += 1

    if set(group_counts) != set(GROUPS):
        raise RuntimeError(f"{split} split groups mismatch: {sorted(group_counts)}")
    mismatched_counts = {group: count for group, count in group_counts.items() if count != expected_group_count}
    if mismatched_counts:
        raise RuntimeError(f"{split} split group counts mismatch: {mismatched_counts}")

    return {
        "row_count": len(rows),
        "group_counts": dict(sorted(group_counts.items())),
        "family_count": len(family_counts),
        "shift_mode_counts": dict(sorted(shift_mode_counts.items())),
    }


def verify_private_answer_key(
    payload: dict[str, object],
    private_rows: list[dict[str, object]],
) -> tuple[dict[str, dict[str, int] | int], dict[str, tuple[str, ...]]]:
    if payload.get("version") != "cogflex_v2_multi_turn":
        raise RuntimeError("private answer key has an unsupported version")
    if payload.get("split") != "private":
        raise RuntimeError("private answer key must declare split='private'")

    episodes = payload.get("episodes")
    if not isinstance(episodes, list):
        raise RuntimeError("private answer key must expose an episodes list")

    rows_by_id = {str(row["episode_id"]): row for row in private_rows}
    episode_targets: dict[str, tuple[str, ...]] = {}
    label_counts: Counter[str] = Counter()

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

        for required in (
            "faculty_id",
            "group_id",
            "transition_family_id",
            "initial_rule_id",
            "shift_rule_id",
            "shift_mode",
            "learn_turn_examples",
            "shift_turn_examples",
            "final_probes",
            "turns",
        ):
            if required not in answer:
                raise RuntimeError(f"private answer key episode {episode_id} is missing {required}")

        if answer["turns"] != row["inference"]["turns"]:
            raise RuntimeError(f"private answer key turns mismatch for episode {episode_id}")
        for key in ("faculty_id", "group_id", "transition_family_id", "initial_rule_id", "shift_rule_id", "shift_mode"):
            if answer[key] != row["analysis"][key]:
                raise RuntimeError(f"private answer key {key} mismatch for episode {episode_id}")

        validate_episode_constraints(row, _row_to_answer_like(row, targets))
        episode_targets[episode_id] = targets
        label_counts.update(targets)

    missing_ids = set(rows_by_id) - set(episode_targets)
    if missing_ids:
        raise RuntimeError(f"private answer key is missing episode_ids: {sorted(missing_ids)}")

    return (
        {
            "answer_key_episode_count": len(episodes),
            "label_counts": dict(sorted(label_counts.items())),
        },
        episode_targets,
    )


def attach_private_scoring(
    private_rows: list[dict[str, object]],
    payload: dict[str, object],
) -> list[dict[str, object]]:
    _summary, episode_targets = verify_private_answer_key(payload, private_rows)
    enriched_rows: list[dict[str, object]] = []
    for row in private_rows:
        episode_id = str(row["episode_id"])
        enriched_rows.append(
            {
                "episode_id": row["episode_id"],
                "inference": {
                    "turns": list(row["inference"]["turns"]),
                },
                "scoring": {
                    "final_probe_targets": episode_targets[episode_id],
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
    return enriched_rows


def semantic_signature(row: dict[str, object]) -> tuple[object, ...]:
    targets = normalize_labels(row["scoring"]["final_probe_targets"])
    if targets is None:
        raise RuntimeError(f"row {row.get('episode_id')} has invalid final_probe_targets")
    return (
        row["analysis"]["group_id"],
        row["analysis"]["transition_family_id"],
        tuple(row["inference"]["turns"]),
        targets,
    )


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

    public_families = {str(row["analysis"]["transition_family_id"]) for row in public_rows}
    private_families = {str(row["analysis"]["transition_family_id"]) for row in private_rows}
    overlap = sorted(public_families & private_families)
    if overlap:
        raise RuntimeError(f"public/private transition families overlap: {overlap}")


def _parse_final_probe_specs(row: dict[str, object]) -> list[dict[str, object]]:
    return _parse_probe_lines(row["inference"]["turns"][2])


def _run_episode_predictions(rows: list[dict[str, object]], predictor) -> tuple[int, int]:
    numerator = 0
    denominator = 0
    for row in rows:
        predictions = predictor(row)
        result = score_episode(predictions, row["scoring"]["final_probe_targets"])
        numerator += result[0]
        denominator += result[1]
    return (numerator, denominator)


def run_oracle(rows: list[dict[str, object]]) -> tuple[int, int]:
    return _run_episode_predictions(rows, lambda row: row["scoring"]["final_probe_targets"])


def run_invalid(rows: list[dict[str, object]]) -> tuple[int, int]:
    return _run_episode_predictions(rows, lambda row: ("invalid",))


def run_previous_rule_baseline(rows: list[dict[str, object]]) -> tuple[int, int]:
    def predict(row: dict[str, object]) -> tuple[str, ...]:
        initial_rule = RULE_BY_ID[str(row["analysis"]["initial_rule_id"])]
        labels = []
        for probe in _parse_final_probe_specs(row):
            point = (int(probe["r1"]), int(probe["r2"]))
            labels.append(TYPE_TRUE if initial_rule.label(point) else TYPE_FALSE)
        return tuple(labels)

    return _run_episode_predictions(rows, predict)


def run_majority_label_baseline(rows: list[dict[str, object]]) -> tuple[int, int]:
    def predict(row: dict[str, object]) -> tuple[str, ...]:
        counts = Counter()
        for turn in row["inference"]["turns"][:2]:
            for item in _parse_example_lines(turn):
                counts.update([str(item["label"])])
        if counts[TYPE_TRUE] >= counts[TYPE_FALSE]:
            label = TYPE_TRUE
        else:
            label = TYPE_FALSE
        return (label,) * FINAL_PROBE_COUNT

    return _run_episode_predictions(rows, predict)


def run_context_agnostic_baseline(rows: list[dict[str, object]]) -> tuple[int, int]:
    context_rows = [row for row in rows if row["analysis"]["group_id"] == "context_switch"]

    def predict(row: dict[str, object]) -> tuple[str, ...]:
        shift_rule = RULE_BY_ID[str(row["analysis"]["shift_rule_id"])]
        labels = []
        for probe in _parse_final_probe_specs(row):
            point = (int(probe["r1"]), int(probe["r2"]))
            labels.append(TYPE_TRUE if shift_rule.label(point) else TYPE_FALSE)
        return tuple(labels)

    return _run_episode_predictions(context_rows, predict) if context_rows else (0, 0)


def verify_split(split: str) -> None:
    rows = load_rows(split)
    schema_summary = verify_schema(rows, split)
    answer_key_summary: dict[str, dict[str, int] | int] = {}

    if split == "private":
        answer_key = load_private_answer_key()
        extra_summary, _episode_targets = verify_private_answer_key(answer_key, rows)
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
    else:
        manifest = None
        scored_rows = rows

    if PUBLIC_ROWS_PATH.exists() and PRIVATE_ROWS_PATH.exists():
        public_rows = scored_rows if split == "public" else load_rows("public")
        private_rows: list[dict[str, object]]
        if split == "private":
            private_rows = scored_rows
        elif PRIVATE_ANSWER_KEY_PATH.exists():
            private_rows = attach_private_scoring(load_rows("private"), load_private_answer_key())
        else:
            private_rows = []
        if private_rows:
            verify_split_isolation(public_rows, private_rows)

    oracle_score = run_oracle(scored_rows)
    invalid_score = run_invalid(scored_rows)
    previous_rule_score = run_previous_rule_baseline(scored_rows)
    majority_label_score = run_majority_label_baseline(scored_rows)
    context_agnostic_score = run_context_agnostic_baseline(scored_rows)

    if oracle_score != (len(scored_rows) * FINAL_PROBE_COUNT, len(scored_rows) * FINAL_PROBE_COUNT):
        raise RuntimeError(f"{split} split oracle run produced unexpected score {oracle_score}")
    if invalid_score != (0, len(scored_rows) * FINAL_PROBE_COUNT):
        raise RuntimeError(f"{split} split invalid-response score produced unexpected score {invalid_score}")

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
                "oracle_score": {"numerator": oracle_score[0], "denominator": oracle_score[1]},
                "invalid_score": {"numerator": invalid_score[0], "denominator": invalid_score[1]},
                "previous_rule_score": {"numerator": previous_rule_score[0], "denominator": previous_rule_score[1]},
                "majority_label_score": {"numerator": majority_label_score[0], "denominator": majority_label_score[1]},
                "context_agnostic_score": {
                    "numerator": context_agnostic_score[0],
                    "denominator": context_agnostic_score[1],
                },
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the deterministic RuleShift CogFlex v2 evaluation path.")
    parser.add_argument("--split", choices=("public", "private"), default="public")
    args = parser.parse_args()
    verify_split(args.split)


if __name__ == "__main__":
    main()
