#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Final

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROWS_PATH = ROOT / "kaggle/dataset/public/public_leaderboard_rows.json"
PRIVATE_ROWS_PATH = ROOT / "kaggle/dataset/private/private_leaderboard_rows.json"
EXPECTED_COUNTS: Final[dict[str, int]] = {"public": 80, "private": 400}
EXPECTED_GROUP_COUNTS: Final[dict[str, int]] = {"public": 20, "private": 100}
EXPECTED_GROUPS: Final[set[str]] = {"simple", "exception", "distractor", "hard"}
EXPECTED_RULE_COUNT: Final[int] = 20
PROBE_COUNT: Final[int] = 4
ALLOWED_LABELS: Final[set[str]] = {"type_a", "type_b"}


def dataset_path(split: str) -> Path:
    return PUBLIC_ROWS_PATH if split == "public" else PRIVATE_ROWS_PATH


def load_rows(split: str) -> list[dict[str, object]]:
    rows = json.loads(dataset_path(split).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise RuntimeError(f"{split} dataset payload must be a list")
    return rows


def normalize_labels(values: object) -> tuple[str, ...] | None:
    if not isinstance(values, (list, tuple)):
        return None
    normalized = tuple(str(value).strip().lower() for value in values)
    if len(normalized) != PROBE_COUNT or any(label not in ALLOWED_LABELS for label in normalized):
        return None
    return normalized


def score_episode(predictions: object, targets: object) -> tuple[int, int]:
    norm_targets = normalize_labels(targets)
    if norm_targets is None:
        raise ValueError("targets must contain exactly four valid labels")
    norm_predictions = normalize_labels(predictions)
    if norm_predictions is None:
        return (0, PROBE_COUNT)
    return (sum(pred == target for pred, target in zip(norm_predictions, norm_targets)), PROBE_COUNT)


def verify_schema(rows: list[dict[str, object]], split: str) -> dict[str, dict[str, int] | int]:
    expected_group_count = EXPECTED_GROUP_COUNTS[split]
    if len(rows) != EXPECTED_COUNTS[split]:
        raise RuntimeError(f"{split} split expected {EXPECTED_COUNTS[split]} rows, found {len(rows)}")

    group_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    for row in rows:
        if sorted(row.keys()) != ["analysis", "episode_id", "inference", "scoring"]:
            raise RuntimeError(f"row {row.get('episode_id')} has inconsistent top-level keys")

        episode_id = row["episode_id"]
        analysis = row["analysis"]
        inference = row["inference"]
        scoring = row["scoring"]
        expected_analysis_keys = ["group_id", "rule_id", "shortcut_rule_id", "shortcut_type"]
        if not isinstance(analysis, dict) or sorted(analysis.keys()) != expected_analysis_keys:
            raise RuntimeError(f"row {episode_id} has inconsistent analysis keys")
        if not isinstance(inference, dict) or sorted(inference.keys()) != ["prompt"]:
            raise RuntimeError(f"row {episode_id} has inconsistent inference keys")
        if not isinstance(scoring, dict) or sorted(scoring.keys()) != ["probe_targets"]:
            raise RuntimeError(f"row {episode_id} has inconsistent scoring keys")

        group_id = analysis["group_id"]
        if group_id not in EXPECTED_GROUPS:
            raise RuntimeError(f"row {episode_id} exposes unsupported group {group_id!r}")
        if not isinstance(inference["prompt"], str) or not inference["prompt"].strip():
            raise RuntimeError(f"row {episode_id} has empty inference prompt")

        targets = normalize_labels(scoring["probe_targets"])
        if targets is None:
            raise RuntimeError(f"row {episode_id} has invalid probe targets")

        group_counts[str(group_id)] += 1
        label_counts.update(targets)

    if set(group_counts) != EXPECTED_GROUPS:
        raise RuntimeError(f"{split} split groups mismatch: {sorted(group_counts)}")
    mismatched_counts = {group: count for group, count in group_counts.items() if count != expected_group_count}
    if mismatched_counts:
        raise RuntimeError(f"{split} split group counts mismatch: {mismatched_counts}")

    rule_counts = Counter(str(row["analysis"]["rule_id"]) for row in rows)
    shortcut_counts = Counter(str(row["analysis"]["shortcut_type"]) for row in rows)
    if len(rule_counts) != EXPECTED_RULE_COUNT:
        raise RuntimeError(f"{split} split expected {EXPECTED_RULE_COUNT} rules, found {len(rule_counts)}")

    return {
        "row_count": len(rows),
        "group_counts": dict(sorted(group_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "rule_count": len(rule_counts),
        "shortcut_counts": dict(sorted(shortcut_counts.items())),
    }


def semantic_signature(row: dict[str, object]) -> tuple[object, ...]:
    prompt = str(row["inference"]["prompt"])
    parts = prompt.split("\n\n")
    if len(parts) != 4:
        raise RuntimeError(f"row {row.get('episode_id')} has malformed prompt blocks")
    _, examples_block, probes_block, _ = parts
    targets = normalize_labels(row["scoring"]["probe_targets"])
    if targets is None:
        raise RuntimeError(f"row {row.get('episode_id')} has invalid probe targets")
    return (
        row["analysis"]["group_id"],
        row["analysis"]["rule_id"],
        examples_block,
        probes_block,
        targets,
    )


def verify_split_isolation(
    public_rows: list[dict[str, object]],
    private_rows: list[dict[str, object]],
) -> None:
    public_signatures = {
        semantic_signature(row): str(row["episode_id"])
        for row in public_rows
    }
    for row in private_rows:
        signature = semantic_signature(row)
        public_episode_id = public_signatures.get(signature)
        if public_episode_id is None:
            continue
        raise RuntimeError(
            "public/private split isolation violated: "
            f"public episode {public_episode_id} overlaps private episode {row['episode_id']}"
        )


def run_oracle(rows: list[dict[str, object]]) -> tuple[int, int]:
    numerator = 0
    denominator = 0
    for row in rows:
        result = score_episode(row["scoring"]["probe_targets"], row["scoring"]["probe_targets"])
        numerator += result[0]
        denominator += result[1]
    return (numerator, denominator)


def run_invalid(rows: list[dict[str, object]]) -> tuple[int, int]:
    numerator = 0
    denominator = 0
    for row in rows:
        result = score_episode(("invalid",), row["scoring"]["probe_targets"])
        numerator += result[0]
        denominator += result[1]
    return (numerator, denominator)


def verify_split(split: str) -> None:
    rows = load_rows(split)
    schema_summary = verify_schema(rows, split)
    if PUBLIC_ROWS_PATH.exists() and PRIVATE_ROWS_PATH.exists():
        public_rows = rows if split == "public" else load_rows("public")
        private_rows = rows if split == "private" else load_rows("private")
        verify_split_isolation(public_rows, private_rows)
    oracle_a = run_oracle(rows)
    oracle_b = run_oracle(rows)
    invalid_score = run_invalid(rows)

    if oracle_a != oracle_b:
        raise RuntimeError(f"{split} split oracle run is not reproducible: {oracle_a} != {oracle_b}")
    if oracle_a != (len(rows) * PROBE_COUNT, len(rows) * PROBE_COUNT):
        raise RuntimeError(f"{split} split oracle run produced unexpected score {oracle_a}")
    if invalid_score != (0, len(rows) * PROBE_COUNT):
        raise RuntimeError(f"{split} split invalid-response score produced unexpected score {invalid_score}")

    print(json.dumps(
        {
            "split": split,
            "rows_path": str(dataset_path(split)),
            **schema_summary,
            "oracle_score": {"numerator": oracle_a[0], "denominator": oracle_a[1]},
            "invalid_score": {"numerator": invalid_score[0], "denominator": invalid_score[1]},
        },
        indent=2,
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the deterministic RuleShift evaluation path.")
    parser.add_argument("--split", choices=("public", "private"), default="public")
    args = parser.parse_args()
    verify_split(args.split)


if __name__ == "__main__":
    main()
