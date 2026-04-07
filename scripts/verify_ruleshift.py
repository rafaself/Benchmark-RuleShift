#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Final

from scripts.build_ruleshift_dataset import (
    PRIVATE_ANSWER_KEY_PATH,
    PRIVATE_MANIFEST_PATH,
    build_private_artifacts,
    load_private_manifest,
    private_answer_key_payload,
    sanitize_private_rows,
)

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
    for row in rows:
        expected_row_keys = ["analysis", "episode_id", "inference", "scoring"]
        if split == "private":
            expected_row_keys = ["analysis", "episode_id", "inference"]
        if sorted(row.keys()) != expected_row_keys:
            raise RuntimeError(f"row {row.get('episode_id')} has inconsistent top-level keys")

        episode_id = row["episode_id"]
        analysis = row["analysis"]
        inference = row["inference"]
        expected_analysis_keys = ["group_id", "rule_id", "shortcut_rule_id", "shortcut_type"]
        if not isinstance(analysis, dict) or sorted(analysis.keys()) != expected_analysis_keys:
            raise RuntimeError(f"row {episode_id} has inconsistent analysis keys")
        if not isinstance(inference, dict) or sorted(inference.keys()) != ["prompt"]:
            raise RuntimeError(f"row {episode_id} has inconsistent inference keys")
        if split == "public":
            scoring = row["scoring"]
            if not isinstance(scoring, dict) or sorted(scoring.keys()) != ["probe_targets"]:
                raise RuntimeError(f"row {episode_id} has inconsistent scoring keys")
            targets = normalize_labels(scoring["probe_targets"])
            if targets is None:
                raise RuntimeError(f"row {episode_id} has invalid probe targets")
        elif "scoring" in row:
            raise RuntimeError(f"row {episode_id} leaks scoring fields in the private split")

        group_id = analysis["group_id"]
        if group_id not in EXPECTED_GROUPS:
            raise RuntimeError(f"row {episode_id} exposes unsupported group {group_id!r}")
        if not isinstance(inference["prompt"], str) or not inference["prompt"].strip():
            raise RuntimeError(f"row {episode_id} has empty inference prompt")

        group_counts[str(group_id)] += 1

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
        "rule_count": len(rule_counts),
        "shortcut_counts": dict(sorted(shortcut_counts.items())),
    }


def verify_private_answer_key(
    payload: dict[str, object],
    private_rows: list[dict[str, object]],
) -> tuple[dict[str, dict[str, int] | int], dict[str, tuple[str, ...]]]:
    if payload.get("version") != "scoped_single_turn":
        raise RuntimeError("private answer key has an unsupported version")
    if payload.get("split") != "private":
        raise RuntimeError("private answer key must declare split='private'")

    episodes = payload.get("episodes")
    if not isinstance(episodes, list):
        raise RuntimeError("private answer key must expose an episodes list")

    expected_ids = {str(row["episode_id"]) for row in private_rows}
    episode_targets: dict[str, tuple[str, ...]] = {}
    label_counts: Counter[str] = Counter()

    for answer in episodes:
        if not isinstance(answer, dict):
            raise RuntimeError("private answer key episodes must be objects")
        episode_id = str(answer.get("episode_id"))
        if episode_id in episode_targets:
            raise RuntimeError(f"private answer key duplicates episode_id {episode_id}")
        if episode_id not in expected_ids:
            raise RuntimeError(f"private answer key contains unknown episode_id {episode_id}")

        targets = normalize_labels(answer.get("probe_targets"))
        if targets is None:
            raise RuntimeError(f"private answer key episode {episode_id} has invalid probe_targets")

        for required in ("group_id", "rule_id", "rule_description", "shortcut_type", "examples", "probes"):
            if required not in answer:
                raise RuntimeError(f"private answer key episode {episode_id} is missing {required}")

        if not isinstance(answer["examples"], list) or not isinstance(answer["probes"], list):
            raise RuntimeError(f"private answer key episode {episode_id} has malformed examples/probes")

        episode_targets[episode_id] = targets
        label_counts.update(targets)

    missing_ids = expected_ids - set(episode_targets)
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
                    "prompt": row["inference"]["prompt"],
                },
                "scoring": {
                    "probe_targets": episode_targets[episode_id],
                },
                "analysis": {
                    "group_id": row["analysis"]["group_id"],
                    "rule_id": row["analysis"]["rule_id"],
                    "shortcut_type": row["analysis"]["shortcut_type"],
                    "shortcut_rule_id": row["analysis"]["shortcut_rule_id"],
                },
            }
        )
    return enriched_rows


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
    answer_key_summary: dict[str, dict[str, int] | int] = {}

    if split == "private":
        answer_key_payload = load_private_answer_key()
        extra_summary, _episode_targets = verify_private_answer_key(answer_key_payload, rows)
        answer_key_summary = extra_summary
        scored_rows = attach_private_scoring(rows, answer_key_payload)
        manifest = load_private_manifest(PRIVATE_MANIFEST_PATH)
        regenerated_rows, regenerated_answers, _ = build_private_artifacts(PRIVATE_MANIFEST_PATH)
        expected_private_rows = sanitize_private_rows(regenerated_rows)
        expected_answer_key = private_answer_key_payload(regenerated_answers)
        if rows != expected_private_rows:
            raise RuntimeError("private split rows are not reproducible from the private split manifest")
        if answer_key_payload != expected_answer_key:
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

    oracle_a = run_oracle(scored_rows)
    oracle_b = run_oracle(scored_rows)
    invalid_score = run_invalid(scored_rows)

    if oracle_a != oracle_b:
        raise RuntimeError(f"{split} split oracle run is not reproducible: {oracle_a} != {oracle_b}")
    if oracle_a != (len(scored_rows) * PROBE_COUNT, len(scored_rows) * PROBE_COUNT):
        raise RuntimeError(f"{split} split oracle run produced unexpected score {oracle_a}")
    if invalid_score != (0, len(scored_rows) * PROBE_COUNT):
        raise RuntimeError(f"{split} split invalid-response score produced unexpected score {invalid_score}")

    print(json.dumps(
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
