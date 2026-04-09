#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.build_cogflex_dataset import (  # noqa: E402
    FACULTY_ID,
    PRIVATE_ANSWER_KEY_FILENAME,
    PRIVATE_ANSWER_KEY_VERSION,
    PRIVATE_CALIBRATION_PREDICTIONS_FILENAME,
    PRIVATE_CALIBRATION_PREDICTIONS_VERSION,
    PRIVATE_BUNDLE_ENV_VAR,
    PRIVATE_BUNDLE_VERSION,
    PRIVATE_QUALITY_REPORT_FILENAME,
    PRIVATE_QUALITY_REPORT_VERSION,
    PRIVATE_RELEASE_MANIFEST_FILENAME,
    PRIVATE_ROWS_FILENAME,
    PUBLIC_BUNDLE_VERSION,
    PUBLIC_DIFFICULTY_CALIBRATION_PATH,
    PUBLIC_EPISODES_PER_TASK,
    PUBLIC_QUALITY_REPORT_PATH,
    PUBLIC_ROWS_PATH,
    REQUIRED_PRIVATE_STRUCTURE_FAMILY_IDS,
    SUPPORTED_OPERATOR_CLASSES,
    SHIFT_MODES,
    SUITE_TASKS,
    public_generator_reference,
    empirical_difficulty_entries_from_predictions,
    load_public_difficulty_calibration,
    TASK_NAME,
    TURN_HEADER_PREFIX,
    build_public_artifacts,
    compute_sha256,
    normalized_turn_text,
    parse_case_line,
    parse_turn_items,
)

EXPECTED_PUBLIC_ROW_COUNT: Final[int] = len(SUITE_TASKS) * PUBLIC_EPISODES_PER_TASK
PRIVATE_NEAR_DUPLICATE_OVERLAP_THRESHOLD: Final[float] = 0.9
PRIVATE_PANEL_MODEL_COUNT: Final[int] = 3
AUDIT_REPORT_VERSION: Final[int] = 1
AUDIT_VERIFIER_VERSION: Final[str] = "audit-v1"
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
ATTACK_SUITE_DIMENSIONS: Final[tuple[str, ...]] = (
    "difficulty_bin",
    "shift_mode",
    "structure_family_id",
    "suite_task_id",
)
PUBLIC_AUDIT_CHECKS: Final[tuple[str, ...]] = (
    "schema",
    "difficulty_calibration",
    "public_rows_reproducibility",
    "public_quality_report_reproducibility",
    "public_quality_report_consistency",
)
PRIVATE_AUDIT_CHECKS: Final[tuple[str, ...]] = (
    "schema",
    "answer_key_consistency",
    "calibration_predictions",
    "empirical_difficulty",
    "release_manifest_digests",
    "private_quality_report_schema",
    "private_quality_report_reproducibility",
    "public_private_split_isolation",
    "generator_non_overlap",
    "required_structure_family_coverage",
)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _resolve_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    commit = result.stdout.strip()
    if result.returncode != 0 or len(commit) != 40:
        return None
    return commit


def write_audit_report(
    report_path: Path,
    *,
    split: str,
    artifact_digests: dict[str, str],
    checks_executed: tuple[str, ...],
    summary: dict[str, object],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "audit_report_version": AUDIT_REPORT_VERSION,
                "verifier": {
                    "script": "scripts.verify_cogflex",
                    "version": AUDIT_VERIFIER_VERSION,
                },
                "git_commit": _resolve_git_commit(),
                "split": split,
                "timestamp_utc": _utc_timestamp(),
                "verification_result": "passed",
                "artifact_digests": dict(sorted(artifact_digests.items())),
                "checks_executed": list(checks_executed),
                "summary": summary,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def normalize_labels(values: object, label_vocab: list[str]) -> tuple[str, ...] | None:
    if not isinstance(values, (list, tuple)):
        return None
    normalized = tuple(str(value).strip() for value in values)
    if len(normalized) == 0 or any(label not in label_vocab for label in normalized):
        return None
    return normalized


def _normalize_generator_metadata(episode: dict[str, object], *, episode_id: str) -> dict[str, str]:
    generator = episode.get("generator")
    if not isinstance(generator, dict):
        raise RuntimeError(f"private answer key episode {episode_id} must include generator metadata")
    expected_keys = {"family_id", "operator_class", "template_id"}
    if set(generator) != expected_keys:
        raise RuntimeError(
            f"private answer key episode {episode_id} generator must expose {sorted(expected_keys)}"
        )
    normalized = {key: str(generator[key]).strip() for key in expected_keys}
    if any(not value for value in normalized.values()):
        raise RuntimeError(f"private answer key episode {episode_id} has empty generator metadata")
    if normalized["operator_class"] not in SUPPORTED_OPERATOR_CLASSES:
        raise RuntimeError(
            f"private answer key episode {episode_id} has unsupported operator_class {normalized['operator_class']!r}"
        )
    return normalized


def _rounded_accuracy(correct: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(correct / total, 6)


def _verify_metric_value(value: object, *, context: str) -> None:
    if not isinstance(value, (int, float)):
        raise RuntimeError(f"{context} must be numeric")


def _verify_model_metric_payload(model: object, *, context: str) -> None:
    if not isinstance(model, dict):
        raise RuntimeError(f"{context} entries must be objects")
    expected_keys = {"macro_accuracy", "micro_accuracy", "name", "per_task_accuracy"}
    if set(model) != expected_keys:
        raise RuntimeError(f"{context} entries must expose {sorted(expected_keys)}")
    if not isinstance(model.get("name"), str) or not model["name"].strip():
        raise RuntimeError(f"{context} name must be a non-empty string")
    _verify_metric_value(model.get("macro_accuracy"), context=f"{context} macro_accuracy")
    _verify_metric_value(model.get("micro_accuracy"), context=f"{context} micro_accuracy")
    per_task_accuracy = model.get("per_task_accuracy")
    if not isinstance(per_task_accuracy, dict) or set(per_task_accuracy) != set(SUITE_TASKS):
        raise RuntimeError(f"{context} per_task_accuracy must cover all suite tasks")
    for suite_task_id in SUITE_TASKS:
        _verify_metric_value(
            per_task_accuracy[suite_task_id],
            context=f"{context} per_task_accuracy[{suite_task_id}]",
        )


def _episode_ids_by_dimension(
    private_rows: list[dict[str, object]],
    dimension: str,
) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for row in private_rows:
        if dimension in row["analysis"]:
            value = str(row["analysis"][dimension])
        else:
            raise RuntimeError(f"unsupported attack suite dimension {dimension}")
        grouped.setdefault(value, []).append(str(row["episode_id"]))
    return dict(sorted(grouped.items()))


def _metrics_for_episode_subset(
    episode_ids: list[str],
    *,
    private_rows: list[dict[str, object]],
    episode_targets: dict[str, tuple[str, ...]],
    predictions_by_model: list[dict[str, object]],
) -> dict[str, object]:
    rows_by_id = {str(row["episode_id"]): row for row in private_rows}
    total_probe_count = sum(len(episode_targets[episode_id]) for episode_id in episode_ids)
    models_summary: list[dict[str, object]] = []
    for model in predictions_by_model:
        model_total_correct = 0
        per_task_correct = {suite_task_id: 0 for suite_task_id in SUITE_TASKS}
        per_task_total = {suite_task_id: 0 for suite_task_id in SUITE_TASKS}
        model_episodes = model["episodes"]
        for episode_id in episode_ids:
            suite_task_id = str(rows_by_id[episode_id]["analysis"]["suite_task_id"])
            targets = episode_targets[episode_id]
            predictions = model_episodes[episode_id]
            correct = sum(1 for predicted, target in zip(predictions, targets, strict=True) if predicted == target)
            model_total_correct += correct
            per_task_correct[suite_task_id] += correct
            per_task_total[suite_task_id] += len(targets)
        per_task_accuracy = {
            suite_task_id: _rounded_accuracy(per_task_correct[suite_task_id], per_task_total[suite_task_id])
            for suite_task_id in SUITE_TASKS
        }
        represented_task_accuracies = [
            per_task_accuracy[suite_task_id] for suite_task_id in SUITE_TASKS if per_task_total[suite_task_id] > 0
        ]
        models_summary.append(
            {
                "name": model["name"],
                "micro_accuracy": _rounded_accuracy(model_total_correct, total_probe_count),
                "macro_accuracy": round(sum(represented_task_accuracies) / len(represented_task_accuracies), 6)
                if represented_task_accuracies
                else 0.0,
                "per_task_accuracy": per_task_accuracy,
            }
        )
    return {"models": models_summary}


def build_private_quality_report(
    private_rows: list[dict[str, object]],
    answer_key: dict[str, object],
    predictions: dict[str, object],
    *,
    public_rows: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    _summary, episode_targets, private_generator_metadata = verify_private_answer_key(answer_key, private_rows)
    predictions_by_model = verify_private_calibration_predictions(predictions, private_rows, episode_targets)
    verify_private_empirical_difficulty(private_rows, answer_key, predictions_by_model, episode_targets)
    scored_private_rows = attach_private_scoring(private_rows, answer_key)
    if public_rows is None:
        public_rows = load_rows(PUBLIC_ROWS_PATH)
    isolation_summary = verify_split_isolation(public_rows, scored_private_rows)
    generator_isolation_summary = verify_generator_isolation(private_generator_metadata)
    row_summary = _summary_from_rows(private_rows)
    all_episode_ids = [str(row["episode_id"]) for row in private_rows]
    attack_suite = {
        dimension: {
            value: {
                "row_count": len(episode_ids),
                **_metrics_for_episode_subset(
                    episode_ids,
                    private_rows=private_rows,
                    episode_targets=episode_targets,
                    predictions_by_model=predictions_by_model,
                ),
            }
            for value, episode_ids in _episode_ids_by_dimension(private_rows, dimension).items()
        }
        for dimension in ATTACK_SUITE_DIMENSIONS
    }
    return {
        "version": PRIVATE_QUALITY_REPORT_VERSION,
        "split": "private",
        "row_count": len(private_rows),
        **row_summary,
        "calibration_summary": _metrics_for_episode_subset(
            all_episode_ids,
            private_rows=private_rows,
            episode_targets=episode_targets,
            predictions_by_model=predictions_by_model,
        ),
        "attack_suite": attack_suite,
        "semantic_isolation_summary": isolation_summary,
        "generator_isolation_summary": generator_isolation_summary,
    }


def _response_spec(row: dict[str, object]) -> dict[str, object]:
    inference = row.get("inference")
    if not isinstance(inference, dict):
        raise RuntimeError(f"row {row.get('episode_id')} is missing inference")
    spec = inference.get("response_spec")
    if not isinstance(spec, dict):
        raise RuntimeError(f"row {row.get('episode_id')} is missing response_spec")
    return spec


def _normalize_nominal_maps(turns: list[list[dict[str, object]]], label_vocab: list[str]) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    field_maps: dict[str, dict[str, str]] = {}
    label_map = {label: f"label_{index}" for index, label in enumerate(label_vocab)}
    for items in turns:
        for item in items:
            for key, value in item.items():
                if key in {"index", "label", "r1", "r2"}:
                    continue
                if not isinstance(value, str):
                    continue
                field_maps.setdefault(key, {})
                field_maps[key].setdefault(value, f"{key}_{len(field_maps[key])}")
    return field_maps, label_map


def _canonical_item(item: dict[str, object], *, field_maps: dict[str, dict[str, str]], label_map: dict[str, str]) -> tuple[object, ...]:
    pieces: list[object] = [int(item["r1"]), int(item["r2"]), label_map[str(item["label"])]]
    for key in sorted(key for key in item if key not in {"index", "label", "r1", "r2", "rule_id"}):
        value = item[key]
        if isinstance(value, str):
            pieces.extend((key, field_maps[key][value]))
        else:
            pieces.extend((key, value))
    return tuple(pieces)


def _turn_payloads(row: dict[str, object], *, with_targets: bool) -> list[list[dict[str, object]]]:
    turns = row["inference"]["turns"]
    specs = row["inference"]["turn_specs"]
    label_vocab = [str(label) for label in row["inference"]["response_spec"]["label_vocab"]]
    targets = normalize_labels(row.get("scoring", {}).get("final_probe_targets"), label_vocab) if with_targets else None
    payloads: list[list[dict[str, object]]] = []
    for turn, spec in zip(turns, specs, strict=True):
        kind = str(spec["kind"])
        items = parse_turn_items(turn, kind=kind)
        if kind == "decision" and targets is not None:
            scored_items = []
            for item, target in zip(items, targets, strict=True):
                scored = dict(item)
                scored["label"] = target
                scored_items.append(scored)
            payloads.append(scored_items)
        else:
            payloads.append(items)
    return payloads


def semantic_signature(row: dict[str, object]) -> tuple[object, ...]:
    spec = _response_spec(row)
    label_vocab = [str(label) for label in spec["label_vocab"]]
    targets = normalize_labels(row["scoring"]["final_probe_targets"], label_vocab)
    if targets is None or len(targets) != int(spec["probe_count"]):
        raise RuntimeError(f"row {row.get('episode_id')} has invalid final_probe_targets")
    return (
        str(row["analysis"]["suite_task_id"]),
        tuple(normalized_turn_text(turn) for turn in row["inference"]["turns"]),
        tuple(label_vocab),
        targets,
    )


def structural_signature(row: dict[str, object]) -> tuple[object, ...]:
    label_vocab = [str(label) for label in row["inference"]["response_spec"]["label_vocab"]]
    payloads = _turn_payloads(row, with_targets="scoring" in row)
    field_maps, label_map = _normalize_nominal_maps(payloads, label_vocab)
    return (
        str(row["analysis"]["suite_task_id"]),
        str(row["analysis"]["structure_family_id"]),
        tuple((spec["kind"], spec["item_count"]) for spec in row["inference"]["turn_specs"]),
        tuple(
            tuple(sorted(_canonical_item(item, field_maps=field_maps, label_map=label_map) for item in items))
            for items in payloads
        ),
    )


def structural_case_counter(row: dict[str, object]) -> Counter[tuple[object, ...]]:
    label_vocab = [str(label) for label in row["inference"]["response_spec"]["label_vocab"]]
    payloads = _turn_payloads(row, with_targets="scoring" in row)
    field_maps, label_map = _normalize_nominal_maps(payloads, label_vocab)
    counter: Counter[tuple[object, ...]] = Counter()
    for turn_index, items in enumerate(payloads):
        kind = str(row["inference"]["turn_specs"][turn_index]["kind"])
        for item in items:
            counter[(turn_index, kind, *_canonical_item(item, field_maps=field_maps, label_map=label_map))] += 1
    return counter


def structural_overlap_score(left: dict[str, object], right: dict[str, object]) -> float:
    left_counter = structural_case_counter(left)
    right_counter = structural_case_counter(right)
    total = max(sum(left_counter.values()), sum(right_counter.values()))
    if total == 0:
        return 0.0
    return sum((left_counter & right_counter).values()) / total


def verify_split_isolation(public_rows: list[dict[str, object]], private_rows: list[dict[str, object]]) -> dict[str, object]:
    public_semantic = {semantic_signature(row): str(row["episode_id"]) for row in public_rows}
    public_structural = {structural_signature(row): str(row["episode_id"]) for row in public_rows}
    public_by_task: dict[str, list[dict[str, object]]] = {}
    for row in public_rows:
        public_by_task.setdefault(str(row["analysis"]["suite_task_id"]), []).append(row)

    exact_overlap: list[tuple[str, str]] = []
    structural_overlap: list[tuple[str, str]] = []
    near_duplicate_overlap: list[tuple[str, str, float]] = []

    for row in private_rows:
        private_id = str(row["episode_id"])
        exact = public_semantic.get(semantic_signature(row))
        if exact is not None:
            exact_overlap.append((exact, private_id))
            continue
        structural = public_structural.get(structural_signature(row))
        if structural is not None:
            structural_overlap.append((structural, private_id))
            continue
        suite_task_id = str(row["analysis"]["suite_task_id"])
        best_score = -1.0
        best_public_id = ""
        for public_row in public_by_task.get(suite_task_id, []):
            score = structural_overlap_score(public_row, row)
            if score > best_score:
                best_score = score
                best_public_id = str(public_row["episode_id"])
        if best_score >= PRIVATE_NEAR_DUPLICATE_OVERLAP_THRESHOLD:
            near_duplicate_overlap.append((best_public_id, private_id, round(best_score, 4)))

    if exact_overlap:
        raise RuntimeError(f"public/private semantic overlap detected: {exact_overlap}")
    if structural_overlap:
        raise RuntimeError(f"public/private structural overlap detected: {structural_overlap}")
    if near_duplicate_overlap:
        raise RuntimeError(f"public/private near-duplicate overlap detected: {near_duplicate_overlap}")
    return {
        "exact_public_overlap_count": 0,
        "structural_overlap_count": 0,
        "near_duplicate_overlap_count": 0,
    }


def verify_generator_isolation(
    private_generator_metadata: dict[str, dict[str, str]],
    *,
    public_reference: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, object]:
    if public_reference is None:
        public_reference = public_generator_reference()
    family_ids = sorted({metadata["family_id"] for metadata in private_generator_metadata.values()})
    template_ids = sorted({metadata["template_id"] for metadata in private_generator_metadata.values()})
    operator_class_counts = Counter(
        metadata["operator_class"] for metadata in private_generator_metadata.values()
    )
    public_family_ids = set(public_reference["family_ids"])
    public_template_ids = set(public_reference["template_ids"])
    public_operator_classes = set(public_reference["operator_classes"])
    family_overlap = sorted(set(family_ids) & public_family_ids)
    template_overlap = sorted(set(template_ids) & public_template_ids)
    operator_overlap = sorted(set(operator_class_counts) & public_operator_classes)
    if family_overlap:
        raise RuntimeError(f"public/private generator family_id overlap detected: {family_overlap}")
    if template_overlap:
        raise RuntimeError(f"public/private generator template_id overlap detected: {template_overlap}")
    if operator_overlap:
        raise RuntimeError(f"public/private generator operator_class overlap detected: {operator_overlap}")
    return {
        "family_ids": family_ids,
        "template_ids": template_ids,
        "operator_class_counts": dict(sorted(operator_class_counts.items())),
        "operator_diversity": {
            "distinct_operator_class_count": len(operator_class_counts),
        },
        "public_non_overlap_assertion": {
            "family_ids": True,
            "template_ids": True,
            "operator_classes": True,
        },
    }


def _validate_response_spec(episode_id: str, spec: dict[str, object]) -> tuple[int, list[str]]:
    if spec.get("format") != "ordered_labels":
        raise RuntimeError(f"row {episode_id} has unsupported response format")
    probe_count = spec.get("probe_count")
    label_vocab = spec.get("label_vocab")
    if not isinstance(probe_count, int) or probe_count <= 0:
        raise RuntimeError(f"row {episode_id} has invalid probe_count")
    if not isinstance(label_vocab, list) or len(label_vocab) < 2:
        raise RuntimeError(f"row {episode_id} must declare at least two labels")
    if len({str(label) for label in label_vocab}) != len(label_vocab):
        raise RuntimeError(f"row {episode_id} label_vocab must be unique")
    return probe_count, [str(label) for label in label_vocab]


def _summary_from_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    structure_counts = Counter(str(row["analysis"]["structure_family_id"]) for row in rows)
    turn_counts = Counter(len(row["inference"]["turns"]) for row in rows)
    probe_counts = Counter(int(row["inference"]["response_spec"]["probe_count"]) for row in rows)
    label_vocab_sizes = Counter(len(row["inference"]["response_spec"]["label_vocab"]) for row in rows)
    difficulty_counts = Counter(str(row["analysis"]["difficulty_bin"]) for row in rows)
    optional_field_keys: set[str] = set()
    numeric_r1: list[int] = []
    numeric_r2: list[int] = []
    nominal_values: dict[str, set[str]] = {"shape": set(), "tone": set()}
    for row in rows:
        for turn, spec in zip(row["inference"]["turns"], row["inference"]["turn_specs"], strict=True):
            items = parse_turn_items(turn, kind=str(spec["kind"]))
            for item in items:
                numeric_r1.append(int(item["r1"]))
                numeric_r2.append(int(item["r2"]))
                nominal_values.setdefault("shape", set()).add(str(item["shape"]))
                nominal_values.setdefault("tone", set()).add(str(item["tone"]))
                for key, value in item.items():
                    if key in {"index", "label", "r1", "r2", "shape", "tone", "rule_id"}:
                        continue
                    optional_field_keys.add(key)
                    if isinstance(value, str):
                        nominal_values.setdefault(key, set()).add(value)
    return {
        "difficulty_bin_counts": dict(sorted(difficulty_counts.items())),
        "structure_family_counts": dict(sorted(structure_counts.items())),
        "turn_count_distribution": {str(key): value for key, value in sorted(turn_counts.items())},
        "probe_count_distribution": {str(key): value for key, value in sorted(probe_counts.items())},
        "label_vocab_size_distribution": {str(key): value for key, value in sorted(label_vocab_sizes.items())},
        "stimulus_space_summary": {
            "numeric_range": {
                "r1": {"min": min(numeric_r1), "max": max(numeric_r1)},
                "r2": {"min": min(numeric_r2), "max": max(numeric_r2)},
            },
            "nominal_cardinality": {key: len(values) for key, values in sorted(nominal_values.items())},
            "optional_field_keys": sorted(optional_field_keys),
        },
    }


def verify_schema(rows: list[dict[str, object]], split: str) -> dict[str, object]:
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"{split} rows must be a non-empty JSON list")
    task_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    structure_counts: Counter[str] = Counter()
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError(f"{split} rows must contain objects")
        episode_id = str(row.get("episode_id", ""))
        analysis = row.get("analysis")
        if not isinstance(analysis, dict) or sorted(analysis.keys()) != [
            "difficulty_bin",
            "faculty_id",
            "shift_mode",
            "structure_family_id",
            "suite_task_id",
        ]:
            raise RuntimeError(f"row {episode_id} has invalid analysis keys")
        suite_task_id = str(analysis["suite_task_id"])
        if suite_task_id not in SUITE_TASKS:
            raise RuntimeError(f"row {episode_id} has unsupported suite_task_id {suite_task_id!r}")
        if analysis["faculty_id"] != FACULTY_ID:
            raise RuntimeError(f"row {episode_id} has unsupported faculty_id")
        if analysis["shift_mode"] != SHIFT_MODES[suite_task_id]:
            raise RuntimeError(f"row {episode_id} has mismatched shift_mode")
        if analysis["difficulty_bin"] not in {"hard", "medium"}:
            raise RuntimeError(f"row {episode_id} has unsupported difficulty_bin")

        inference = row.get("inference")
        if not isinstance(inference, dict) or sorted(inference.keys()) != ["response_spec", "turn_specs", "turns"]:
            raise RuntimeError(f"row {episode_id} has invalid inference payload")
        turns = inference.get("turns")
        specs = inference.get("turn_specs")
        if not isinstance(turns, list) or not isinstance(specs, list) or len(turns) != len(specs) or len(turns) < 2:
            raise RuntimeError(f"row {episode_id} has invalid turn layout")
        probe_count, label_vocab = _validate_response_spec(episode_id, inference["response_spec"])
        decision_positions = [index for index, spec in enumerate(specs) if spec.get("kind") == "decision"]
        if decision_positions != [len(specs) - 1]:
            raise RuntimeError(f"row {episode_id} must end with exactly one decision turn")
        for turn_index, (turn, spec) in enumerate(zip(turns, specs, strict=True), start=1):
            if not isinstance(spec, dict) or sorted(spec.keys()) != ["item_count", "kind"]:
                raise RuntimeError(f"row {episode_id} has malformed turn_spec {turn_index}")
            if spec["kind"] not in {"evidence", "decision"} or not isinstance(spec["item_count"], int) or spec["item_count"] <= 0:
                raise RuntimeError(f"row {episode_id} has invalid turn_spec {turn_index}")
            prefix = f"{TURN_HEADER_PREFIX}{episode_id}. Turn {turn_index} of {len(turns)}."
            if not isinstance(turn, str) or not turn.startswith(prefix):
                raise RuntimeError(f"row {episode_id} has malformed turn header {turn_index}")
            items = parse_turn_items(turn, kind=str(spec["kind"]))
            if len(items) != int(spec["item_count"]):
                raise RuntimeError(f"row {episode_id} turn {turn_index} count does not match turn_specs")
            if spec["kind"] == "decision":
                if int(spec["item_count"]) != probe_count:
                    raise RuntimeError(f"row {episode_id} decision turn count mismatches response_spec")
                if response_instruction_from_spec(inference["response_spec"]) not in turn:
                    raise RuntimeError(f"row {episode_id} decision turn must include the response instruction")
            else:
                for item in items:
                    if str(item["label"]) not in label_vocab:
                        raise RuntimeError(f"row {episode_id} evidence turn has label outside label_vocab")

        if split == "public":
            scoring = row.get("scoring")
            if not isinstance(scoring, dict) or sorted(scoring.keys()) != ["final_probe_targets"]:
                raise RuntimeError(f"row {episode_id} has invalid scoring payload")
            targets = normalize_labels(scoring["final_probe_targets"], label_vocab)
            if targets is None or len(targets) != probe_count:
                raise RuntimeError(f"row {episode_id} has invalid final_probe_targets")
        elif "scoring" in row:
            raise RuntimeError(f"row {episode_id} leaks scoring fields in the private split")

        task_counts[suite_task_id] += 1
        difficulty_counts[str(analysis["difficulty_bin"])] += 1
        structure_counts[str(analysis["structure_family_id"])] += 1

    if split == "public":
        if len(rows) != EXPECTED_PUBLIC_ROW_COUNT:
            raise RuntimeError(f"public split must contain {EXPECTED_PUBLIC_ROW_COUNT} rows")
        if task_counts != Counter({suite_task_id: PUBLIC_EPISODES_PER_TASK for suite_task_id in SUITE_TASKS}):
            raise RuntimeError(f"public task counts mismatch: {task_counts}")

    return {
        "row_count": len(rows),
        "suite_task_counts": dict(sorted(task_counts.items())),
        "difficulty_bin_counts": dict(sorted(difficulty_counts.items())),
        "structure_family_counts": dict(sorted(structure_counts.items())),
    }


def response_instruction_from_spec(spec: dict[str, object]) -> str:
    vocab = ", ".join(str(label) for label in spec["label_vocab"])
    return (
        f"Return exactly {spec['probe_count']} outputs in order, one per probe. "
        f"Use only labels from: {vocab}."
    )


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


def verify_public_difficulty_calibration(rows: list[dict[str, object]]) -> dict[str, object]:
    _payload, entries_by_episode = load_public_difficulty_calibration()
    rows_by_id = {str(row["episode_id"]): row for row in rows}
    missing_episode_ids = sorted(set(entries_by_episode) - set(rows_by_id))
    extra_episode_ids = sorted(set(rows_by_id) - set(entries_by_episode))
    if missing_episode_ids or extra_episode_ids:
        raise RuntimeError(
            "public difficulty calibration coverage mismatch: "
            f"missing={missing_episode_ids}, extra={extra_episode_ids}"
        )
    for episode_id, row in rows_by_id.items():
        expected = str(entries_by_episode[episode_id]["difficulty_bin"])
        actual = str(row["analysis"]["difficulty_bin"])
        if actual != expected:
            raise RuntimeError(f"public difficulty calibration mismatch for episode {episode_id}")
    return {
        "difficulty_calibration_path": str(PUBLIC_DIFFICULTY_CALIBRATION_PATH),
        "difficulty_bin_counts": dict(
            sorted(Counter(str(entry["difficulty_bin"]) for entry in entries_by_episode.values()).items())
        ),
    }


def verify_private_empirical_difficulty(
    private_rows: list[dict[str, object]],
    answer_key: dict[str, object],
    predictions_by_model: list[dict[str, object]],
    episode_targets: dict[str, tuple[str, ...]],
) -> dict[str, dict[str, object]]:
    entries_by_episode = empirical_difficulty_entries_from_predictions(episode_targets, predictions_by_model)
    rows_by_id = {str(row["episode_id"]): row for row in private_rows}
    answer_key_by_id = {str(episode["episode_id"]): episode for episode in answer_key["episodes"]}
    missing_row_ids = sorted(set(entries_by_episode) - set(rows_by_id))
    extra_row_ids = sorted(set(rows_by_id) - set(entries_by_episode))
    if missing_row_ids or extra_row_ids:
        raise RuntimeError(
            "private empirical difficulty coverage mismatch for rows: "
            f"missing={missing_row_ids}, extra={extra_row_ids}"
        )
    missing_answer_ids = sorted(set(entries_by_episode) - set(answer_key_by_id))
    extra_answer_ids = sorted(set(answer_key_by_id) - set(entries_by_episode))
    if missing_answer_ids or extra_answer_ids:
        raise RuntimeError(
            "private empirical difficulty coverage mismatch for answer key: "
            f"missing={missing_answer_ids}, extra={extra_answer_ids}"
        )
    for episode_id, entry in entries_by_episode.items():
        expected = str(entry["difficulty_bin"])
        if str(rows_by_id[episode_id]["analysis"]["difficulty_bin"]) != expected:
            raise RuntimeError(f"private empirical difficulty mismatch for row {episode_id}")
        if str(answer_key_by_id[episode_id]["difficulty_bin"]) != expected:
            raise RuntimeError(f"private empirical difficulty mismatch for answer key episode {episode_id}")
    return entries_by_episode


def verify_public_report(payload: dict[str, object], rows: list[dict[str, object]]) -> dict[str, object]:
    if payload.get("version") != PUBLIC_BUNDLE_VERSION:
        raise RuntimeError("public quality report has an unsupported version")
    if payload.get("task_name") != TASK_NAME:
        raise RuntimeError("public quality report task_name mismatch")
    if payload.get("row_count") != len(rows):
        raise RuntimeError("public quality report row_count mismatch")
    expected = _summary_from_rows(rows)
    for key in (
        "difficulty_bin_counts",
        "structure_family_counts",
        "turn_count_distribution",
        "probe_count_distribution",
        "label_vocab_size_distribution",
        "stimulus_space_summary",
    ):
        if payload.get(key) != expected[key]:
            raise RuntimeError(f"public quality report {key} mismatch")
    suite_task_structure_counts = payload.get("suite_task_structure_counts")
    if not isinstance(suite_task_structure_counts, dict):
        raise RuntimeError("public quality report must include suite_task_structure_counts")
    for suite_task_id in SUITE_TASKS:
        per_task = suite_task_structure_counts.get(suite_task_id)
        if not isinstance(per_task, dict) or len(per_task) < 2:
            raise RuntimeError(f"public task {suite_task_id} must use at least two structure families")
    return {
        "row_count": payload["row_count"],
        "structure_family_counts": payload["structure_family_counts"],
        "turn_count_distribution": payload["turn_count_distribution"],
        "probe_count_distribution": payload["probe_count_distribution"],
        "label_vocab_size_distribution": payload["label_vocab_size_distribution"],
    }


def verify_public_split(*, emit_audit_report: Path | None = None) -> None:
    rows = load_rows(PUBLIC_ROWS_PATH)
    schema_summary = verify_schema(rows, "public")
    calibration_summary = verify_public_difficulty_calibration(rows)
    expected_rows, _answers, report = build_public_artifacts()
    if rows != expected_rows:
        raise RuntimeError("public split rows are not reproducible from the generator")
    tracked_report = public_quality_report_payload()
    if tracked_report != report:
        raise RuntimeError("public quality report is not reproducible from the generator")
    report_summary = verify_public_report(tracked_report, rows)
    if emit_audit_report is not None:
        write_audit_report(
            emit_audit_report,
            split="public",
            artifact_digests={
                PUBLIC_ROWS_PATH.name: compute_sha256(PUBLIC_ROWS_PATH),
                PUBLIC_QUALITY_REPORT_PATH.name: compute_sha256(PUBLIC_QUALITY_REPORT_PATH),
                PUBLIC_DIFFICULTY_CALIBRATION_PATH.name: compute_sha256(PUBLIC_DIFFICULTY_CALIBRATION_PATH),
            },
            checks_executed=PUBLIC_AUDIT_CHECKS,
            summary={"row_count": len(rows)},
        )
    print(
        json.dumps(
            {
                "split": "public",
                "rows_path": str(PUBLIC_ROWS_PATH),
                "quality_report_path": str(PUBLIC_QUALITY_REPORT_PATH),
                **schema_summary,
                **calibration_summary,
                **report_summary,
            },
            indent=2,
        )
    )


def private_bundle_paths(bundle_dir: Path) -> dict[str, Path]:
    return {
        "rows": bundle_dir / PRIVATE_ROWS_FILENAME,
        "answer_key": bundle_dir / PRIVATE_ANSWER_KEY_FILENAME,
        "predictions": bundle_dir / PRIVATE_CALIBRATION_PREDICTIONS_FILENAME,
        "manifest": bundle_dir / PRIVATE_RELEASE_MANIFEST_FILENAME,
        "quality": bundle_dir / PRIVATE_QUALITY_REPORT_FILENAME,
    }


def resolve_private_bundle_dir(explicit_path: str | None) -> Path:
    if explicit_path:
        bundle_dir = Path(explicit_path).expanduser().resolve()
    else:
        env_raw = os.environ.get(PRIVATE_BUNDLE_ENV_VAR)
        if not env_raw:
            raise RuntimeError(f"private verification requires --private-bundle-dir or {PRIVATE_BUNDLE_ENV_VAR}")
        bundle_dir = Path(env_raw).expanduser().resolve()
    if not bundle_dir.exists() or not bundle_dir.is_dir():
        raise RuntimeError(f"private bundle directory does not exist: {bundle_dir}")
    return bundle_dir


def load_private_answer_key(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("private answer key payload must be a JSON object")
    if payload.get("version") != PRIVATE_ANSWER_KEY_VERSION:
        raise RuntimeError("private answer key has an unsupported version")
    if payload.get("split") != "private":
        raise RuntimeError("private answer key must declare split='private'")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list):
        raise RuntimeError("private answer key must expose an episodes list")
    return payload


def load_private_calibration_predictions(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("private calibration predictions payload must be a JSON object")
    if payload.get("version") != PRIVATE_CALIBRATION_PREDICTIONS_VERSION:
        raise RuntimeError("private calibration predictions have an unsupported version")
    if payload.get("split") != "private":
        raise RuntimeError("private calibration predictions must declare split='private'")
    models = payload.get("models")
    if not isinstance(models, list):
        raise RuntimeError("private calibration predictions must expose a models list")
    return payload


def verify_private_answer_key(
    payload: dict[str, object],
    private_rows: list[dict[str, object]],
) -> tuple[dict[str, object], dict[str, tuple[str, ...]], dict[str, dict[str, str]]]:
    rows_by_id = {str(row["episode_id"]): row for row in private_rows}
    episode_targets: dict[str, tuple[str, ...]] = {}
    episode_generators: dict[str, dict[str, str]] = {}
    label_counts: Counter[str] = Counter()
    for episode in payload["episodes"]:
        if not isinstance(episode, dict):
            raise RuntimeError("private answer key episodes must be JSON objects")
        episode_id = str(episode.get("episode_id"))
        if episode_id in episode_targets:
            raise RuntimeError(f"private answer key duplicates episode_id {episode_id}")
        row = rows_by_id.get(episode_id)
        if row is None:
            raise RuntimeError(f"private answer key contains unknown episode_id {episode_id}")
        for key in ("faculty_id", "suite_task_id", "shift_mode", "difficulty_bin", "structure_family_id"):
            if episode.get(key) != row["analysis"][key]:
                raise RuntimeError(f"private answer key {key} mismatch for episode {episode_id}")
        if episode.get("inference") != row["inference"]:
            raise RuntimeError(f"private answer key inference mismatch for episode {episode_id}")
        label_vocab = [str(label) for label in row["inference"]["response_spec"]["label_vocab"]]
        targets = normalize_labels(episode.get("final_probe_targets"), label_vocab)
        if targets is None or len(targets) != int(row["inference"]["response_spec"]["probe_count"]):
            raise RuntimeError(f"private answer key episode {episode_id} has invalid final_probe_targets")
        episode_generators[episode_id] = _normalize_generator_metadata(episode, episode_id=episode_id)
        episode_targets[episode_id] = targets
        label_counts.update(targets)
    missing_ids = set(rows_by_id) - set(episode_targets)
    if missing_ids:
        raise RuntimeError(f"private answer key is missing episode_ids: {sorted(missing_ids)}")
    return {
        "answer_key_episode_count": len(episode_targets),
        "answer_key_label_counts": dict(sorted(label_counts.items())),
    }, episode_targets, episode_generators


def verify_private_calibration_predictions(
    payload: dict[str, object],
    private_rows: list[dict[str, object]],
    episode_targets: dict[str, tuple[str, ...]],
) -> list[dict[str, object]]:
    rows_by_id = {str(row["episode_id"]): row for row in private_rows}
    models = payload.get("models")
    if not isinstance(models, list) or len(models) != PRIVATE_PANEL_MODEL_COUNT:
        raise RuntimeError(
            f"private calibration predictions must include exactly {PRIVATE_PANEL_MODEL_COUNT} models"
        )
    normalized_models: list[dict[str, object]] = []
    seen_names: set[str] = set()
    for model in models:
        if not isinstance(model, dict):
            raise RuntimeError("private calibration prediction models must be JSON objects")
        if set(model) != {"episodes", "name"}:
            raise RuntimeError("private calibration prediction models must expose name and episodes")
        name = str(model.get("name", "")).strip()
        if not name:
            raise RuntimeError("private calibration prediction model name must be non-empty")
        if name in seen_names:
            raise RuntimeError(f"private calibration predictions duplicate model name {name}")
        seen_names.add(name)
        episodes = model.get("episodes")
        if not isinstance(episodes, list):
            raise RuntimeError(f"private calibration prediction model {name} must expose an episodes list")
        seen_episode_ids: set[str] = set()
        normalized_episodes: dict[str, tuple[str, ...]] = {}
        for episode in episodes:
            if not isinstance(episode, dict):
                raise RuntimeError(f"private calibration prediction model {name} episodes must be objects")
            if set(episode) != {"episode_id", "predicted_labels"}:
                raise RuntimeError(
                    f"private calibration prediction model {name} episodes must expose episode_id and predicted_labels"
                )
            episode_id = str(episode.get("episode_id", "")).strip()
            if not episode_id:
                raise RuntimeError(f"private calibration prediction model {name} has an empty episode_id")
            if episode_id in seen_episode_ids:
                raise RuntimeError(f"private calibration prediction model {name} duplicates episode_id {episode_id}")
            seen_episode_ids.add(episode_id)
            row = rows_by_id.get(episode_id)
            if row is None:
                raise RuntimeError(f"private calibration prediction model {name} contains unknown episode_id {episode_id}")
            label_vocab = [str(label) for label in row["inference"]["response_spec"]["label_vocab"]]
            predictions = normalize_labels(episode.get("predicted_labels"), label_vocab)
            if predictions is None or len(predictions) != len(episode_targets[episode_id]):
                raise RuntimeError(
                    f"private calibration prediction model {name} episode {episode_id} has invalid predicted_labels"
                )
            normalized_episodes[episode_id] = predictions
        missing_episode_ids = set(rows_by_id) - set(normalized_episodes)
        if missing_episode_ids:
            raise RuntimeError(
                f"private calibration prediction model {name} is missing episode_ids: {sorted(missing_episode_ids)}"
            )
        normalized_models.append({"name": name, "episodes": normalized_episodes})
    return normalized_models


def attach_private_scoring(private_rows: list[dict[str, object]], answer_key: dict[str, object]) -> list[dict[str, object]]:
    _summary, episode_targets, _episode_generators = verify_private_answer_key(answer_key, private_rows)
    attached: list[dict[str, object]] = []
    for row in private_rows:
        episode_id = str(row["episode_id"])
        attached.append(
            {
                "episode_id": row["episode_id"],
                "analysis": dict(row["analysis"]),
                "inference": dict(row["inference"]),
                "scoring": {"final_probe_targets": list(episode_targets[episode_id])},
            }
        )
    return attached


def verify_manifest(path: Path, bundle_paths: dict[str, Path]) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("private release manifest must be a JSON object")
    if payload.get("version") != PRIVATE_BUNDLE_VERSION:
        raise RuntimeError("private release manifest has an unsupported version")
    if payload.get("split") != "private":
        raise RuntimeError("private release manifest must declare split='private'")
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
    return payload


def verify_quality_report(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("private quality report must be a JSON object")
    if payload.get("version") != PRIVATE_QUALITY_REPORT_VERSION:
        raise RuntimeError("private quality report has an unsupported version")
    if payload.get("split") != "private":
        raise RuntimeError("private quality report must declare split='private'")
    if not isinstance(payload.get("row_count"), int) or int(payload["row_count"]) <= 0:
        raise RuntimeError("private quality report row_count must be positive")
    for key in (
        "difficulty_bin_counts",
        "structure_family_counts",
        "turn_count_distribution",
        "probe_count_distribution",
        "label_vocab_size_distribution",
        "stimulus_space_summary",
        "generator_isolation_summary",
    ):
        if key not in payload:
            raise RuntimeError(f"private quality report must include {key}")
    calibration_summary = payload.get("calibration_summary")
    if not isinstance(calibration_summary, dict):
        raise RuntimeError("private quality report must include calibration_summary")
    models = calibration_summary.get("models")
    if not isinstance(models, list) or len(models) != PRIVATE_PANEL_MODEL_COUNT:
        raise RuntimeError(
            f"private quality report calibration_summary must include exactly {PRIVATE_PANEL_MODEL_COUNT} models"
        )
    for index, model in enumerate(models):
        _verify_model_metric_payload(model, context=f"private quality report calibration_summary model {index}")
    attack_suite = payload.get("attack_suite")
    if not isinstance(attack_suite, dict):
        raise RuntimeError("private quality report must include attack_suite")
    if set(attack_suite) != set(ATTACK_SUITE_DIMENSIONS):
        raise RuntimeError("private quality report attack_suite must include only supported dimensions")
    for dimension in ATTACK_SUITE_DIMENSIONS:
        slices = attack_suite[dimension]
        if not isinstance(slices, dict) or not slices:
            raise RuntimeError(f"private quality report attack_suite[{dimension}] must be a non-empty object")
        for slice_name, summary in slices.items():
            if not isinstance(slice_name, str) or not slice_name:
                raise RuntimeError(f"private quality report attack_suite[{dimension}] has an invalid slice name")
            if not isinstance(summary, dict):
                raise RuntimeError(f"private quality report attack_suite[{dimension}][{slice_name}] must be an object")
            if set(summary) != {"models", "row_count"}:
                raise RuntimeError(
                    f"private quality report attack_suite[{dimension}][{slice_name}] must expose row_count and models"
                )
            if not isinstance(summary["row_count"], int) or summary["row_count"] <= 0:
                raise RuntimeError(
                    f"private quality report attack_suite[{dimension}][{slice_name}] row_count must be positive"
                )
            slice_models = summary["models"]
            if not isinstance(slice_models, list) or len(slice_models) != PRIVATE_PANEL_MODEL_COUNT:
                raise RuntimeError(
                    f"private quality report attack_suite[{dimension}][{slice_name}] must include exactly "
                    f"{PRIVATE_PANEL_MODEL_COUNT} models"
                )
            for index, model in enumerate(slice_models):
                _verify_model_metric_payload(
                    model,
                    context=f"private quality report attack_suite[{dimension}][{slice_name}] model {index}",
                )
    isolation = payload.get("semantic_isolation_summary")
    if not isinstance(isolation, dict):
        raise RuntimeError("private quality report must include semantic_isolation_summary")
    generator_isolation = payload.get("generator_isolation_summary")
    if not isinstance(generator_isolation, dict):
        raise RuntimeError("private quality report must include generator_isolation_summary")
    if set(generator_isolation) != {
        "family_ids",
        "operator_class_counts",
        "operator_diversity",
        "public_non_overlap_assertion",
        "template_ids",
    }:
        raise RuntimeError("private quality report generator_isolation_summary has invalid keys")
    for key in ("family_ids", "template_ids"):
        values = generator_isolation.get(key)
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) or not value.strip() for value in values)
        ):
            raise RuntimeError(f"private quality report generator_isolation_summary[{key}] must be a non-empty string list")
    operator_class_counts = generator_isolation.get("operator_class_counts")
    if not isinstance(operator_class_counts, dict) or not operator_class_counts:
        raise RuntimeError("private quality report generator_isolation_summary must include operator_class_counts")
    for operator_class, count in operator_class_counts.items():
        if operator_class not in SUPPORTED_OPERATOR_CLASSES or not isinstance(count, int) or count <= 0:
            raise RuntimeError("private quality report generator_isolation_summary has invalid operator_class_counts")
    operator_diversity = generator_isolation.get("operator_diversity")
    if (
        not isinstance(operator_diversity, dict)
        or set(operator_diversity) != {"distinct_operator_class_count"}
        or not isinstance(operator_diversity.get("distinct_operator_class_count"), int)
        or int(operator_diversity["distinct_operator_class_count"]) <= 0
    ):
        raise RuntimeError("private quality report generator_isolation_summary must include operator_diversity")
    if int(operator_diversity["distinct_operator_class_count"]) != len(operator_class_counts):
        raise RuntimeError(
            "private quality report generator_isolation_summary operator_diversity must match operator_class_counts"
        )
    public_non_overlap_assertion = generator_isolation.get("public_non_overlap_assertion")
    if (
        not isinstance(public_non_overlap_assertion, dict)
        or set(public_non_overlap_assertion) != {"family_ids", "operator_classes", "template_ids"}
        or any(value is not True for value in public_non_overlap_assertion.values())
    ):
        raise RuntimeError(
            "private quality report generator_isolation_summary must assert non-overlap with public generator metadata"
        )
    return payload


def verify_private_bundle(bundle_dir: Path, *, emit_audit_report: Path | None = None) -> None:
    bundle_paths = private_bundle_paths(bundle_dir)
    missing = [name for name, path in bundle_paths.items() if not path.exists()]
    if missing:
        raise RuntimeError(f"private bundle missing required files: {missing}")
    private_rows = load_rows(bundle_paths["rows"])
    schema_summary = verify_schema(private_rows, "private")
    answer_key = load_private_answer_key(bundle_paths["answer_key"])
    answer_summary, _episode_targets, _episode_generators = verify_private_answer_key(answer_key, private_rows)
    predictions = load_private_calibration_predictions(bundle_paths["predictions"])
    prediction_models = verify_private_calibration_predictions(predictions, private_rows, _episode_targets)
    verify_private_empirical_difficulty(private_rows, answer_key, prediction_models, _episode_targets)
    manifest = verify_manifest(bundle_paths["manifest"], bundle_paths)
    quality_report = verify_quality_report(bundle_paths["quality"])
    public_rows = load_rows(PUBLIC_ROWS_PATH)
    expected_quality_report = build_private_quality_report(
        private_rows,
        answer_key,
        predictions,
        public_rows=public_rows,
    )
    if quality_report.get("row_count") != expected_quality_report["row_count"]:
        raise RuntimeError("private quality report row_count mismatch")
    for key in (
        "difficulty_bin_counts",
        "structure_family_counts",
        "turn_count_distribution",
        "probe_count_distribution",
        "label_vocab_size_distribution",
        "stimulus_space_summary",
        "calibration_summary",
        "attack_suite",
        "semantic_isolation_summary",
        "generator_isolation_summary",
    ):
        if quality_report.get(key) != expected_quality_report[key]:
            raise RuntimeError(f"private quality report {key} mismatch")
    missing_families = [
        family_id
        for family_id in REQUIRED_PRIVATE_STRUCTURE_FAMILY_IDS
        if expected_quality_report["structure_family_counts"].get(family_id, 0) <= 0
    ]
    if missing_families:
        raise RuntimeError(f"private quality report is missing required structure families: {missing_families}")
    if emit_audit_report is not None:
        manifest_digests = manifest.get("sha256")
        if not isinstance(manifest_digests, dict):
            raise RuntimeError("private release manifest must expose sha256 digests")
        write_audit_report(
            emit_audit_report,
            split="private",
            artifact_digests={
                PRIVATE_ROWS_FILENAME: str(manifest_digests[PRIVATE_ROWS_FILENAME]),
                PRIVATE_ANSWER_KEY_FILENAME: str(manifest_digests[PRIVATE_ANSWER_KEY_FILENAME]),
                PRIVATE_CALIBRATION_PREDICTIONS_FILENAME: str(manifest_digests[PRIVATE_CALIBRATION_PREDICTIONS_FILENAME]),
                PRIVATE_QUALITY_REPORT_FILENAME: str(manifest_digests[PRIVATE_QUALITY_REPORT_FILENAME]),
            },
            checks_executed=PRIVATE_AUDIT_CHECKS,
            summary={"row_count": len(private_rows)},
        )
    print(
        json.dumps(
            {
                "split": "private",
                "bundle_dir": str(bundle_dir),
                **schema_summary,
                **answer_summary,
                "prediction_model_names": [model["name"] for model in prediction_models],
                "semantic_isolation_summary": expected_quality_report["semantic_isolation_summary"],
                "generator_isolation_summary": expected_quality_report["generator_isolation_summary"],
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("public", "private"), required=True)
    parser.add_argument("--private-bundle-dir")
    parser.add_argument("--emit-audit-report")
    args = parser.parse_args()
    audit_report_path = Path(args.emit_audit_report).expanduser() if args.emit_audit_report else None
    if args.split == "public":
        verify_public_split(emit_audit_report=audit_report_path)
        return
    verify_private_bundle(
        resolve_private_bundle_dir(args.private_bundle_dir),
        emit_audit_report=audit_report_path,
    )


if __name__ == "__main__":
    main()
