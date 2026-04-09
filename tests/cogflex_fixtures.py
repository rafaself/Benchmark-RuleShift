from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from scripts.build_cogflex_dataset import (
    FACULTY_ID,
    PRIVATE_ANSWER_KEY_FILENAME,
    PRIVATE_ANSWER_KEY_VERSION,
    PRIVATE_BUNDLE_VERSION,
    PRIVATE_CALIBRATION_PREDICTIONS_FILENAME,
    PRIVATE_CALIBRATION_PREDICTIONS_VERSION,
    PRIVATE_QUALITY_REPORT_FILENAME,
    PRIVATE_QUALITY_REPORT_VERSION,
    PRIVATE_RELEASE_MANIFEST_FILENAME,
    PRIVATE_ROWS_FILENAME,
    PRIVATE_GENERATOR_OPERATOR_CLASS_BY_STRUCTURE,
    REQUIRED_PRIVATE_STRUCTURE_FAMILY_IDS,
    SHIFT_MODES,
    SUITE_TASKS,
    EpisodeStructure,
    build_domain,
    build_episode_payload,
    build_public_artifacts,
    compute_sha256,
    empirical_difficulty_entries_from_predictions,
    enumerate_items,
    make_three_label_rule,
    make_two_label_rule,
    sample_mixed_route_examples,
    sample_for_rule,
)
from scripts.verify_cogflex import build_private_quality_report


PRIVATE_DOMAIN = build_domain(
    tuple(range(-9, 10)),
    ("diamond", "pentagon", "kite", "oval"),
    ("bright", "dim", "muted"),
    extras={
        "pattern": ("striped", "dotted", "grain"),
        "channel": ("aurora", "signal", "hinge"),
    },
)

PRIVATE_RULES = {
    "orbit_anchor_signal": make_two_label_rule(
        "orbit_anchor_signal",
        "private_router",
        ("orbit", "anchor"),
        "orbit when channel=signal or r1>0",
        lambda stimulus: str(stimulus["channel"]) == "signal" or int(stimulus["r1"]) > 0,
    ),
    "orbit_anchor_pattern": make_two_label_rule(
        "orbit_anchor_pattern",
        "private_router",
        ("orbit", "anchor"),
        "orbit when pattern is striped or r2 is even",
        lambda stimulus: str(stimulus["pattern"]) == "striped" or int(stimulus["r2"]) % 2 == 0,
    ),
    "ember_mist_tide_band": make_three_label_rule(
        "ember_mist_tide_band",
        "private_tri_band",
        ("ember", "mist", "tide"),
        "band over r1-r2 with distractors",
        lambda stimulus: (
            "ember"
            if int(stimulus["r1"]) - int(stimulus["r2"]) <= -4
            else "tide"
            if int(stimulus["r1"]) - int(stimulus["r2"]) >= 4
            else "mist"
        ),
    ),
    "ember_mist_tide_symbolic": make_three_label_rule(
        "ember_mist_tide_symbolic",
        "private_tri_band",
        ("ember", "mist", "tide"),
        "pattern/tone symbolic router",
        lambda stimulus: (
            "ember"
            if str(stimulus["pattern"]) == "dotted"
            else "tide"
            if str(stimulus["tone"]) == "bright"
            else "mist"
        ),
    ),
}

PRIVATE_STRUCTURES = {
    "delayed_reversal": EpisodeStructure("delayed_reversal", (3, 2, 3), 5),
    "irrelevant_feature_interference": EpisodeStructure("irrelevant_feature_interference", (4, 4), 6),
    "competitive_rule_switch": EpisodeStructure("competitive_rule_switch", (3, 3, 3), 6),
    "latent_rebinding": EpisodeStructure("latent_rebinding", (4, 2, 3), 5),
    "variable_evidence_budget": EpisodeStructure("variable_evidence_budget", (2, 5), 7),
    "interleaved_context_rebinding": EpisodeStructure("interleaved_context_rebinding", (2, 2, 2, 2), 6),
}
PRIVATE_PANEL_MODEL_NAMES = ("panel-model-a", "panel-model-b", "panel-model-c")

PRIVATE_GENERATOR_FAMILY_BY_STRUCTURE = {
    "delayed_reversal": "private::reversal_family",
    "irrelevant_feature_interference": "private::interference_family",
    "competitive_rule_switch": "private::competition_family",
    "latent_rebinding": "private::rebinding_family",
    "variable_evidence_budget": "private::budget_family",
    "interleaved_context_rebinding": "private::interleaved_rebinding_family",
}


def public_fixture() -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    return build_public_artifacts()


def _private_row_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    difficulty_counts = Counter(str(row["analysis"]["difficulty_bin"]) for row in rows)
    structure_counts = Counter(str(row["analysis"]["structure_family_id"]) for row in rows)
    turn_counts = Counter(len(row["inference"]["turns"]) for row in rows)
    probe_counts = Counter(int(row["inference"]["response_spec"]["probe_count"]) for row in rows)
    label_vocab_sizes = Counter(len(row["inference"]["response_spec"]["label_vocab"]) for row in rows)
    numeric_r1: list[int] = []
    numeric_r2: list[int] = []
    nominal_values: dict[str, set[str]] = {}
    optional_keys: set[str] = set()
    for row in rows:
        for turn, spec in zip(row["inference"]["turns"], row["inference"]["turn_specs"], strict=True):
            from scripts.build_cogflex_dataset import parse_turn_items

            for item in parse_turn_items(turn, kind=str(spec["kind"])):
                numeric_r1.append(int(item["r1"]))
                numeric_r2.append(int(item["r2"]))
                for key, value in item.items():
                    if key in {"index", "label", "rule_id"}:
                        continue
                    if isinstance(value, str):
                        nominal_values.setdefault(key, set()).add(value)
                    if key not in {"r1", "r2", "shape", "tone"}:
                        optional_keys.add(key)
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
            "optional_field_keys": sorted(optional_keys),
        },
    }


def _build_private_episode(
    episode_id: str,
    suite_task_id: str,
    structure_family_id: str,
    variant: int,
) -> tuple[dict[str, object], dict[str, object]]:
    import random

    seed = variant * 97 + len(episode_id) * 11
    rng = random.Random(seed)
    structure = PRIVATE_STRUCTURES[structure_family_id]
    if structure_family_id in {"delayed_reversal", "competitive_rule_switch", "variable_evidence_budget"}:
        initial_rule = PRIVATE_RULES["orbit_anchor_signal"]
        shift_rule = PRIVATE_RULES["orbit_anchor_pattern"]
    else:
        initial_rule = PRIVATE_RULES["ember_mist_tide_band"]
        shift_rule = PRIVATE_RULES["ember_mist_tide_symbolic"]
    label_vocab = initial_rule.label_vocab
    used: set[tuple[object, ...]] = set()
    turn_items: list[list[dict[str, object]]] = []
    prompts: list[str] = []

    if structure_family_id == "interleaved_context_rebinding":
        context_terms = ("mesa", "fjord")
        prompts = [
            "Use the context tag to track which routing rule generated each labeled example.",
            "Keep both context-specific routing hypotheses active as more interleaved evidence arrives.",
            "Reconcile the alternating contexts without dropping either routing rule.",
            "Use the interleaved contexts to maintain the current routing map.",
            "Classify each probe using its context to select the implied routing behavior.",
        ]
        for evidence_index, count in enumerate(structure.evidence_counts):
            mixed = sample_mixed_route_examples(
                rng,
                PRIVATE_DOMAIN,
                [
                    (context_terms[0], initial_rule, "primary"),
                    (context_terms[1], shift_rule, "secondary"),
                ],
                count,
                route_key="context",
                exclude=used,
            )
            used.update(
                tuple((key, item[key]) for key in sorted(item) if key not in {"context", "index", "label", "rule_id"})
                for item in mixed
            )
            turn_items.append(mixed)
        probes = sample_mixed_route_examples(
            rng,
            PRIVATE_DOMAIN,
            [
                (context_terms[0], initial_rule, "primary"),
                (context_terms[1], shift_rule, "secondary"),
            ],
            structure.probe_count,
            route_key="context",
            exclude=used,
        )
        turn_items.append(probes)
        row, answer = build_episode_payload(
            episode_id,
            suite_task_id=suite_task_id,
            structure=structure,
            label_vocab=label_vocab,
            turn_prompts=prompts,
            turn_items=turn_items,
        )
        row["analysis"]["structure_family_id"] = structure_family_id
        answer["analysis"]["structure_family_id"] = structure_family_id
        return row, answer

    for evidence_index, count in enumerate(structure.evidence_counts):
        prompts.append(
            [
                "Learn the active private routing pattern from these examples.",
                "Update your routing hypothesis using the new evidence.",
                "Revisit the earlier evidence in light of these labels.",
            ][evidence_index % 3]
        )
        current_rule = initial_rule
        if structure_family_id in {"delayed_reversal", "competitive_rule_switch"} and evidence_index == len(structure.evidence_counts) - 1:
            current_rule = shift_rule
        if structure_family_id == "latent_rebinding" and evidence_index >= 1:
            current_rule = shift_rule
        sampled = sample_for_rule(
            rng,
            PRIVATE_DOMAIN,
            current_rule,
            count,
            exclude=used,
            rotation=variant + evidence_index,
            mismatch_rule=initial_rule if current_rule is shift_rule else None,
            min_mismatch=1 if current_rule is shift_rule else 0,
        )
        for stimulus in sampled:
            used.add(tuple((key, stimulus[key]) for key in sorted(stimulus)))
        items = enumerate_items(sampled, current_rule)
        if structure_family_id in {"competitive_rule_switch"}:
            cue_value = "copper" if evidence_index % 2 == 0 else "silver"
            for item in items:
                item["cue"] = cue_value
        if structure_family_id in {"latent_rebinding"}:
            context_value = "mesa" if evidence_index % 2 == 0 else "fjord"
            for item in items:
                item["context"] = context_value
        turn_items.append(items)

    decision_rule = shift_rule if structure_family_id != "irrelevant_feature_interference" else initial_rule
    prompts.append("Classify each probe using the currently implied routing behavior.")
    probes = sample_for_rule(
        rng,
        PRIVATE_DOMAIN,
        decision_rule,
        structure.probe_count,
        exclude=used,
        rotation=variant + 9,
        mismatch_rule=initial_rule if decision_rule is shift_rule else None,
        min_mismatch=1 if decision_rule is shift_rule else 0,
    )
    probe_items = enumerate_items(probes, decision_rule)
    if structure_family_id in {"competitive_rule_switch"}:
        for index, item in enumerate(probe_items):
            item["cue"] = "copper" if index % 2 == 0 else "silver"
    if structure_family_id in {"latent_rebinding"}:
        for index, item in enumerate(probe_items):
            item["context"] = "mesa" if index % 2 == 0 else "fjord"
    turn_items.append(probe_items)

    row, answer = build_episode_payload(
        episode_id,
        suite_task_id=suite_task_id,
        structure=structure,
        label_vocab=label_vocab,
        turn_prompts=prompts,
        turn_items=turn_items,
    )
    row["analysis"]["structure_family_id"] = structure_family_id
    answer["analysis"]["structure_family_id"] = structure_family_id
    return row, answer


def _private_generator_metadata(structure_family_id: str, suite_task_id: str) -> dict[str, str]:
    return {
        "family_id": PRIVATE_GENERATOR_FAMILY_BY_STRUCTURE[structure_family_id],
        "template_id": f"private::{structure_family_id}::{suite_task_id}",
        "operator_class": PRIVATE_GENERATOR_OPERATOR_CLASS_BY_STRUCTURE[structure_family_id],
    }


def _rotate_label(target: str, label_vocab: list[str], step: int) -> str:
    label_index = label_vocab.index(target)
    return label_vocab[(label_index + step) % len(label_vocab)]


def _predicted_labels_for_model(
    *,
    model_name: str,
    episode_index: int,
    answer: dict[str, object],
) -> list[str]:
    label_vocab = [str(label) for label in answer["inference"]["response_spec"]["label_vocab"]]
    targets = [str(label) for label in answer["final_probe_targets"]]
    predictions: list[str] = []
    for probe_index, target in enumerate(targets):
        miss = False
        step = 1
        if model_name == "panel-model-a":
            miss = (episode_index + probe_index) % 5 == 0
        elif model_name == "panel-model-b":
            miss = (episode_index + probe_index) % 3 == 0
        else:
            miss = (episode_index + probe_index) % 2 == 0
            step = 2 if len(label_vocab) > 2 else 1
        predictions.append(_rotate_label(target, label_vocab, step) if miss else target)
    return predictions


def _build_private_predictions_payload(answers: list[dict[str, object]]) -> dict[str, object]:
    return {
        "version": PRIVATE_CALIBRATION_PREDICTIONS_VERSION,
        "split": "private",
        "models": [
            {
                "name": model_name,
                "episodes": [
                    {
                        "episode_id": answer["episode_id"],
                        "predicted_labels": _predicted_labels_for_model(
                            model_name=model_name,
                            episode_index=episode_index,
                            answer=answer,
                        ),
                    }
                    for episode_index, answer in enumerate(answers)
                ],
            }
            for model_name in PRIVATE_PANEL_MODEL_NAMES
        ],
    }


def write_private_bundle(bundle_dir: Path) -> dict[str, Path]:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    answers: list[dict[str, object]] = []
    episode_number = 1
    for family_index, structure_family_id in enumerate(REQUIRED_PRIVATE_STRUCTURE_FAMILY_IDS):
        for task_index, suite_task_id in enumerate(SUITE_TASKS):
            episode_id = f"p{episode_number:04d}"
            row, answer = _build_private_episode(
                episode_id,
                suite_task_id,
                structure_family_id,
                variant=family_index * 10 + task_index,
            )
            rows.append(row)
            answers.append(answer)
            episode_number += 1

    rows_path = bundle_dir / PRIVATE_ROWS_FILENAME
    answer_key_path = bundle_dir / PRIVATE_ANSWER_KEY_FILENAME
    predictions_path = bundle_dir / PRIVATE_CALIBRATION_PREDICTIONS_FILENAME
    quality_path = bundle_dir / PRIVATE_QUALITY_REPORT_FILENAME
    manifest_path = bundle_dir / PRIVATE_RELEASE_MANIFEST_FILENAME
    predictions_payload = _build_private_predictions_payload(answers)
    normalized_models = [
        {
            "name": str(model["name"]),
            "episodes": {
                str(episode["episode_id"]): tuple(str(label) for label in episode["predicted_labels"])
                for episode in model["episodes"]
            },
        }
        for model in predictions_payload["models"]
    ]
    episode_targets = {
        str(answer["episode_id"]): tuple(str(label) for label in answer["final_probe_targets"])
        for answer in answers
    }
    difficulty_entries = empirical_difficulty_entries_from_predictions(episode_targets, normalized_models)
    for row in rows:
        row["analysis"]["difficulty_bin"] = str(difficulty_entries[str(row["episode_id"])]["difficulty_bin"])
    for answer in answers:
        answer["analysis"]["difficulty_bin"] = str(difficulty_entries[str(answer["episode_id"])]["difficulty_bin"])
    private_rows = [
        {
            "episode_id": row["episode_id"],
            "analysis": row["analysis"],
            "inference": row["inference"],
        }
        for row in rows
    ]
    private_answers = [
        {
            "episode_id": answer["episode_id"],
            "faculty_id": answer["analysis"]["faculty_id"],
            "suite_task_id": answer["analysis"]["suite_task_id"],
            "shift_mode": answer["analysis"]["shift_mode"],
            "difficulty_bin": answer["analysis"]["difficulty_bin"],
            "structure_family_id": answer["analysis"]["structure_family_id"],
            "generator": _private_generator_metadata(
                str(answer["analysis"]["structure_family_id"]),
                str(answer["analysis"]["suite_task_id"]),
            ),
            "inference": answer["inference"],
            "final_probe_targets": answer["final_probe_targets"],
        }
        for answer in answers
    ]

    rows_path.write_text(json.dumps(private_rows, indent=2) + "\n", encoding="utf-8")
    answer_key_path.write_text(
        json.dumps(
            {
                "version": PRIVATE_ANSWER_KEY_VERSION,
                "split": "private",
                "episodes": private_answers,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    predictions_path.write_text(json.dumps(predictions_payload, indent=2) + "\n", encoding="utf-8")
    quality_payload = build_private_quality_report(
        private_rows,
        {
            "version": PRIVATE_ANSWER_KEY_VERSION,
            "split": "private",
            "episodes": private_answers,
        },
        predictions_payload,
    )
    quality_path.write_text(json.dumps(quality_payload, indent=2) + "\n", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "version": PRIVATE_BUNDLE_VERSION,
                "split": "private",
                "row_count": len(private_rows),
                "sha256": {
                    PRIVATE_ROWS_FILENAME: compute_sha256(rows_path),
                    PRIVATE_ANSWER_KEY_FILENAME: compute_sha256(answer_key_path),
                    PRIVATE_CALIBRATION_PREDICTIONS_FILENAME: compute_sha256(predictions_path),
                    PRIVATE_QUALITY_REPORT_FILENAME: compute_sha256(quality_path),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "rows": rows_path,
        "answer_key": answer_key_path,
        "predictions": predictions_path,
        "quality": quality_path,
        "manifest": manifest_path,
    }
