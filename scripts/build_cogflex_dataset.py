#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Final

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROWS_PATH = ROOT / "kaggle/dataset/public/public_leaderboard_rows.json"
PUBLIC_METADATA_PATH = ROOT / "kaggle/dataset/public/dataset-metadata.json"
PUBLIC_QUALITY_REPORT_PATH = ROOT / "kaggle/dataset/public/public_quality_report.json"
PUBLIC_DIFFICULTY_CALIBRATION_PATH = ROOT / "kaggle/dataset/public/public_difficulty_calibration.json"

PUBLIC_DATASET_ID = "raptorengineer/cogflex-suite-runtime"
PRIVATE_DATASET_ID = "raptorengineer/cogflex-suite-runtime-private"
NOTEBOOK_ID = "raptorengineer/cogflex-suite-notebook"
TASK_NAME = "cogflex_suite_flexible"
FACULTY_ID = "executive_functions/cognitive_flexibility"

PUBLIC_ROWS_FILENAME = "public_leaderboard_rows.json"
PUBLIC_DIFFICULTY_CALIBRATION_FILENAME = "public_difficulty_calibration.json"
PRIVATE_ROWS_FILENAME = "private_leaderboard_rows.json"
PRIVATE_ANSWER_KEY_FILENAME = "private_answer_key.json"
PRIVATE_CALIBRATION_PREDICTIONS_FILENAME = "private_calibration_predictions.json"
PRIVATE_RELEASE_MANIFEST_FILENAME = "private_release_manifest.json"
PRIVATE_QUALITY_REPORT_FILENAME = "private_quality_report.json"
PRIVATE_BUNDLE_ENV_VAR = "COGFLEX_PRIVATE_BUNDLE_DIR"

PUBLIC_BUNDLE_VERSION = "cogflex_public"
PUBLIC_DIFFICULTY_CALIBRATION_VERSION = "cogflex_public_difficulty"
PRIVATE_BUNDLE_VERSION = "cogflex_private_bundle"
PRIVATE_QUALITY_REPORT_VERSION = "cogflex_private_quality"
PRIVATE_ANSWER_KEY_VERSION = "cogflex_private_answer_key"
PRIVATE_CALIBRATION_PREDICTIONS_VERSION = "cogflex_private_predictions"

PUBLIC_EPISODES_PER_TASK = 30
EXPECTED_PUBLIC_ROW_COUNT = PUBLIC_EPISODES_PER_TASK * 4

TURN_HEADER_PREFIX = "CogFlex suite task. Episode "
LINE_RE = re.compile(r"^(?P<index>\d+)\.\s+(?P<body>.+?)\s+->\s+(?P<label>[a-z0-9_:-]+|\?)$")
POINT_RE = re.compile(r"^r1=(?P<r1>[+-]\d+),\s*r2=(?P<r2>[+-]\d+)$")
RETRIEVAL_HEADER_RE = re.compile(r"Episode \d+")

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

PUBLIC_GENERATOR_OPERATOR_CLASS_BY_TASK: Final[dict[str, str]] = {
    "explicit_rule_update": "explicit_rule_update",
    "latent_rule_update": "latent_rule_update",
    "context_binding": "context_binding",
    "trial_cued_switch": "trial_cued_switch",
}

PUBLIC_STRUCTURE_FAMILY_IDS: Final[tuple[str, ...]] = (
    "two_step_focus",
    "three_step_bridge",
)

REQUIRED_PRIVATE_STRUCTURE_FAMILY_IDS: Final[tuple[str, ...]] = (
    "delayed_reversal",
    "irrelevant_feature_interference",
    "competitive_rule_switch",
    "latent_rebinding",
    "variable_evidence_budget",
    "interleaved_context_rebinding",
)

PRIVATE_GENERATOR_OPERATOR_CLASS_BY_STRUCTURE: Final[dict[str, str]] = {
    "delayed_reversal": "delayed_reversal",
    "irrelevant_feature_interference": "irrelevant_feature_interference",
    "competitive_rule_switch": "competitive_rule_switch",
    "latent_rebinding": "latent_rebinding",
    "variable_evidence_budget": "variable_evidence_budget",
    "interleaved_context_rebinding": "interleaved_context_rebinding",
}

SUPPORTED_OPERATOR_CLASSES: Final[tuple[str, ...]] = tuple(
    sorted(
        {
            *PUBLIC_GENERATOR_OPERATOR_CLASS_BY_TASK.values(),
            *PRIVATE_GENERATOR_OPERATOR_CLASS_BY_STRUCTURE.values(),
        }
    )
)

IDENTIFIABILITY_KIND_SINGLE_LAST: Final[str] = "single_rule_last_turn"
IDENTIFIABILITY_KIND_SINGLE_ALL: Final[str] = "single_rule_all_turns"
IDENTIFIABILITY_KIND_ROUTED_ALL: Final[str] = "routed_all_turns"

PUBLIC_IDENTIFIABILITY_SPEC_BY_TASK: Final[dict[str, tuple[str, str | None]]] = {
    "explicit_rule_update": (IDENTIFIABILITY_KIND_SINGLE_LAST, None),
    "latent_rule_update": (IDENTIFIABILITY_KIND_SINGLE_LAST, None),
    "context_binding": (IDENTIFIABILITY_KIND_ROUTED_ALL, "context"),
    "trial_cued_switch": (IDENTIFIABILITY_KIND_ROUTED_ALL, "cue"),
}

PRIVATE_IDENTIFIABILITY_SPEC_BY_STRUCTURE: Final[dict[str, tuple[str, str | None]]] = {
    "delayed_reversal": (IDENTIFIABILITY_KIND_SINGLE_LAST, None),
    "irrelevant_feature_interference": (IDENTIFIABILITY_KIND_SINGLE_ALL, None),
    "competitive_rule_switch": (IDENTIFIABILITY_KIND_SINGLE_LAST, None),
    "latent_rebinding": (IDENTIFIABILITY_KIND_SINGLE_LAST, None),
    "variable_evidence_budget": (IDENTIFIABILITY_KIND_SINGLE_LAST, None),
    "interleaved_context_rebinding": (IDENTIFIABILITY_KIND_ROUTED_ALL, "context"),
}

IDENTIFIABILITY_RETRY_BUDGET: Final[int] = 32

PUBLIC_CONTEXT_TERMS: Final[tuple[tuple[str, str], ...]] = (
    ("alpha", "beta"),
    ("north", "south"),
    ("cedar", "harbor"),
)

PUBLIC_CUE_TERMS: Final[tuple[tuple[str, str], ...]] = (
    ("keep", "switch"),
    ("stone", "ripple"),
    ("torch", "glint"),
)

PUBLIC_VALUES: Final[tuple[int, ...]] = tuple(range(-5, 6))
PUBLIC_SHAPES: Final[tuple[str, ...]] = ("circle", "triangle", "square", "hex")
PUBLIC_TONES: Final[tuple[str, ...]] = ("warm", "cool", "neutral")


Stimulus = dict[str, object]


@dataclass(frozen=True)
class RuleSpec:
    rule_id: str
    family_id: str
    label_vocab: tuple[str, ...]
    description: str
    resolver: Callable[[Stimulus], str]

    def label(self, stimulus: Stimulus) -> str:
        return self.resolver(stimulus)


@dataclass(frozen=True)
class EpisodeStructure:
    structure_family_id: str
    evidence_counts: tuple[int, ...]
    probe_count: int

    @property
    def turn_count(self) -> int:
        return len(self.evidence_counts) + 1


def empirical_difficulty_entries_from_scores(scores_by_episode: dict[str, float]) -> dict[str, dict[str, object]]:
    """Build ranked empirical difficulty entries from per-episode scores.

    Args:
        scores_by_episode: Mean panel accuracy keyed by episode ID.

    Returns:
        A mapping from episode ID to ranked difficulty metadata.

    Raises:
        RuntimeError: If no episode scores are provided.

    """
    if not scores_by_episode:
        raise RuntimeError("empirical difficulty calibration requires at least one episode")
    ranked = sorted(scores_by_episode.items(), key=lambda item: (item[1], item[0]))
    hard_count = len(ranked) // 2
    assignments: dict[str, dict[str, object]] = {}
    for index, (episode_id, score) in enumerate(ranked, start=1):
        assignments[episode_id] = {
            "panel_mean_accuracy": round(float(score), 6),
            "difficulty_bin": "hard" if index <= hard_count else "medium",
            "rank": index,
        }
    return assignments


def empirical_difficulty_scores_from_predictions(
    episode_targets: dict[str, tuple[str, ...]],
    predictions_by_model: list[dict[str, object]],
) -> dict[str, float]:
    """Compute mean per-episode accuracy from model predictions.

    Args:
        episode_targets: Gold labels keyed by episode ID.
        predictions_by_model: Normalized calibration predictions for each model.

    Returns:
        Mean panel accuracy keyed by episode ID.

    Raises:
        RuntimeError: If no model predictions are provided.

    """
    if not predictions_by_model:
        raise RuntimeError("empirical difficulty calibration requires at least one model")
    scores_by_episode: dict[str, float] = {}
    for episode_id, targets in episode_targets.items():
        per_model_accuracy: list[float] = []
        for model in predictions_by_model:
            predictions = model["episodes"][episode_id]
            correct = sum(1 for predicted, target in zip(predictions, targets, strict=True) if predicted == target)
            per_model_accuracy.append(correct / len(targets))
        scores_by_episode[episode_id] = sum(per_model_accuracy) / len(per_model_accuracy)
    return scores_by_episode


def empirical_difficulty_entries_from_predictions(
    episode_targets: dict[str, tuple[str, ...]],
    predictions_by_model: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    """Build empirical difficulty entries from prediction payloads.

    Args:
        episode_targets: Gold labels keyed by episode ID.
        predictions_by_model: Normalized calibration predictions for each model.

    Returns:
        Ranked empirical difficulty metadata keyed by episode ID.

    """
    return empirical_difficulty_entries_from_scores(
        empirical_difficulty_scores_from_predictions(episode_targets, predictions_by_model)
    )


def public_difficulty_calibration_payload_from_entries(
    entries_by_episode: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Serialize public empirical difficulty entries into the tracked payload.

    Args:
        entries_by_episode: Difficulty metadata keyed by episode ID.

    Returns:
        The public difficulty calibration payload ready for JSON serialization.

    """
    return {
        "version": PUBLIC_DIFFICULTY_CALIBRATION_VERSION,
        "policy": "median_split",
        "score_kind": "mean_panel_episode_accuracy",
        "episodes": [
            {
                "episode_id": episode_id,
                "panel_mean_accuracy": entry["panel_mean_accuracy"],
                "difficulty_bin": entry["difficulty_bin"],
                "rank": entry["rank"],
            }
            for episode_id, entry in sorted(entries_by_episode.items(), key=lambda item: int(item[1]["rank"]))
        ],
    }


def load_public_difficulty_calibration(
    path: Path = PUBLIC_DIFFICULTY_CALIBRATION_PATH,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    """Load and validate the tracked public difficulty calibration file.

    Args:
        path: Location of the calibration JSON artifact.

    Returns:
        The raw payload and a normalized mapping keyed by episode ID.

    Raises:
        RuntimeError: If the payload schema or metadata is invalid.

    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("public difficulty calibration must be a JSON object")
    if payload.get("version") != PUBLIC_DIFFICULTY_CALIBRATION_VERSION:
        raise RuntimeError("public difficulty calibration has an unsupported version")
    if payload.get("policy") != "median_split":
        raise RuntimeError("public difficulty calibration must declare policy='median_split'")
    if payload.get("score_kind") != "mean_panel_episode_accuracy":
        raise RuntimeError("public difficulty calibration must declare score_kind='mean_panel_episode_accuracy'")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise RuntimeError("public difficulty calibration must expose a non-empty episodes list")
    entries_by_episode: dict[str, dict[str, object]] = {}
    seen_ranks: set[int] = set()
    for episode in episodes:
        if not isinstance(episode, dict):
            raise RuntimeError("public difficulty calibration episodes must be objects")
        if set(episode) != {"difficulty_bin", "episode_id", "panel_mean_accuracy", "rank"}:
            raise RuntimeError("public difficulty calibration episodes must expose episode_id, panel_mean_accuracy, difficulty_bin, and rank")
        episode_id = str(episode.get("episode_id", "")).strip()
        if not episode_id:
            raise RuntimeError("public difficulty calibration episode_id must be non-empty")
        if episode_id in entries_by_episode:
            raise RuntimeError(f"public difficulty calibration duplicates episode_id {episode_id}")
        difficulty_bin = str(episode.get("difficulty_bin"))
        if difficulty_bin not in {"hard", "medium"}:
            raise RuntimeError(f"public difficulty calibration episode {episode_id} has unsupported difficulty_bin")
        score = episode.get("panel_mean_accuracy")
        if not isinstance(score, (int, float)):
            raise RuntimeError(f"public difficulty calibration episode {episode_id} has invalid panel_mean_accuracy")
        rank = episode.get("rank")
        if not isinstance(rank, int) or rank <= 0:
            raise RuntimeError(f"public difficulty calibration episode {episode_id} has invalid rank")
        if rank in seen_ranks:
            raise RuntimeError(f"public difficulty calibration duplicates rank {rank}")
        seen_ranks.add(rank)
        entries_by_episode[episode_id] = {
            "panel_mean_accuracy": round(float(score), 6),
            "difficulty_bin": difficulty_bin,
            "rank": rank,
        }
    expected_ranks = set(range(1, len(entries_by_episode) + 1))
    if seen_ranks != expected_ranks:
        raise RuntimeError("public difficulty calibration ranks must form a contiguous sequence starting at 1")
    return payload, entries_by_episode


def apply_empirical_difficulty_to_payloads(
    rows: list[dict[str, object]],
    answers: list[dict[str, object]],
    entries_by_episode: dict[str, dict[str, object]],
) -> None:
    """Apply calibrated difficulty bins to row and answer payloads.

    Args:
        rows: Public rows to annotate.
        answers: Matching answer payloads to annotate.
        entries_by_episode: Difficulty entries keyed by episode ID.

    Returns:
        None.

    Raises:
        RuntimeError: If the calibration coverage does not match the payloads.

    """
    row_episode_ids = {str(row["episode_id"]) for row in rows}
    answer_episode_ids = {str(answer["episode_id"]) for answer in answers}
    calibration_episode_ids = set(entries_by_episode)
    missing_row_ids = sorted(calibration_episode_ids - row_episode_ids)
    extra_row_ids = sorted(row_episode_ids - calibration_episode_ids)
    if missing_row_ids or extra_row_ids:
        raise RuntimeError(
            "public difficulty calibration coverage mismatch for rows: "
            f"missing={missing_row_ids}, extra={extra_row_ids}"
        )
    missing_answer_ids = sorted(calibration_episode_ids - answer_episode_ids)
    extra_answer_ids = sorted(answer_episode_ids - calibration_episode_ids)
    if missing_answer_ids or extra_answer_ids:
        raise RuntimeError(
            "public difficulty calibration coverage mismatch for answers: "
            f"missing={missing_answer_ids}, extra={extra_answer_ids}"
        )
    for payload in [*rows, *answers]:
        episode_id = str(payload["episode_id"])
        payload["analysis"]["difficulty_bin"] = str(entries_by_episode[episode_id]["difficulty_bin"])


def derive_seed(*parts: object) -> int:
    """Derive a deterministic integer seed from arbitrary parts.

    Args:
        *parts: Values that should contribute to the derived seed.

    Returns:
        A deterministic unsigned integer seed.

    """
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def compute_sha256(path: Path) -> str:
    """Compute the SHA-256 digest for a file.

    Args:
        path: File whose contents should be hashed.

    Returns:
        The hexadecimal SHA-256 digest.

    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dataset_metadata(dataset_id: str, title: str) -> dict[str, object]:
    """Build Kaggle dataset metadata for a release artifact.

    Args:
        dataset_id: Dataset identifier to publish.
        title: Human-readable dataset title.

    Returns:
        A dataset metadata payload.

    """
    return {
        "id": dataset_id,
        "title": title,
        "licenses": [{"name": "CC0-1.0"}],
    }


def fmt_signed(value: int) -> str:
    """Render an integer with an explicit sign prefix.

    Args:
        value: Integer value to render.

    Returns:
        The signed decimal representation.

    """
    return f"{value:+d}"


def normalized_turn_text(turn: str) -> str:
    """Normalize episode-specific headers in rendered turn text.

    Args:
        turn: Rendered turn text.

    Returns:
        The turn text with episode numbers normalized for comparison.

    """
    return RETRIEVAL_HEADER_RE.sub("Episode XXXX", turn)


def build_domain(
    values: tuple[int, ...],
    shapes: tuple[str, ...],
    tones: tuple[str, ...],
    *,
    extras: dict[str, tuple[str, ...]] | None = None,
) -> list[Stimulus]:
    """Enumerate the full stimulus domain for a task family.

    Args:
        values: Numeric values used for both `r1` and `r2`.
        shapes: Shape vocabulary.
        tones: Tone vocabulary.
        extras: Optional additional categorical dimensions.

    Returns:
        The cartesian product of all requested stimulus attributes.

    """
    domain: list[Stimulus] = []
    extras = extras or {}
    extra_keys = list(extras)

    def visit(index: int, payload: Stimulus) -> None:
        if index == len(extra_keys):
            for r1 in values:
                for r2 in values:
                    for shape in shapes:
                        for tone in tones:
                            domain.append(
                                {
                                    "r1": r1,
                                    "r2": r2,
                                    "shape": shape,
                                    "tone": tone,
                                    **payload,
                                }
                            )
            return
        key = extra_keys[index]
        for value in extras[key]:
            visit(index + 1, {**payload, key: value})

    visit(0, {})
    return domain


PUBLIC_DOMAIN = build_domain(PUBLIC_VALUES, PUBLIC_SHAPES, PUBLIC_TONES)


def make_two_label_rule(
    rule_id: str,
    family_id: str,
    labels: tuple[str, str],
    description: str,
    predicate: Callable[[Stimulus], bool],
) -> RuleSpec:
    """Build a binary rule specification from a predicate.

    Args:
        rule_id: Stable rule identifier.
        family_id: Rule family identifier.
        labels: Ordered binary label vocabulary.
        description: Human-readable rule description.
        predicate: Predicate selecting the first label.

    Returns:
        A binary rule specification.

    """
    return RuleSpec(
        rule_id=rule_id,
        family_id=family_id,
        label_vocab=labels,
        description=description,
        resolver=lambda stimulus: labels[0] if predicate(stimulus) else labels[1],
    )


def make_three_label_rule(
    rule_id: str,
    family_id: str,
    labels: tuple[str, str, str],
    description: str,
    resolver: Callable[[Stimulus], str],
) -> RuleSpec:
    """Build a ternary rule specification from a resolver.

    Args:
        rule_id: Stable rule identifier.
        family_id: Rule family identifier.
        labels: Ordered ternary label vocabulary.
        description: Human-readable rule description.
        resolver: Resolver that assigns a label to each stimulus.

    Returns:
        A ternary rule specification.

    """
    return RuleSpec(
        rule_id=rule_id,
        family_id=family_id,
        label_vocab=labels,
        description=description,
        resolver=resolver,
    )


PUBLIC_RULES: Final[dict[str, RuleSpec]] = {
    "accept_r1_nonnegative": make_two_label_rule(
        "accept_r1_nonnegative",
        "numeric_threshold",
        ("accept", "reject"),
        "accept when r1 is non-negative",
        lambda stimulus: int(stimulus["r1"]) >= 0,
    ),
    "accept_abs_sum_large": make_two_label_rule(
        "accept_abs_sum_large",
        "magnitude_gate",
        ("accept", "reject"),
        "accept when |r1| + |r2| is at least 6",
        lambda stimulus: abs(int(stimulus["r1"])) + abs(int(stimulus["r2"])) >= 6,
    ),
    "accept_shape_round": make_two_label_rule(
        "accept_shape_round",
        "symbolic_gate",
        ("accept", "reject"),
        "accept when the shape is circle or hex",
        lambda stimulus: str(stimulus["shape"]) in {"circle", "hex"},
    ),
    "accept_parity_match": make_two_label_rule(
        "accept_parity_match",
        "relational_gate",
        ("accept", "reject"),
        "accept when r1 and r2 share parity",
        lambda stimulus: int(stimulus["r1"]) % 2 == int(stimulus["r2"]) % 2,
    ),
    "north_r1_ge_r2": make_two_label_rule(
        "north_r1_ge_r2",
        "relational_gate",
        ("north", "south"),
        "north when r1 is at least r2",
        lambda stimulus: int(stimulus["r1"]) >= int(stimulus["r2"]),
    ),
    "north_warm_or_positive": make_two_label_rule(
        "north_warm_or_positive",
        "symbolic_gate",
        ("north", "south"),
        "north when tone is warm or r2 is positive",
        lambda stimulus: str(stimulus["tone"]) == "warm" or int(stimulus["r2"]) > 0,
    ),
    "amber_cobalt_jade_sum_band": make_three_label_rule(
        "amber_cobalt_jade_sum_band",
        "tri_band",
        ("amber", "cobalt", "jade"),
        "sum band over r1+r2",
        lambda stimulus: (
            "amber"
            if int(stimulus["r1"]) + int(stimulus["r2"]) <= -3
            else "jade"
            if int(stimulus["r1"]) + int(stimulus["r2"]) >= 3
            else "cobalt"
        ),
    ),
    "amber_cobalt_jade_shape_tone": make_three_label_rule(
        "amber_cobalt_jade_shape_tone",
        "symbolic_router",
        ("amber", "cobalt", "jade"),
        "shape/tone router",
        lambda stimulus: (
            "amber"
            if str(stimulus["shape"]) in {"circle", "hex"}
            else "jade"
            if str(stimulus["tone"]) == "warm"
            else "cobalt"
        ),
    ),
}


PUBLIC_STRUCTURES: Final[dict[str, EpisodeStructure]] = {
    "two_step_focus": EpisodeStructure("two_step_focus", (4, 4), 5),
    "three_step_bridge": EpisodeStructure("three_step_bridge", (3, 2, 3), 6),
}


TASK_RULE_PAIRS: Final[dict[str, tuple[tuple[str, str], ...]]] = {
    "explicit_rule_update": (
        ("accept_r1_nonnegative", "accept_abs_sum_large"),
        ("accept_shape_round", "accept_parity_match"),
    ),
    "latent_rule_update": (
        ("north_r1_ge_r2", "north_warm_or_positive"),
        ("accept_r1_nonnegative", "accept_shape_round"),
    ),
    "context_binding": (
        ("amber_cobalt_jade_sum_band", "amber_cobalt_jade_shape_tone"),
        ("accept_shape_round", "accept_abs_sum_large"),
    ),
    "trial_cued_switch": (
        ("accept_shape_round", "accept_parity_match"),
        ("north_r1_ge_r2", "north_warm_or_positive"),
    ),
}


def public_generator_metadata(suite_task_id: str, *, variant: int) -> dict[str, str]:
    """Build generator metadata for a public episode variant.

    Args:
        suite_task_id: Public suite task identifier.
        variant: Variant index within the suite task.

    Returns:
        Generator family, template, and operator metadata.

    Raises:
        ValueError: If the suite task is unsupported.

    """
    if suite_task_id not in TASK_RULE_PAIRS:
        raise ValueError(f"unsupported suite_task_id {suite_task_id!r}")
    pair_index = variant % len(TASK_RULE_PAIRS[suite_task_id])
    source_rule_id, target_rule_id = TASK_RULE_PAIRS[suite_task_id][pair_index]
    return {
        "family_id": f"public::{suite_task_id}",
        "template_id": f"public::{suite_task_id}::{source_rule_id}__{target_rule_id}",
        "operator_class": PUBLIC_GENERATOR_OPERATOR_CLASS_BY_TASK[suite_task_id],
    }


def public_generator_reference() -> dict[str, tuple[str, ...]]:
    """Collect the set of generator metadata used by the public split.

    Returns:
        Distinct family IDs, template IDs, and operator classes used publicly.

    """
    family_ids: set[str] = set()
    template_ids: set[str] = set()
    operator_classes: set[str] = set()
    for suite_task_id in SUITE_TASKS:
        for variant in range(len(TASK_RULE_PAIRS[suite_task_id])):
            metadata = public_generator_metadata(suite_task_id, variant=variant)
            family_ids.add(metadata["family_id"])
            template_ids.add(metadata["template_id"])
            operator_classes.add(metadata["operator_class"])
    return {
        "family_ids": tuple(sorted(family_ids)),
        "template_ids": tuple(sorted(template_ids)),
        "operator_classes": tuple(sorted(operator_classes)),
    }


def stimulus_signature(stimulus: Stimulus) -> tuple[object, ...]:
    """Create a stable comparable signature for a stimulus.

    Args:
        stimulus: Stimulus payload to normalize.

    Returns:
        A tuple representation sorted by key.

    """
    return tuple((key, stimulus[key]) for key in sorted(stimulus))


def serialize_case(
    index: int,
    stimulus: Stimulus,
    label: str,
    *,
    context: str | None = None,
    cue: str | None = None,
    rule_id: str | None = None,
) -> dict[str, object]:
    """Serialize one labeled stimulus into a turn item payload.

    Args:
        index: Display index within the turn.
        stimulus: Stimulus attributes to serialize.
        label: Assigned label for the stimulus.
        context: Optional context tag.
        cue: Optional cue tag.
        rule_id: Optional originating rule identifier.

    Returns:
        The serialized item payload.

    """
    item = {"index": index, **stimulus, "label": label}
    if context is not None:
        item["context"] = context
    if cue is not None:
        item["cue"] = cue
    if rule_id is not None:
        item["rule_id"] = rule_id
    return item


def response_spec(label_vocab: tuple[str, ...], probe_count: int) -> dict[str, object]:
    """Build the response specification for a decision turn.

    Args:
        label_vocab: Allowed output labels.
        probe_count: Number of probes in the decision turn.

    Returns:
        A normalized response specification payload.

    """
    return {
        "format": "ordered_labels",
        "probe_count": probe_count,
        "label_vocab": list(label_vocab),
    }


def turn_spec(kind: str, item_count: int) -> dict[str, object]:
    """Build metadata describing a rendered turn.

    Args:
        kind: Turn kind, such as `evidence` or `decision`.
        item_count: Number of serialized items in the turn.

    Returns:
        A turn specification payload.

    """
    return {"kind": kind, "item_count": item_count}


def response_instruction(spec: dict[str, object]) -> str:
    """Render the response instruction text from a response spec.

    Args:
        spec: Response specification payload.

    Returns:
        The instruction string appended to decision turns.

    """
    vocab = ", ".join(str(label) for label in spec["label_vocab"])
    return (
        f"Return exactly {spec['probe_count']} outputs in order, one per probe. "
        f"Use only labels from: {vocab}."
    )


def render_case_text(item: dict[str, object], *, attribute_variant: int) -> str:
    """Render one serialized item into the textual case format.

    Args:
        item: Serialized case payload.
        attribute_variant: Rotation offset for attribute ordering.

    Returns:
        The textual representation shown in a turn.

    """
    chunk_map = {
        "point": f"r1={fmt_signed(int(item['r1']))}, r2={fmt_signed(int(item['r2']))}",
        "shape": f"shape={item['shape']}",
        "tone": f"tone={item['tone']}",
    }
    for key in sorted(key for key in item if key not in {"index", "label", "r1", "r2", "shape", "tone", "rule_id"}):
        chunk_map[key] = f"{key}={item[key]}"
    order = list(chunk_map)
    rotation = attribute_variant % len(order)
    ordered_keys = order[rotation:] + order[:rotation]
    return " | ".join(chunk_map[key] for key in ordered_keys)


def render_items(items: list[dict[str, object]], *, hide_labels: bool, attribute_variant: int) -> str:
    """Render multiple serialized items into newline-delimited text.

    Args:
        items: Serialized case payloads to render.
        hide_labels: Whether to replace labels with question marks.
        attribute_variant: Rotation offset for attribute ordering.

    Returns:
        The rendered block of line-oriented items.

    """
    lines = []
    for offset, item in enumerate(items):
        label = "?" if hide_labels else str(item["label"])
        lines.append(
            f"{item['index']}. {render_case_text(item, attribute_variant=attribute_variant + offset)} -> {label}"
        )
    return "\n".join(lines)


def render_turn(
    episode_id: str,
    turn_index: int,
    turn_count: int,
    kind: str,
    prompt: str,
    items: list[dict[str, object]],
    *,
    attribute_variant: int,
    spec: dict[str, object] | None = None,
) -> str:
    """Render a full turn including header, prompt, items, and instructions.

    Args:
        episode_id: Episode identifier used in the header.
        turn_index: One-based position of the turn.
        turn_count: Total number of turns in the episode.
        kind: Turn kind, such as `evidence` or `decision`.
        prompt: Instructional prompt for the turn.
        items: Serialized case payloads shown in the turn.
        attribute_variant: Rotation offset for attribute ordering.
        spec: Optional response specification for decision turns.

    Returns:
        The rendered turn text.

    """
    sections = [
        f"{TURN_HEADER_PREFIX}{episode_id}. Turn {turn_index} of {turn_count}.",
        prompt,
        ("Examples:\n" if kind == "evidence" else "Probes:\n") + render_items(
            items,
            hide_labels=kind == "decision",
            attribute_variant=attribute_variant,
        ),
    ]
    if kind == "decision" and spec is not None:
        sections.append(response_instruction(spec))
    return "\n\n".join(sections)


def parse_case_line(line: str) -> dict[str, object] | None:
    """Parse one rendered case line back into its structured payload.

    Args:
        line: Rendered line from an evidence or decision turn.

    Returns:
        The parsed item payload, or `None` when the line is not a case line.

    Raises:
        ValueError: If the line omits the required numeric point fields.

    """
    match = LINE_RE.match(line.strip())
    if match is None:
        return None
    payload: dict[str, object] = {"index": int(match.group("index")), "label": match.group("label")}
    point_found = False
    for chunk in (part.strip() for part in match.group("body").split("|")):
        point_match = POINT_RE.match(chunk)
        if point_match is not None:
            payload["r1"] = int(point_match.group("r1"))
            payload["r2"] = int(point_match.group("r2"))
            point_found = True
            continue
        key, value = chunk.split("=", 1)
        payload[key.strip()] = value.strip()
    if not point_found:
        raise ValueError(f"missing r1/r2 pair in line: {line!r}")
    return payload


def parse_turn_items(turn: str, *, kind: str) -> list[dict[str, object]]:
    """Parse all serialized items from a rendered turn.

    Args:
        turn: Rendered turn text.
        kind: Turn kind used to validate visible labels.

    Returns:
        The parsed item payloads in display order.

    """
    items: list[dict[str, object]] = []
    for line in turn.splitlines():
        parsed = parse_case_line(line)
        if parsed is None:
            continue
        if kind == "decision" and parsed["label"] != "?":
            continue
        if kind == "evidence" and parsed["label"] == "?":
            continue
        items.append(parsed)
    return items


def label_distribution(label_vocab: tuple[str, ...], count: int, *, rotation: int = 0) -> dict[str, int]:
    """Distribute a sample count as evenly as possible across labels.

    Args:
        label_vocab: Ordered label vocabulary.
        count: Total number of examples to assign.
        rotation: Offset used when assigning remainder examples.

    Returns:
        The requested count per label.

    """
    ordered = list(label_vocab)
    if ordered:
        shift = rotation % len(ordered)
        ordered = ordered[shift:] + ordered[:shift]
    base = count // len(label_vocab)
    remainder = count % len(label_vocab)
    distribution = {label: base for label in label_vocab}
    for label in ordered[:remainder]:
        distribution[label] += 1
    return distribution


def sample_for_rule(
    rng: random.Random,
    domain: list[Stimulus],
    rule: RuleSpec,
    count: int,
    *,
    exclude: set[tuple[object, ...]] | None = None,
    rotation: int = 0,
    mismatch_rule: RuleSpec | None = None,
    min_mismatch: int = 0,
) -> list[Stimulus]:
    """Sample stimuli that satisfy a rule with optional mismatch pressure.

    Args:
        rng: Random generator used for deterministic sampling.
        domain: Available stimulus domain.
        rule: Rule used to label the sampled stimuli.
        count: Number of stimuli to sample.
        exclude: Signatures that must not be reused.
        rotation: Offset used when balancing labels.
        mismatch_rule: Optional reference rule used to enforce disagreement.
        min_mismatch: Minimum number of disagreements to induce.

    Returns:
        The sampled stimuli.

    Raises:
        RuntimeError: If there are not enough candidate stimuli for a label.

    """
    excluded = exclude or set()
    distribution = label_distribution(rule.label_vocab, count, rotation=rotation)
    buckets: dict[str, list[Stimulus]] = {label: [] for label in rule.label_vocab}
    for stimulus in domain:
        signature = stimulus_signature(stimulus)
        if signature in excluded:
            continue
        buckets[rule.label(stimulus)].append(stimulus)
    selected: list[Stimulus] = []
    for label in rule.label_vocab:
        candidates = list(buckets[label])
        rng.shuffle(candidates)
        need = distribution[label]
        if len(candidates) < need:
            raise RuntimeError(f"not enough stimuli for {rule.rule_id=} {label=} {need=}")
        selected.extend(candidates[:need])
    rng.shuffle(selected)
    if mismatch_rule is not None:
        mismatch_count = sum(mismatch_rule.label(stimulus) != rule.label(stimulus) for stimulus in selected)
        if mismatch_count < min_mismatch:
            stronger = [
                stimulus
                for stimulus in domain
                if stimulus_signature(stimulus) not in excluded and mismatch_rule.label(stimulus) != rule.label(stimulus)
            ]
            rng.shuffle(stronger)
            replacements = iter(stronger)
            for index, stimulus in enumerate(list(selected)):
                if mismatch_count >= min_mismatch:
                    break
                if mismatch_rule.label(stimulus) != rule.label(stimulus):
                    continue
                replacement = next(replacements, None)
                if replacement is None:
                    break
                replacement_signature = stimulus_signature(replacement)
                if replacement_signature in {stimulus_signature(item) for item in selected}:
                    continue
                replacement_label = rule.label(replacement)
                current_label = rule.label(stimulus)
                if distribution[replacement_label] > sum(rule.label(item) == replacement_label for item in selected):
                    continue
                if distribution[current_label] < 1:
                    continue
                selected[index] = replacement
                mismatch_count += 1
    return selected


def enumerate_items(
    stimuli: list[Stimulus],
    rule: RuleSpec,
    *,
    context: str | None = None,
    cue: str | None = None,
    start_index: int = 1,
) -> list[dict[str, object]]:
    """Convert sampled stimuli into serialized turn items.

    Args:
        stimuli: Stimuli to serialize.
        rule: Rule used to assign labels.
        context: Optional context tag to attach to each item.
        cue: Optional cue tag to attach to each item.
        start_index: First display index to use.

    Returns:
        The serialized items.

    """
    return [
        serialize_case(index, stimulus, rule.label(stimulus), context=context, cue=cue, rule_id=rule.rule_id)
        for index, stimulus in enumerate(stimuli, start=start_index)
    ]


def sample_mixed_route_examples(
    rng: random.Random,
    domain: list[Stimulus],
    assignments: list[tuple[str, RuleSpec, str]],
    count: int,
    *,
    route_key: str,
    exclude: set[tuple[object, ...]] | None = None,
    disagreement_rule: tuple[RuleSpec, RuleSpec] | None = None,
) -> list[dict[str, object]]:
    """Sample and interleave items from multiple routing assignments.

    Args:
        rng: Random generator used for deterministic sampling.
        domain: Available stimulus domain.
        assignments: Route value, rule, and route role assignments to sample.
        count: Total number of items to generate.
        route_key: Field name that stores the route value on each item.
        exclude: Signatures that must not be reused.
        disagreement_rule: Optional rule pair used to enforce contrasting items.

    Returns:
        The mixed serialized items in randomized order.

    """
    excluded = exclude or set()
    per_route = max(1, count // len(assignments))
    remainder = count - (per_route * len(assignments))
    items: list[dict[str, object]] = []
    for route_index, (route_value, rule, rule_id) in enumerate(assignments):
        take = per_route + (1 if route_index < remainder else 0)
        stimuli = sample_for_rule(
            rng,
            domain,
            rule,
            take,
            exclude=excluded,
            rotation=route_index,
            mismatch_rule=disagreement_rule[0] if disagreement_rule and rule_id == "alternate" else None,
            min_mismatch=1 if take > 1 and disagreement_rule and rule_id == "alternate" else 0,
        )
        for stimulus in stimuli:
            excluded.add(stimulus_signature(stimulus))
        for index, stimulus in enumerate(stimuli, start=len(items) + 1):
            item = serialize_case(index, stimulus, rule.label(stimulus), rule_id=rule.rule_id)
            item[route_key] = route_value
            items.append(item)
    rng.shuffle(items)
    for index, item in enumerate(items, start=1):
        item["index"] = index
    return items


def compute_probe_annotations(
    probes: list[Stimulus],
    active_rule: RuleSpec,
    contrast_rule: RuleSpec,
) -> list[str]:
    """Tag each probe as incongruent or congruent across two rules.

    Args:
        probes: Probe stimuli to annotate.
        active_rule: Rule used to label the probes.
        contrast_rule: Rule representing the alternative hypothesis.

    Returns:
        A list of ``"incongruent"`` or ``"congruent"`` annotations parallel
        to the probe list.

    """
    return [
        "incongruent" if active_rule.label(stimulus) != contrast_rule.label(stimulus) else "congruent"
        for stimulus in probes
    ]


def build_episode_payload(
    episode_id: str,
    *,
    suite_task_id: str,
    structure: EpisodeStructure,
    label_vocab: tuple[str, ...],
    turn_prompts: list[str],
    turn_items: list[list[dict[str, object]]],
    probe_annotations: list[str] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Assemble the row and answer payloads for one episode.

    Args:
        episode_id: Stable identifier to assign.
        suite_task_id: Suite task represented by the episode.
        structure: Episode structure controlling turn layout.
        label_vocab: Allowed output labels for the decision turn.
        turn_prompts: Prompt text for each turn.
        turn_items: Serialized items for each turn.
        probe_annotations: Optional per-probe congruency annotations.

    Returns:
        The public row payload and the answer payload.

    """
    specs = [turn_spec("evidence", len(items)) for items in turn_items[:-1]] + [turn_spec("decision", len(turn_items[-1]))]
    spec = response_spec(label_vocab, len(turn_items[-1]))
    turns = [
        render_turn(
            episode_id,
            turn_index,
            structure.turn_count,
            specs[turn_index - 1]["kind"],
            turn_prompts[turn_index - 1],
            items,
            attribute_variant=turn_index,
            spec=spec if turn_index == structure.turn_count else None,
        )
        for turn_index, items in enumerate(turn_items, start=1)
    ]
    analysis = {
        "faculty_id": FACULTY_ID,
        "suite_task_id": suite_task_id,
        "shift_mode": SHIFT_MODES[suite_task_id],
        "difficulty_bin": "pending_calibration",
        "structure_family_id": structure.structure_family_id,
    }
    inference = {"turns": turns, "turn_specs": specs, "response_spec": spec}
    targets = [str(item["label"]) for item in turn_items[-1]]
    scoring: dict[str, object] = {"final_probe_targets": targets}
    if probe_annotations is not None:
        scoring["probe_annotations"] = probe_annotations
    row = {
        "episode_id": episode_id,
        "inference": inference,
        "analysis": analysis,
        "scoring": scoring,
    }
    answer = {
        "episode_id": episode_id,
        "analysis": analysis,
        "inference": inference,
        "final_probe_targets": targets,
    }
    if probe_annotations is not None:
        answer["probe_annotations"] = probe_annotations
    return row, answer


_IDENTIFIABILITY_METADATA_KEYS: Final[frozenset[str]] = frozenset({"index", "label", "rule_id", "context", "cue"})


def _stimulus_from_parsed_item(item: dict[str, object]) -> Stimulus:
    """Strip metadata keys from a parsed turn item to recover a stimulus.

    Args:
        item: Parsed turn item payload.

    Returns:
        A stimulus dictionary suitable for RuleSpec evaluation.

    """
    return {key: value for key, value in item.items() if key not in _IDENTIFIABILITY_METADATA_KEYS}


def _row_turn_payloads(row: dict[str, object]) -> tuple[list[list[dict[str, object]]], list[dict[str, object]]]:
    """Split a row into evidence turn items and decision turn items.

    Args:
        row: Row payload whose inference turns should be parsed.

    Returns:
        A tuple of (evidence turn items per turn, decision turn items).

    Raises:
        RuntimeError: If the row is missing an evidence or decision turn.

    """
    turns = row["inference"]["turns"]
    specs = row["inference"]["turn_specs"]
    evidence_items: list[list[dict[str, object]]] = []
    decision_items: list[dict[str, object]] | None = None
    for turn, spec in zip(turns, specs, strict=True):
        kind = str(spec["kind"])
        items = parse_turn_items(turn, kind=kind)
        if kind == "evidence":
            evidence_items.append(items)
        elif kind == "decision":
            decision_items = items
    if not evidence_items or decision_items is None:
        raise RuntimeError(f"row {row.get('episode_id')} is missing evidence or decision turns for identifiability")
    return evidence_items, decision_items


def _candidate_rules_for_vocab(
    rule_catalogue: dict[str, RuleSpec],
    label_vocab: tuple[str, ...],
) -> list[RuleSpec]:
    """Select catalogue rules whose label vocabulary matches the episode.

    Args:
        rule_catalogue: Rule catalogue to enumerate.
        label_vocab: Label vocabulary exposed by the episode.

    Returns:
        Rules with a matching label vocabulary.

    """
    return [rule for rule in rule_catalogue.values() if tuple(rule.label_vocab) == label_vocab]


def compute_identifiability(
    row: dict[str, object],
    *,
    rule_catalogue: dict[str, RuleSpec],
    kind: str,
    route_field: str | None = None,
) -> dict[str, object]:
    """Compute identifiability metrics for a single episode row.

    An episode is identifiable iff every hypothesis that is consistent with
    the observed evidence implies the same final probe target tuple. The
    hypothesis space is scoped to the supplied rule catalogue, filtered by
    the episode's label vocabulary, and enumerated either as a single rule
    (for ``single_rule_*`` kinds) or as a route-to-rule assignment (for
    ``routed_all_turns``).

    Args:
        row: Row payload to inspect.
        rule_catalogue: Candidate rule catalogue scoped to the episode split.
        kind: Identifiability kind governing which evidence items anchor the
            hypothesis.
        route_field: Field name used to split routed episodes into
            independent sub-hypotheses.

    Returns:
        A dictionary with ``consistent_hypothesis_count``,
        ``distinct_probe_target_count``, and ``is_identifiable`` keys.

    Raises:
        ValueError: If an unsupported identifiability kind is requested.
        RuntimeError: If a routed kind is requested without a route field or
            no route values are present on the episode items.

    """
    label_vocab = tuple(str(label) for label in row["inference"]["response_spec"]["label_vocab"])
    evidence_turn_items, decision_items = _row_turn_payloads(row)
    candidates = _candidate_rules_for_vocab(rule_catalogue, label_vocab)

    if kind in (IDENTIFIABILITY_KIND_SINGLE_LAST, IDENTIFIABILITY_KIND_SINGLE_ALL):
        if kind == IDENTIFIABILITY_KIND_SINGLE_LAST:
            anchoring_items = list(evidence_turn_items[-1])
        else:
            anchoring_items = [item for turn_items in evidence_turn_items for item in turn_items]
        distinct_probe_targets: set[tuple[str, ...]] = set()
        consistent_count = 0
        for candidate in candidates:
            if not all(
                candidate.label(_stimulus_from_parsed_item(item)) == str(item["label"])
                for item in anchoring_items
            ):
                continue
            consistent_count += 1
            predicted = tuple(
                candidate.label(_stimulus_from_parsed_item(item)) for item in decision_items
            )
            distinct_probe_targets.add(predicted)
        return {
            "consistent_hypothesis_count": consistent_count,
            "distinct_probe_target_count": len(distinct_probe_targets),
            "is_identifiable": consistent_count >= 1 and len(distinct_probe_targets) == 1,
        }

    if kind != IDENTIFIABILITY_KIND_ROUTED_ALL:
        raise ValueError(f"unsupported identifiability kind {kind!r}")
    if route_field is None:
        raise RuntimeError("routed identifiability requires a route_field")

    route_values = sorted(
        {
            str(item[route_field])
            for turn_items in evidence_turn_items
            for item in turn_items
            if route_field in item
        }
        | {
            str(item[route_field])
            for item in decision_items
            if route_field in item
        }
    )
    if not route_values:
        raise RuntimeError(f"routed identifiability found no {route_field!r} values")

    per_route_candidates: dict[str, list[RuleSpec]] = {}
    for route_value in route_values:
        items_for_route = [
            item
            for turn_items in evidence_turn_items
            for item in turn_items
            if route_field in item and str(item[route_field]) == route_value
        ]
        per_route_candidates[route_value] = [
            candidate
            for candidate in candidates
            if all(
                candidate.label(_stimulus_from_parsed_item(item)) == str(item["label"])
                for item in items_for_route
            )
        ]

    consistent_count = 0
    distinct_probe_targets_routed: set[tuple[str, ...]] = set()

    def _visit(index: int, assignment: dict[str, RuleSpec]) -> None:
        nonlocal consistent_count
        if index == len(route_values):
            consistent_count += 1
            predicted = tuple(
                assignment[str(item[route_field])].label(_stimulus_from_parsed_item(item))
                for item in decision_items
            )
            distinct_probe_targets_routed.add(predicted)
            return
        route_value = route_values[index]
        for candidate in per_route_candidates[route_value]:
            assignment[route_value] = candidate
            _visit(index + 1, assignment)
        assignment.pop(route_value, None)

    _visit(0, {})
    return {
        "consistent_hypothesis_count": consistent_count,
        "distinct_probe_target_count": len(distinct_probe_targets_routed),
        "is_identifiable": consistent_count >= 1 and len(distinct_probe_targets_routed) == 1,
    }


def _identifiability_spec_for_row(row: dict[str, object], *, split: str) -> tuple[str, str | None]:
    """Resolve the identifiability kind and optional route field for a row.

    Args:
        row: Row payload whose analysis drives the identifiability spec.
        split: Dataset split name controlling which spec table is used.

    Returns:
        The identifiability kind and optional route field for the row.

    Raises:
        ValueError: If the split is unsupported or the row lacks a registered
            identifiability spec.

    """
    if split == "public":
        suite_task_id = str(row["analysis"]["suite_task_id"])
        if suite_task_id not in PUBLIC_IDENTIFIABILITY_SPEC_BY_TASK:
            raise ValueError(f"no public identifiability spec for suite_task_id {suite_task_id!r}")
        return PUBLIC_IDENTIFIABILITY_SPEC_BY_TASK[suite_task_id]
    if split == "private":
        structure_family_id = str(row["analysis"]["structure_family_id"])
        if structure_family_id not in PRIVATE_IDENTIFIABILITY_SPEC_BY_STRUCTURE:
            raise ValueError(
                f"no private identifiability spec for structure_family_id {structure_family_id!r}"
            )
        return PRIVATE_IDENTIFIABILITY_SPEC_BY_STRUCTURE[structure_family_id]
    raise ValueError(f"unsupported split {split!r}")


def identifiability_report_for_row(
    row: dict[str, object],
    *,
    split: str,
    rule_catalogue: dict[str, RuleSpec],
) -> dict[str, object]:
    """Compute the identifiability report for a row using its split spec.

    Args:
        row: Row payload to inspect.
        split: Dataset split identifier used to resolve the spec.
        rule_catalogue: Candidate rule catalogue for the split.

    Returns:
        The identifiability report dictionary.

    """
    kind, route_field = _identifiability_spec_for_row(row, split=split)
    return compute_identifiability(
        row,
        rule_catalogue=rule_catalogue,
        kind=kind,
        route_field=route_field,
    )


def build_explicit_episode(episode_id: str, *, structure: EpisodeStructure, variant: int, attempt: int = 0) -> tuple[dict[str, object], dict[str, object]]:
    """Build a public explicit-rule-update episode.

    Args:
        episode_id: Stable identifier to assign.
        structure: Structure controlling evidence and probe counts.
        variant: Deterministic variant index for rule selection and sampling.

    Returns:
        The generated row payload and matching answer payload.

    """
    seed_parts: tuple[object, ...] = ("public", "explicit_rule_update", structure.structure_family_id, variant)
    if attempt:
        seed_parts = (*seed_parts, "retry", attempt)
    seed = derive_seed(*seed_parts)
    rng = random.Random(seed)
    initial_rule = PUBLIC_RULES[TASK_RULE_PAIRS["explicit_rule_update"][variant % 2][0]]
    shift_rule = PUBLIC_RULES[TASK_RULE_PAIRS["explicit_rule_update"][variant % 2][1]]
    used: set[tuple[object, ...]] = set()
    turn_items: list[list[dict[str, object]]] = []
    prompts = ["Infer the current rule from these labeled examples."]
    first = sample_for_rule(rng, PUBLIC_DOMAIN, initial_rule, structure.evidence_counts[0], exclude=used, rotation=variant)
    used.update(stimulus_signature(stimulus) for stimulus in first)
    turn_items.append(enumerate_items(first, initial_rule))
    for bridge_index, count in enumerate(structure.evidence_counts[1:], start=1):
        rule = shift_rule if bridge_index == len(structure.evidence_counts) - 1 else initial_rule
        prompts.append(
            "The task rule has changed. Learn the replacement behavior from these examples."
            if rule is shift_rule
            else "Continue collecting evidence before the upcoming update."
        )
        sampled = sample_for_rule(
            rng,
            PUBLIC_DOMAIN,
            rule,
            count,
            exclude=used,
            rotation=variant + bridge_index,
            mismatch_rule=initial_rule if rule is shift_rule else None,
            min_mismatch=max(1, count // 2) if rule is shift_rule else 0,
        )
        used.update(stimulus_signature(stimulus) for stimulus in sampled)
        turn_items.append(enumerate_items(sampled, rule))
    probes = sample_for_rule(
        rng,
        PUBLIC_DOMAIN,
        shift_rule,
        structure.probe_count,
        exclude=used,
        rotation=variant + 5,
        mismatch_rule=initial_rule,
        min_mismatch=max(1, structure.probe_count // 2),
    )
    turn_items.append(enumerate_items(probes, shift_rule))
    prompts.append("Apply the active rule to every probe.")
    annotations = compute_probe_annotations(probes, shift_rule, initial_rule)
    return build_episode_payload(
        episode_id,
        suite_task_id="explicit_rule_update",
        structure=structure,
        label_vocab=shift_rule.label_vocab,
        turn_prompts=prompts,
        turn_items=turn_items,
        probe_annotations=annotations,
    )


def build_latent_episode(episode_id: str, *, structure: EpisodeStructure, variant: int, attempt: int = 0) -> tuple[dict[str, object], dict[str, object]]:
    """Build a public latent-rule-update episode.

    Args:
        episode_id: Stable identifier to assign.
        structure: Structure controlling evidence and probe counts.
        variant: Deterministic variant index for rule selection and sampling.

    Returns:
        The generated row payload and matching answer payload.

    """
    seed_parts: tuple[object, ...] = ("public", "latent_rule_update", structure.structure_family_id, variant)
    if attempt:
        seed_parts = (*seed_parts, "retry", attempt)
    seed = derive_seed(*seed_parts)
    rng = random.Random(seed)
    initial_rule = PUBLIC_RULES[TASK_RULE_PAIRS["latent_rule_update"][variant % 2][0]]
    shift_rule = PUBLIC_RULES[TASK_RULE_PAIRS["latent_rule_update"][variant % 2][1]]
    used: set[tuple[object, ...]] = set()
    prompts = ["Infer the live routing rule from these examples."]
    turn_items: list[list[dict[str, object]]] = []
    sampled = sample_for_rule(rng, PUBLIC_DOMAIN, initial_rule, structure.evidence_counts[0], exclude=used, rotation=variant)
    used.update(stimulus_signature(stimulus) for stimulus in sampled)
    turn_items.append(enumerate_items(sampled, initial_rule))
    for bridge_index, count in enumerate(structure.evidence_counts[1:], start=1):
        current_rule = shift_rule if bridge_index == len(structure.evidence_counts) - 1 else initial_rule
        prompts.append("Continue using the most recent labeled evidence.")
        sampled = sample_for_rule(
            rng,
            PUBLIC_DOMAIN,
            current_rule,
            count,
            exclude=used,
            rotation=variant + bridge_index,
            mismatch_rule=initial_rule if current_rule is shift_rule else None,
            min_mismatch=max(1, count // 2) if current_rule is shift_rule else 0,
        )
        used.update(stimulus_signature(stimulus) for stimulus in sampled)
        turn_items.append(enumerate_items(sampled, current_rule))
    probes = sample_for_rule(
        rng,
        PUBLIC_DOMAIN,
        shift_rule,
        structure.probe_count,
        exclude=used,
        rotation=variant + 8,
        mismatch_rule=initial_rule,
        min_mismatch=max(1, structure.probe_count // 2),
    )
    turn_items.append(enumerate_items(probes, shift_rule))
    prompts.append("Classify each probe using the latest behavior implied by the sequence.")
    annotations = compute_probe_annotations(probes, shift_rule, initial_rule)
    return build_episode_payload(
        episode_id,
        suite_task_id="latent_rule_update",
        structure=structure,
        label_vocab=shift_rule.label_vocab,
        turn_prompts=prompts,
        turn_items=turn_items,
        probe_annotations=annotations,
    )


def build_context_episode(episode_id: str, *, structure: EpisodeStructure, variant: int, attempt: int = 0) -> tuple[dict[str, object], dict[str, object]]:
    """Build a public context-binding episode.

    Args:
        episode_id: Stable identifier to assign.
        structure: Structure controlling evidence and probe counts.
        variant: Deterministic variant index for context and rule selection.

    Returns:
        The generated row payload and matching answer payload.

    """
    seed_parts: tuple[object, ...] = ("public", "context_binding", structure.structure_family_id, variant)
    if attempt:
        seed_parts = (*seed_parts, "retry", attempt)
    seed = derive_seed(*seed_parts)
    rng = random.Random(seed)
    context_terms = PUBLIC_CONTEXT_TERMS[variant % len(PUBLIC_CONTEXT_TERMS)]
    primary_rule = PUBLIC_RULES[TASK_RULE_PAIRS["context_binding"][variant % 2][0]]
    secondary_rule = PUBLIC_RULES[TASK_RULE_PAIRS["context_binding"][variant % 2][1]]
    label_vocab = primary_rule.label_vocab
    used: set[tuple[object, ...]] = set()
    prompts = [f"Learn the rule bound to context={context_terms[0]}."]
    turn_items: list[list[dict[str, object]]] = []
    sampled = sample_for_rule(rng, PUBLIC_DOMAIN, primary_rule, structure.evidence_counts[0], exclude=used, rotation=variant)
    used.update(stimulus_signature(stimulus) for stimulus in sampled)
    turn_items.append(enumerate_items(sampled, primary_rule, context=context_terms[0]))
    if len(structure.evidence_counts) > 1:
        prompts.append(f"Now learn the rule bound to context={context_terms[1]}.")
        sampled = sample_for_rule(
            rng,
            PUBLIC_DOMAIN,
            secondary_rule,
            structure.evidence_counts[1],
            exclude=used,
            rotation=variant + 2,
        )
        used.update(stimulus_signature(stimulus) for stimulus in sampled)
        turn_items.append(enumerate_items(sampled, secondary_rule, context=context_terms[1]))
    for extra_index, count in enumerate(structure.evidence_counts[2:], start=2):
        prompts.append("Use the context tag to keep the active rule aligned.")
        mixed = sample_mixed_route_examples(
            rng,
            PUBLIC_DOMAIN,
            [
                (context_terms[0], primary_rule, "primary"),
                (context_terms[1], secondary_rule, "secondary"),
            ],
            count,
            route_key="context",
            exclude=used,
        )
        used.update(stimulus_signature({key: item[key] for key in item if key not in {"index", "label", "context", "rule_id"}}) for item in mixed)
        turn_items.append(mixed)
    prompts.append("For each probe, use its context to select the right rule before labeling it.")
    probes = sample_mixed_route_examples(
        rng,
        PUBLIC_DOMAIN,
        [
            (context_terms[0], primary_rule, "primary"),
            (context_terms[1], secondary_rule, "secondary"),
        ],
        structure.probe_count,
        route_key="context",
        exclude=used,
    )
    turn_items.append(probes)
    probe_stimuli = [
        {key: item[key] for key in item if key not in {"index", "label", "context", "rule_id"}}
        for item in probes
    ]
    annotations = compute_probe_annotations(probe_stimuli, primary_rule, secondary_rule)
    return build_episode_payload(
        episode_id,
        suite_task_id="context_binding",
        structure=structure,
        label_vocab=label_vocab,
        turn_prompts=prompts,
        turn_items=turn_items,
        probe_annotations=annotations,
    )


def build_cued_episode(episode_id: str, *, structure: EpisodeStructure, variant: int, attempt: int = 0) -> tuple[dict[str, object], dict[str, object]]:
    """Build a public trial-cued-switch episode.

    Args:
        episode_id: Stable identifier to assign.
        structure: Structure controlling evidence and probe counts.
        variant: Deterministic variant index for cue and rule selection.

    Returns:
        The generated row payload and matching answer payload.

    """
    seed_parts: tuple[object, ...] = ("public", "trial_cued_switch", structure.structure_family_id, variant)
    if attempt:
        seed_parts = (*seed_parts, "retry", attempt)
    seed = derive_seed(*seed_parts)
    rng = random.Random(seed)
    cue_terms = PUBLIC_CUE_TERMS[variant % len(PUBLIC_CUE_TERMS)]
    keep_rule = PUBLIC_RULES[TASK_RULE_PAIRS["trial_cued_switch"][variant % 2][0]]
    switch_rule = PUBLIC_RULES[TASK_RULE_PAIRS["trial_cued_switch"][variant % 2][1]]
    label_vocab = keep_rule.label_vocab
    used: set[tuple[object, ...]] = set()
    prompts = ["Learn the base rule before any cue is introduced."]
    turn_items: list[list[dict[str, object]]] = []
    sampled = sample_for_rule(rng, PUBLIC_DOMAIN, keep_rule, structure.evidence_counts[0], exclude=used, rotation=variant)
    used.update(stimulus_signature(stimulus) for stimulus in sampled)
    turn_items.append(enumerate_items(sampled, keep_rule))
    if len(structure.evidence_counts) == 3:
        prompts.append("Now study examples from the alternate rule before cue routing begins.")
        sampled = sample_for_rule(
            rng,
            PUBLIC_DOMAIN,
            switch_rule,
            structure.evidence_counts[1],
            exclude=used,
            rotation=variant + 2,
            mismatch_rule=keep_rule,
            min_mismatch=1,
        )
        used.update(stimulus_signature(stimulus) for stimulus in sampled)
        turn_items.append(enumerate_items(sampled, switch_rule))
        mixed_count = structure.evidence_counts[2]
    else:
        mixed_count = structure.evidence_counts[1]
    prompts.append(
        f"Each item now has a cue. cue={cue_terms[0]} keeps the base rule and cue={cue_terms[1]} selects the alternate rule."
    )
    mixed = sample_mixed_route_examples(
        rng,
        PUBLIC_DOMAIN,
        [
            (cue_terms[0], keep_rule, "keep"),
            (cue_terms[1], switch_rule, "alternate"),
        ],
        mixed_count,
        route_key="cue",
        exclude=used,
        disagreement_rule=(keep_rule, switch_rule),
    )
    used.update(stimulus_signature({key: item[key] for key in item if key not in {"index", "label", "cue", "rule_id"}}) for item in mixed)
    turn_items.append(mixed)
    prompts.append("Use each probe cue to choose the active rule before labeling it.")
    probes = sample_mixed_route_examples(
        rng,
        PUBLIC_DOMAIN,
        [
            (cue_terms[0], keep_rule, "keep"),
            (cue_terms[1], switch_rule, "alternate"),
        ],
        structure.probe_count,
        route_key="cue",
        exclude=used,
        disagreement_rule=(keep_rule, switch_rule),
    )
    turn_items.append(probes)
    probe_stimuli = [
        {key: item[key] for key in item if key not in {"index", "label", "cue", "rule_id"}}
        for item in probes
    ]
    annotations = compute_probe_annotations(probe_stimuli, keep_rule, switch_rule)
    return build_episode_payload(
        episode_id,
        suite_task_id="trial_cued_switch",
        structure=structure,
        label_vocab=label_vocab,
        turn_prompts=prompts,
        turn_items=turn_items,
        probe_annotations=annotations,
    )


BUILDERS: Final[dict[str, Callable[[str], tuple[dict[str, object], dict[str, object]]]]] = {}


_PUBLIC_EPISODE_BUILDERS: Final[
    dict[
        str,
        Callable[..., tuple[dict[str, object], dict[str, object]]],
    ]
] = {
    "explicit_rule_update": build_explicit_episode,
    "latent_rule_update": build_latent_episode,
    "context_binding": build_context_episode,
    "trial_cued_switch": build_cued_episode,
}


def build_identifiable_public_episode(
    suite_task_id: str,
    episode_id: str,
    *,
    structure: EpisodeStructure,
    variant: int,
    retry_budget: int = IDENTIFIABILITY_RETRY_BUDGET,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Build an identifiable public episode, retrying until the check passes.

    Args:
        suite_task_id: Suite task selector used to pick the concrete builder.
        episode_id: Stable identifier for the episode.
        structure: Structure governing evidence layout.
        variant: Deterministic variant index.
        retry_budget: Maximum number of attempts before raising.

    Returns:
        The row payload, answer payload, and identifiability report for the
        accepted attempt.

    Raises:
        ValueError: If the suite task has no registered builder.
        RuntimeError: If no identifiable episode can be produced within the
            retry budget.

    """
    builder = _PUBLIC_EPISODE_BUILDERS.get(suite_task_id)
    if builder is None:
        raise ValueError(f"unsupported suite task {suite_task_id}")
    for attempt in range(retry_budget):
        row, answer = builder(episode_id, structure=structure, variant=variant, attempt=attempt)
        report = identifiability_report_for_row(row, split="public", rule_catalogue=PUBLIC_RULES)
        if report["is_identifiable"]:
            return row, answer, report
    raise RuntimeError(
        f"unable to build identifiable public episode {episode_id} for {suite_task_id} "
        f"within {retry_budget} attempts"
    )


def build_public_artifacts() -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Build the full tracked public split and its derived report.

    Returns:
        The public rows, public answers, and public quality report.

    """
    rows: list[dict[str, object]] = []
    answers: list[dict[str, object]] = []
    episode_number = 1
    for suite_task_id in SUITE_TASKS:
        for variant in range(PUBLIC_EPISODES_PER_TASK):
            structure = PUBLIC_STRUCTURES[PUBLIC_STRUCTURE_FAMILY_IDS[variant % len(PUBLIC_STRUCTURE_FAMILY_IDS)]]
            episode_id = f"{episode_number:04d}"
            row, answer, _report = build_identifiable_public_episode(
                suite_task_id,
                episode_id,
                structure=structure,
                variant=variant,
            )
            rows.append(row)
            answers.append(answer)
            episode_number += 1
    _payload, calibration_entries = load_public_difficulty_calibration()
    apply_empirical_difficulty_to_payloads(rows, answers, calibration_entries)
    report = build_public_quality_report(rows)
    return rows, answers, report


def build_public_quality_report(rows: list[dict[str, object]]) -> dict[str, object]:
    """Build the aggregate quality report for the public split.

    Args:
        rows: Public split rows to summarize.

    Returns:
        The public quality report payload.

    """
    task_counts = Counter(str(row["analysis"]["suite_task_id"]) for row in rows)
    difficulty_counts = Counter(str(row["analysis"]["difficulty_bin"]) for row in rows)
    structure_counts = Counter(str(row["analysis"]["structure_family_id"]) for row in rows)
    turn_counts = Counter(len(row["inference"]["turns"]) for row in rows)
    probe_counts = Counter(int(row["inference"]["response_spec"]["probe_count"]) for row in rows)
    label_vocab_sizes = Counter(len(row["inference"]["response_spec"]["label_vocab"]) for row in rows)
    suite_task_structure_counts: dict[str, dict[str, int]] = {}
    optional_field_keys: set[str] = set()
    nominal_values: dict[str, set[str]] = {"shape": set(), "tone": set()}
    for row in rows:
        task = str(row["analysis"]["suite_task_id"])
        structure = str(row["analysis"]["structure_family_id"])
        suite_task_structure_counts.setdefault(task, {})
        suite_task_structure_counts[task][structure] = suite_task_structure_counts[task].get(structure, 0) + 1
        for turn, spec in zip(row["inference"]["turns"], row["inference"]["turn_specs"], strict=True):
            for item in parse_turn_items(turn, kind=str(spec["kind"])):
                nominal_values["shape"].add(str(item["shape"]))
                nominal_values["tone"].add(str(item["tone"]))
                optional_field_keys.update(
                    key for key in item if key not in {"index", "label", "r1", "r2", "shape", "tone"}
                )
                for key, value in item.items():
                    if key in {"index", "label", "r1", "r2", "shape", "tone", "rule_id"}:
                        continue
                    if isinstance(value, str):
                        nominal_values.setdefault(key, set()).add(value)
    return {
        "version": PUBLIC_BUNDLE_VERSION,
        "task_name": TASK_NAME,
        "row_count": len(rows),
        "task_counts": dict(sorted(task_counts.items())),
        "difficulty_bin_counts": dict(sorted(difficulty_counts.items())),
        "structure_family_counts": dict(sorted(structure_counts.items())),
        "turn_count_distribution": {str(key): value for key, value in sorted(turn_counts.items())},
        "probe_count_distribution": {str(key): value for key, value in sorted(probe_counts.items())},
        "label_vocab_size_distribution": {str(key): value for key, value in sorted(label_vocab_sizes.items())},
        "suite_task_structure_counts": {
            key: dict(sorted(value.items())) for key, value in sorted(suite_task_structure_counts.items())
        },
        "stimulus_space_summary": {
            "numeric_range": {"r1": {"min": min(PUBLIC_VALUES), "max": max(PUBLIC_VALUES)}, "r2": {"min": min(PUBLIC_VALUES), "max": max(PUBLIC_VALUES)}},
            "nominal_cardinality": {key: len(values) for key, values in sorted(nominal_values.items())},
            "optional_field_keys": sorted(optional_field_keys),
        },
    }


def write_json(path: Path, payload: object) -> None:
    """Write a JSON payload with stable formatting.

    Args:
        path: Destination file path.
        payload: JSON-serializable payload to write.

    Returns:
        None.

    """
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Regenerate the tracked public dataset artifacts.

    Returns:
        None.

    """
    rows, _answers, report = build_public_artifacts()
    write_json(PUBLIC_ROWS_PATH, rows)
    write_json(PUBLIC_QUALITY_REPORT_PATH, report)
    write_json(PUBLIC_METADATA_PATH, dataset_metadata(PUBLIC_DATASET_ID, "CogFlex Cognitive Flexibility Runtime"))


if __name__ == "__main__":
    main()
