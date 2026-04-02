from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
import re

from core.kaggle.runner import score_episode
from tasks.ruleshift_benchmark.generator import generate_episode
from tasks.ruleshift_benchmark.protocol import (
    LABELED_ITEM_COUNT,
    PROBE_COUNT,
    InteractionLabel,
    ItemKind,
    Phase,
    PUBLIC_CONTRACT_VERSION,
    RuleName,
    Split,
    TemplateFamily,
    TEMPLATES,
    format_public_label,
    parse_public_label,
    parse_label,
)
from tasks.ruleshift_benchmark.render import render_binary_prompt
from tasks.ruleshift_benchmark.rules import label
from tasks.ruleshift_benchmark.schema import (
    build_contradiction_count_post,
    build_probe_label_counts,
    build_probe_sign_pattern_counts,
    build_updated_sign_patterns,
    probe_sign_pattern,
)

_BASELINE_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "ruleshift_canonical_baseline.json"
)
_ORIGINAL_CANONICAL_OUTRO = (
    "Return exactly 4 labels in order, one per probe. Use only attract or repel."
)


def _render_original_binary_prompt(episode: object) -> str:
    labeled_items = episode.items[:LABELED_ITEM_COUNT]
    probe_items = episode.items[LABELED_ITEM_COUNT:]

    if episode.template_family is TemplateFamily.CANONICAL:
        intro = (
            "You are given labeled interactions between two electric charges.\n"
            "Each labeled line shows q1, q2, and the observed result.\n"
            "Use the full sequence to infer which sign combinations were revised by the later evidence, "
            "then answer the final unlabeled cases."
        )
        labeled_heading = "Labeled examples:"
        probe_heading = "Probes:"
        labeled_lines = tuple(
            f"{item.position}. q1={item.q1:+d}, q2={item.q2:+d} -> {_render_original_label(item.label)}"
            for item in labeled_items
        )
        probe_lines = tuple(
            f"{item.position}. q1={item.q1:+d}, q2={item.q2:+d} -> ?"
            for item in probe_items
        )
    elif episode.template_family is TemplateFamily.OBSERVATION_LOG:
        intro = (
            "Review the observation log for interactions between two electric charges.\n"
            "Each entry records q1, q2, and the observed outcome.\n"
            "Use the full log to infer which sign combinations were revised later, then answer the unlabeled probe entries."
        )
        labeled_heading = "Resolved log entries:"
        probe_heading = "Unresolved probe entries:"
        labeled_lines = tuple(
            f"[{item.position:02d}] q1={item.q1:+d} | q2={item.q2:+d} | observed={_render_original_label(item.label)}"
            for item in labeled_items
        )
        probe_lines = tuple(
            f"[{item.position:02d}] q1={item.q1:+d} | q2={item.q2:+d} | observed=?"
            for item in probe_items
        )
    else:
        intro = (
            "Review the case ledger for interactions between two electric charges.\n"
            "Each row records the charge pair and the observed result.\n"
            "Use the full ledger to infer which sign combinations were revised by the later evidence, "
            "then complete the pending rows."
        )
        labeled_heading = "Confirmed ledger rows:"
        probe_heading = "Pending ledger rows:"
        labeled_lines = tuple(
            f"row {item.position:02d} | pair=({item.q1:+d}, {item.q2:+d}) | result={_render_original_label(item.label)}"
            for item in labeled_items
        )
        probe_lines = tuple(
            f"row {item.position:02d} | pair=({item.q1:+d}, {item.q2:+d}) | result=?"
            for item in probe_items
        )

    return "\n".join(
        (
            intro,
            "",
            labeled_heading,
            *labeled_lines,
            "",
            probe_heading,
            *probe_lines,
            "",
            _ORIGINAL_CANONICAL_OUTRO,
        )
    )


def _render_original_label(value: InteractionLabel | None) -> str:
    return "?" if value is None else value.value


def _extract_original_prompt_signature(
    prompt: str,
    template_family: TemplateFamily,
) -> tuple[tuple[int, int, int, InteractionLabel | None], ...]:
    if template_family is TemplateFamily.CANONICAL:
        pattern = re.compile(
            r"^(?P<position>\d+)\. q1=(?P<q1>[+-]\d+), q2=(?P<q2>[+-]\d+) -> (?P<label>\?|attract|repel)$"
        )
    elif template_family is TemplateFamily.OBSERVATION_LOG:
        pattern = re.compile(
            r"^\[(?P<position>\d{2})\] q1=(?P<q1>[+-]\d+) \| q2=(?P<q2>[+-]\d+) \| observed=(?P<label>\?|attract|repel)$"
        )
    else:
        pattern = re.compile(
            r"^row (?P<position>\d{2}) \| pair=\((?P<q1>[+-]\d+), (?P<q2>[+-]\d+)\) \| result=(?P<label>\?|attract|repel)$"
        )
    return _extract_prompt_signature(prompt, pattern, _parse_original_surface_label)


def _extract_reframed_prompt_signature(
    prompt: str,
    template_family: TemplateFamily,
) -> tuple[tuple[int, int, int, InteractionLabel | None], ...]:
    if template_family is TemplateFamily.CANONICAL:
        pattern = re.compile(
            r"^(?P<position>\d+)\. r1=(?P<q1>[+-]\d+), r2=(?P<q2>[+-]\d+) -> (?P<label>\?|zark|blim)$"
        )
    elif template_family is TemplateFamily.OBSERVATION_LOG:
        pattern = re.compile(
            r"^\[(?P<position>\d{2})\] r1=(?P<q1>[+-]\d+) \| r2=(?P<q2>[+-]\d+) \| observed=(?P<label>\?|zark|blim)$"
        )
    else:
        pattern = re.compile(
            r"^row (?P<position>\d{2}) \| r1=(?P<q1>[+-]\d+) \| r2=(?P<q2>[+-]\d+) \| state=(?P<label>\?|zark|blim)$"
        )
    return _extract_prompt_signature(prompt, pattern, _parse_reframed_surface_label)


def _extract_prompt_signature(
    prompt: str,
    pattern: re.Pattern[str],
    parse_label_value: Callable[[str], InteractionLabel | None],
) -> tuple[tuple[int, int, int, InteractionLabel | None], ...]:
    signature: list[tuple[int, int, int, InteractionLabel | None]] = []
    for line in prompt.splitlines():
        match = pattern.match(line)
        if match is None:
            continue
        label_value = match.group("label")
        signature.append(
            (
                int(match.group("position")),
                int(match.group("q1")),
                int(match.group("q2")),
                parse_label_value(label_value),
            )
        )
    return tuple(signature)


def _parse_original_surface_label(value: str) -> InteractionLabel | None:
    return None if value == "?" else InteractionLabel(value)


def _parse_reframed_surface_label(value: str) -> InteractionLabel | None:
    if value == "?":
        return None
    return {
        "zark": InteractionLabel.ATTRACT,
        "blim": InteractionLabel.REPEL,
    }[value]


def _flip_latent_prediction(
    predictions: tuple[InteractionLabel, ...],
) -> tuple[InteractionLabel, ...]:
    first, *rest = predictions
    flipped_first = (
        InteractionLabel.REPEL if first is InteractionLabel.ATTRACT else InteractionLabel.ATTRACT
    )
    return (flipped_first, *rest)


def _serialize_episode(episode: object) -> dict[str, object]:
    return {
        "episode_id": episode.episode_id,
        "split": episode.split.value,
        "difficulty": episode.difficulty.value,
        "template_id": episode.template_id.value,
        "template_family": episode.template_family.value,
        "transition": episode.transition.value,
        "rule_A": episode.rule_A.value,
        "rule_B": episode.rule_B.value,
        "pre_count": episode.pre_count,
        "post_labeled_count": episode.post_labeled_count,
        "shift_after_position": episode.shift_after_position,
        "contradiction_count_post": episode.contradiction_count_post,
        "difficulty_profile_id": episode.difficulty_profile_id.value,
        "difficulty_factors": {
            "conflict_strength": episode.difficulty_factors.conflict_strength.value,
            "post_shift_evidence_clarity": (
                episode.difficulty_factors.post_shift_evidence_clarity.value
            ),
            "probe_ambiguity": episode.difficulty_factors.probe_ambiguity.value,
            "evidence_to_final_probe_distance": (
                episode.difficulty_factors.evidence_to_final_probe_distance.value
            ),
            "pre_shift_distractor_pressure": (
                episode.difficulty_factors.pre_shift_distractor_pressure.value
            ),
        },
        "items": [
            {
                "position": item.position,
                "phase": item.phase.value,
                "kind": item.kind.value,
                "q1": item.q1,
                "q2": item.q2,
                "label": None if item.label is None else item.label.value,
            }
            for item in episode.items
        ],
        "probe_targets": [target.value for target in episode.probe_targets],
        "probe_label_counts": [
            [target.value, count] for target, count in episode.probe_label_counts
        ],
        "probe_sign_pattern_counts": [
            [pattern, count] for pattern, count in episode.probe_sign_pattern_counts
        ],
        "probe_metadata": [
            {
                "position": metadata.position,
                "is_disagreement_probe": metadata.is_disagreement_probe,
                "old_rule_label": metadata.old_rule_label.value,
                "new_rule_label": metadata.new_rule_label.value,
            }
            for metadata in episode.probe_metadata
        ],
    }


def _load_baseline_fixture() -> list[dict[str, object]]:
    return json.loads(_BASELINE_FIXTURE_PATH.read_text(encoding="utf-8"))


def test_rule_logic_freezes_same_and_opposite_sign_behavior() -> None:
    assert label(RuleName.R_STD, 2, 3) is InteractionLabel.REPEL
    assert label(RuleName.R_STD, -2, 3) is InteractionLabel.ATTRACT
    assert label(RuleName.R_INV, 2, 3) is InteractionLabel.ATTRACT
    assert label(RuleName.R_INV, -2, 3) is InteractionLabel.REPEL


def test_template_layouts_freeze_episode_structure() -> None:
    assert {template_id.value: spec.pre_count for template_id, spec in TEMPLATES.items()} == {
        "T1": 2,
        "T2": 3,
        "T3": 1,
    }
    assert {
        template_id.value: spec.post_labeled_count
        for template_id, spec in TEMPLATES.items()
    } == {
        "T1": 3,
        "T2": 2,
        "T3": 4,
    }


def test_generate_episode_is_deterministic_for_same_seed_and_split() -> None:
    first = _serialize_episode(generate_episode(0, split=Split.PUBLIC))
    second = _serialize_episode(generate_episode(0, split=Split.PUBLIC))

    assert first == second


def test_split_label_is_the_only_generation_difference_between_public_and_private() -> None:
    public_episode = _serialize_episode(generate_episode(27, split=Split.PUBLIC))
    private_episode = _serialize_episode(generate_episode(27, split=Split.PRIVATE))

    assert public_episode["split"] == "public"
    assert private_episode["split"] == "private"

    public_without_split = dict(public_episode)
    private_without_split = dict(private_episode)
    public_without_split.pop("split")
    private_without_split.pop("split")

    assert public_without_split == private_without_split


def test_baseline_fixture_matches_canonical_generation() -> None:
    fixture_rows = _load_baseline_fixture()
    actual_rows = []
    for row in fixture_rows:
        episode = generate_episode(row["seed"], split=Split.PUBLIC)
        actual_row = {"seed": row["seed"], **_serialize_episode(episode)}
        actual_rows.append(actual_row)

    assert actual_rows == fixture_rows


def test_baseline_seeds_preserve_shift_probe_and_metadata_invariants() -> None:
    for row in _load_baseline_fixture():
        episode = generate_episode(row["seed"], split=Split.PUBLIC)
        labeled_items = episode.items[:LABELED_ITEM_COUNT]
        probe_items = episode.items[LABELED_ITEM_COUNT:]
        post_labeled_items = labeled_items[episode.pre_count :]
        updated_sign_patterns = build_updated_sign_patterns(post_labeled_items)

        assert len(episode.items) == LABELED_ITEM_COUNT + PROBE_COUNT
        assert all(item.kind is ItemKind.LABELED for item in labeled_items)
        assert all(item.kind is ItemKind.PROBE for item in probe_items)
        assert all(item.phase is Phase.PRE for item in labeled_items[: episode.pre_count])
        assert all(item.phase is Phase.POST for item in post_labeled_items)
        assert all(item.phase is Phase.POST for item in probe_items)
        assert episode.shift_after_position == episode.pre_count
        assert episode.contradiction_count_post == build_contradiction_count_post(
            labeled_items,
            episode.pre_count,
            episode.rule_A,
            episode.rule_B,
        )
        assert episode.probe_label_counts == build_probe_label_counts(episode.probe_targets)
        assert episode.probe_sign_pattern_counts == build_probe_sign_pattern_counts(
            probe_items
        )

        for probe_item, target, metadata in zip(
            probe_items,
            episode.probe_targets,
            episode.probe_metadata,
        ):
            assert metadata.position == probe_item.position
            assert metadata.old_rule_label is label(
                episode.rule_A,
                probe_item.q1,
                probe_item.q2,
            )
            assert metadata.new_rule_label is label(
                episode.rule_B,
                probe_item.q1,
                probe_item.q2,
            )
            if probe_sign_pattern(probe_item.q1, probe_item.q2) in updated_sign_patterns:
                assert target is metadata.new_rule_label
            else:
                assert target is metadata.old_rule_label
            assert metadata.is_disagreement_probe is True


def test_score_episode_freezes_exact_match_partial_match_and_invalid_predictions() -> None:
    probe_targets = generate_episode(0).probe_targets

    assert score_episode(probe_targets, probe_targets) == (4, 4)
    assert score_episode(
        (
            InteractionLabel.REPEL,
            InteractionLabel.REPEL,
            InteractionLabel.ATTRACT,
            InteractionLabel.REPEL,
        ),
        probe_targets,
    ) == (3, 4)
    assert score_episode(("attract",), probe_targets) == (0, 4)


def test_public_label_mapping_preserves_underlying_correct_answers() -> None:
    probe_targets = generate_episode(0).probe_targets
    public_targets = tuple(format_public_label(target) for target in probe_targets)

    assert score_episode(public_targets, probe_targets) == (4, 4)
    assert score_episode(probe_targets, public_targets) == (4, 4)


def test_public_label_mapping_freezes_type_a_and_type_b_contract() -> None:
    assert PUBLIC_CONTRACT_VERSION == "markers-v1"
    assert format_public_label(InteractionLabel.ATTRACT) == "type_a"
    assert format_public_label(InteractionLabel.REPEL) == "type_b"
    assert parse_public_label("type_a") is InteractionLabel.ATTRACT
    assert parse_public_label("type_b") is InteractionLabel.REPEL


def test_internal_label_parser_rejects_public_output_labels() -> None:
    for public_label in ("type_a", "type_b"):
        try:
            parse_label(public_label)
        except ValueError:
            continue
        raise AssertionError(
            f"internal parse_label unexpectedly accepted locked public output label {public_label!r}"
        )


def test_reframed_prompt_preserves_original_item_level_episode_content_for_baseline_seeds() -> None:
    for row in _load_baseline_fixture():
        seed = row["seed"]
        episode = generate_episode(seed, split=Split.PUBLIC)
        original_prompt = _render_original_binary_prompt(episode)
        reframed_prompt = render_binary_prompt(episode)

        original_signature = _extract_original_prompt_signature(
            original_prompt,
            episode.template_family,
        )
        reframed_signature = _extract_reframed_prompt_signature(
            reframed_prompt,
            episode.template_family,
        )

        assert reframed_signature == original_signature, (
            f"seed {seed} changed item-level prompt content under reframing "
            f"for template_family={episode.template_family.value}"
        )


def test_baseline_seed_reframing_preserves_latent_episode_serialization() -> None:
    for row in _load_baseline_fixture():
        seed = row["seed"]
        episode = generate_episode(seed, split=Split.PUBLIC)

        assert _serialize_episode(episode) == {
            key: value for key, value in row.items() if key != "seed"
        }, f"seed {seed} changed latent serialized episode content"


def test_reframed_public_outputs_preserve_answer_key_and_scoring_for_baseline_seeds() -> None:
    for row in _load_baseline_fixture():
        seed = row["seed"]
        episode = generate_episode(seed, split=Split.PUBLIC)
        latent_targets = episode.probe_targets
        public_targets = tuple(format_public_label(target) for target in latent_targets)
        flipped_latent_predictions = _flip_latent_prediction(latent_targets)
        flipped_public_predictions = tuple(
            format_public_label(target) for target in flipped_latent_predictions
        )

        assert tuple(parse_public_label(target) for target in public_targets) == latent_targets, (
            f"seed {seed} changed answer-key mapping under public outputs"
        )
        assert score_episode(latent_targets, latent_targets) == score_episode(
            public_targets,
            latent_targets,
        ), f"seed {seed} changed perfect-match scoring under public outputs"
        assert score_episode(flipped_latent_predictions, latent_targets) == score_episode(
            flipped_public_predictions,
            latent_targets,
        ), f"seed {seed} changed partial-match scoring under public outputs"
