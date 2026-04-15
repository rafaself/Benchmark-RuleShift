import copy
import unittest

from scripts.build_cogflex_dataset import (
    IDENTIFIABILITY_KIND_ROUTED_ALL,
    IDENTIFIABILITY_KIND_SINGLE_LAST,
    PRIVATE_IDENTIFIABILITY_SPEC_BY_STRUCTURE,
    PUBLIC_IDENTIFIABILITY_SPEC_BY_TASK,
    PUBLIC_RULES,
    PUBLIC_STRUCTURES,
    build_episode_payload,
    build_public_artifacts,
    compute_identifiability,
    enumerate_items,
    identifiability_report_for_row,
    serialize_case,
)
from scripts.private_cogflex_bundle import (
    PRIVATE_VARIANTS_PER_FAMILY_TASK,
    PRIVATE_RULES,
    build_identifiable_private_episode,
)
from scripts.verify_cogflex import verify_identifiability


ACCEPT_R1 = PUBLIC_RULES["accept_r1_nonnegative"]
ACCEPT_ABS_SUM = PUBLIC_RULES["accept_abs_sum_large"]


def _accept_accept_structure():
    return PUBLIC_STRUCTURES["two_step_focus"]


def _build_ambiguous_single_rule_row():
    structure = _accept_accept_structure()
    agree_stims = [
        {"r1": 3, "r2": 3, "shape": "circle", "tone": "warm"},
        {"r1": 4, "r2": 2, "shape": "triangle", "tone": "cool"},
        {"r1": -1, "r2": -2, "shape": "square", "tone": "neutral"},
        {"r1": -2, "r2": 2, "shape": "hex", "tone": "cool"},
    ]
    for stimulus in agree_stims:
        assert ACCEPT_R1.label(stimulus) == ACCEPT_ABS_SUM.label(stimulus), stimulus

    evidence_turn_one = enumerate_items(agree_stims, ACCEPT_R1)
    evidence_turn_two = enumerate_items(agree_stims, ACCEPT_R1)
    probe_stims = [
        {"r1": 5, "r2": 0, "shape": "circle", "tone": "warm"},
        {"r1": -5, "r2": 4, "shape": "square", "tone": "cool"},
        {"r1": 1, "r2": -1, "shape": "triangle", "tone": "neutral"},
        {"r1": -2, "r2": 2, "shape": "hex", "tone": "warm"},
        {"r1": 4, "r2": -1, "shape": "square", "tone": "cool"},
    ]
    for probe in probe_stims:
        if ACCEPT_R1.label(probe) != ACCEPT_ABS_SUM.label(probe):
            break
    else:
        raise AssertionError("probe stimuli must disambiguate between candidate rules")

    probes = enumerate_items(probe_stims, ACCEPT_R1)
    row, _answer = build_episode_payload(
        "ambig01",
        suite_task_id="explicit_rule_update",
        structure=structure,
        label_vocab=ACCEPT_R1.label_vocab,
        turn_prompts=[
            "Infer the current rule from these labeled examples.",
            "Continue collecting evidence before the upcoming update.",
            "Apply the active rule to every probe.",
        ],
        turn_items=[evidence_turn_one, evidence_turn_two, probes],
        probe_annotations=["congruent"] * len(probes),
    )
    row["analysis"]["difficulty_bin"] = "hard"
    return row


def _build_identifiable_single_rule_row():
    structure = _accept_accept_structure()
    disambiguating_stims = [
        {"r1": 5, "r2": 0, "shape": "circle", "tone": "warm"},
        {"r1": 1, "r2": -1, "shape": "triangle", "tone": "cool"},
        {"r1": -1, "r2": -1, "shape": "square", "tone": "neutral"},
        {"r1": -4, "r2": -2, "shape": "hex", "tone": "cool"},
    ]
    mismatches = sum(
        1
        for stimulus in disambiguating_stims
        if ACCEPT_R1.label(stimulus) != ACCEPT_ABS_SUM.label(stimulus)
    )
    assert mismatches >= 1, "last evidence turn must distinguish the two candidates"

    evidence_turn_one = enumerate_items(disambiguating_stims, ACCEPT_R1)
    evidence_turn_two = enumerate_items(disambiguating_stims, ACCEPT_R1)
    probe_stims = [
        {"r1": 5, "r2": 0, "shape": "circle", "tone": "warm"},
        {"r1": -5, "r2": 4, "shape": "square", "tone": "cool"},
        {"r1": 1, "r2": -1, "shape": "triangle", "tone": "neutral"},
        {"r1": -2, "r2": 2, "shape": "hex", "tone": "warm"},
        {"r1": 4, "r2": -1, "shape": "square", "tone": "cool"},
    ]
    probes = enumerate_items(probe_stims, ACCEPT_R1)
    row, _answer = build_episode_payload(
        "idable01",
        suite_task_id="explicit_rule_update",
        structure=structure,
        label_vocab=ACCEPT_R1.label_vocab,
        turn_prompts=[
            "Infer the current rule from these labeled examples.",
            "Continue collecting evidence before the upcoming update.",
            "Apply the active rule to every probe.",
        ],
        turn_items=[evidence_turn_one, evidence_turn_two, probes],
        probe_annotations=["congruent"] * len(probes),
    )
    row["analysis"]["difficulty_bin"] = "hard"
    return row


def _build_ambiguous_routed_row():
    context_terms = ("alpha", "beta")
    primary_rule = PUBLIC_RULES["accept_shape_round"]
    confounding_rule = PUBLIC_RULES["accept_parity_match"]
    secondary_rule = PUBLIC_RULES["accept_abs_sum_large"]

    alpha_agree_stims = [
        {"r1": 2, "r2": 4, "shape": "circle", "tone": "warm"},
        {"r1": 1, "r2": 3, "shape": "hex", "tone": "cool"},
        {"r1": 1, "r2": 2, "shape": "square", "tone": "neutral"},
        {"r1": 2, "r2": 1, "shape": "triangle", "tone": "cool"},
    ]
    for stimulus in alpha_agree_stims:
        assert primary_rule.label(stimulus) == confounding_rule.label(stimulus), stimulus

    beta_stims = [
        {"r1": 4, "r2": 3, "shape": "circle", "tone": "warm"},
        {"r1": -3, "r2": -4, "shape": "square", "tone": "cool"},
        {"r1": 1, "r2": 2, "shape": "triangle", "tone": "neutral"},
        {"r1": -2, "r2": 1, "shape": "hex", "tone": "cool"},
    ]

    turn_items_one = [
        serialize_case(index, stimulus, primary_rule.label(stimulus), context=context_terms[0], rule_id=primary_rule.rule_id)
        for index, stimulus in enumerate(alpha_agree_stims, start=1)
    ]
    turn_items_two = [
        serialize_case(index, stimulus, secondary_rule.label(stimulus), context=context_terms[1], rule_id=secondary_rule.rule_id)
        for index, stimulus in enumerate(beta_stims, start=1)
    ]

    probe_stims_alpha = [
        {"r1": 1, "r2": 2, "shape": "circle", "tone": "warm"},
        {"r1": 2, "r2": 1, "shape": "hex", "tone": "neutral"},
    ]
    assert any(
        primary_rule.label(stimulus) != confounding_rule.label(stimulus)
        for stimulus in probe_stims_alpha
    ), "alpha probes must distinguish the primary rule from the confound"

    probes: list[dict[str, object]] = []
    for index, stimulus in enumerate(probe_stims_alpha, start=1):
        probes.append(
            serialize_case(
                index,
                stimulus,
                primary_rule.label(stimulus),
                context=context_terms[0],
                rule_id=primary_rule.rule_id,
            )
        )
    for index, stimulus in enumerate(beta_stims[:2], start=len(probes) + 1):
        probes.append(
            serialize_case(
                index,
                stimulus,
                secondary_rule.label(stimulus),
                context=context_terms[1],
                rule_id=secondary_rule.rule_id,
            )
        )

    structure = PUBLIC_STRUCTURES["two_step_focus"]
    row, _answer = build_episode_payload(
        "ambigctx01",
        suite_task_id="context_binding",
        structure=structure,
        label_vocab=primary_rule.label_vocab,
        turn_prompts=[
            f"Learn the rule bound to context={context_terms[0]}.",
            f"Now learn the rule bound to context={context_terms[1]}.",
            "For each probe, use its context to select the right rule before labeling it.",
        ],
        turn_items=[turn_items_one, turn_items_two, probes],
        probe_annotations=["congruent"] * len(probes),
    )
    row["analysis"]["difficulty_bin"] = "hard"
    return row


class ComputeIdentifiabilityTests(unittest.TestCase):
    def test_single_rule_ambiguous_row_is_rejected(self) -> None:
        row = _build_ambiguous_single_rule_row()
        report = compute_identifiability(
            row,
            rule_catalogue=PUBLIC_RULES,
            kind=IDENTIFIABILITY_KIND_SINGLE_LAST,
        )
        self.assertFalse(report["is_identifiable"])
        self.assertGreaterEqual(report["consistent_hypothesis_count"], 2)
        self.assertGreaterEqual(report["distinct_probe_target_count"], 2)

    def test_single_rule_identifiable_row_is_accepted(self) -> None:
        row = _build_identifiable_single_rule_row()
        report = compute_identifiability(
            row,
            rule_catalogue=PUBLIC_RULES,
            kind=IDENTIFIABILITY_KIND_SINGLE_LAST,
        )
        self.assertTrue(report["is_identifiable"])
        self.assertEqual(report["distinct_probe_target_count"], 1)
        self.assertGreaterEqual(report["consistent_hypothesis_count"], 1)

    def test_routed_ambiguous_row_is_rejected(self) -> None:
        row = _build_ambiguous_routed_row()
        report = compute_identifiability(
            row,
            rule_catalogue=PUBLIC_RULES,
            kind=IDENTIFIABILITY_KIND_ROUTED_ALL,
            route_field="context",
        )
        self.assertFalse(report["is_identifiable"])
        self.assertGreaterEqual(report["distinct_probe_target_count"], 2)


class GeneratedEpisodeIdentifiabilityTests(unittest.TestCase):
    def test_every_generated_public_episode_is_identifiable(self) -> None:
        rows, _answers, _report = build_public_artifacts()
        for row in rows:
            report = identifiability_report_for_row(row, split="public", rule_catalogue=PUBLIC_RULES)
            self.assertTrue(
                report["is_identifiable"],
                msg=f"episode {row['episode_id']} is ambiguous: {report}",
            )

    def test_every_generated_private_episode_is_identifiable(self) -> None:
        from scripts.build_cogflex_dataset import SUITE_TASKS
        from scripts.private_cogflex_bundle import REQUIRED_PRIVATE_STRUCTURE_FAMILY_IDS

        episode_number = 1
        for family_index, structure_family_id in enumerate(REQUIRED_PRIVATE_STRUCTURE_FAMILY_IDS):
            for task_index, suite_task_id in enumerate(SUITE_TASKS):
                for variant_index in range(PRIVATE_VARIANTS_PER_FAMILY_TASK):
                    episode_id = f"p{episode_number:04d}"
                    row, _answer, report = build_identifiable_private_episode(
                        episode_id,
                        suite_task_id,
                        structure_family_id,
                        variant=(
                            family_index * len(SUITE_TASKS) * PRIVATE_VARIANTS_PER_FAMILY_TASK
                            + task_index * PRIVATE_VARIANTS_PER_FAMILY_TASK
                            + variant_index
                        ),
                    )
                    self.assertTrue(report["is_identifiable"])
                    episode_number += 1

    def test_public_identifiability_specs_cover_every_suite_task(self) -> None:
        from scripts.build_cogflex_dataset import SUITE_TASKS

        self.assertEqual(set(PUBLIC_IDENTIFIABILITY_SPEC_BY_TASK), set(SUITE_TASKS))

    def test_private_identifiability_specs_cover_every_required_structure(self) -> None:
        from scripts.private_cogflex_bundle import REQUIRED_PRIVATE_STRUCTURE_FAMILY_IDS

        self.assertEqual(
            set(PRIVATE_IDENTIFIABILITY_SPEC_BY_STRUCTURE),
            set(REQUIRED_PRIVATE_STRUCTURE_FAMILY_IDS),
        )

    def test_private_rule_catalogue_is_non_empty(self) -> None:
        self.assertGreaterEqual(len(PRIVATE_RULES), 2)


class VerifyIdentifiabilityTests(unittest.TestCase):
    def test_verify_public_identifiability_passes_on_generated_rows(self) -> None:
        rows, _answers, _report = build_public_artifacts()
        summary = verify_identifiability(rows, split="public")
        self.assertEqual(summary["ambiguous_episode_count"], 0)
        self.assertEqual(summary["identifiability_episode_count"], len(rows))

    def test_verify_public_identifiability_rejects_injected_ambiguous_row(self) -> None:
        rows, _answers, _report = build_public_artifacts()
        poisoned = copy.deepcopy(rows)
        poisoned[0] = _build_ambiguous_single_rule_row()
        with self.assertRaisesRegex(RuntimeError, "identifiability check failed"):
            verify_identifiability(poisoned, split="public")

    def test_verify_private_identifiability_rejects_injected_ambiguous_row(self) -> None:
        poisoned_row = _build_ambiguous_single_rule_row()
        poisoned_row["analysis"]["structure_family_id"] = "delayed_reversal"
        with self.assertRaisesRegex(RuntimeError, "identifiability check failed"):
            verify_identifiability([poisoned_row], split="private")


if __name__ == "__main__":
    unittest.main()
