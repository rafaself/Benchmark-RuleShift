"""Tests for core.invariance — perturbation functions, case generation, and reporting.

Acceptance criteria verified:
- At least one minimal-pair invariance case per perturbation class.
- Perturbations produce prompts that differ from the canonical prompt.
- probe_targets are identical to the originating episode across all classes.
- Cases are explicitly versioned with INVARIANCE_VERSION.
- InvarianceReport exposes per-class accuracy separately from aggregate accuracy.
- Binary (probe_targets) labels remain unchanged across all perturbations.
"""
from __future__ import annotations

import pytest

from core.invariance import (
    INVARIANCE_VERSION,
    PERTURBATION_CLASS_ORDER,
    InvarianceCase,
    InvarianceReport,
    PerturbationClass,
    PerturbationClassAccuracy,
    apply_layout_reformat,
    apply_neutral_renaming,
    apply_non_causal_ordering,
    apply_wording_paraphrase,
    build_invariance_report,
    generate_invariance_cases,
)
from core.metrics import MetricSummary
from generator import generate_episode
from parser import ParsedPrediction, ParseStatus
from protocol import PROBE_COUNT, InteractionLabel
from tasks.ruleshift_benchmark.render import render_binary_prompt
from tasks.ruleshift_benchmark.protocol import TemplateFamily, TemplateId

ATTRACT = InteractionLabel.ATTRACT
REPEL = InteractionLabel.REPEL

_FOUR_LABELS = (ATTRACT, REPEL, ATTRACT, REPEL)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_prediction(*labels: InteractionLabel) -> ParsedPrediction:
    return ParsedPrediction(labels=labels, status=ParseStatus.VALID)


def _invalid_prediction() -> ParsedPrediction:
    return ParsedPrediction(labels=(), status=ParseStatus.INVALID)


def _skipped_prediction() -> ParsedPrediction:
    return ParsedPrediction.skipped_provider_failure()


def _episode_canonical() -> object:
    """Return a canonical-family episode (T1 or T2, either template family canonical)."""
    # Iterate seeds until we find a CANONICAL template_family episode.
    for seed in range(100):
        ep = generate_episode(seed)
        if ep.template_family is TemplateFamily.CANONICAL:
            return ep
    raise RuntimeError("No CANONICAL episode found in first 100 seeds")  # pragma: no cover


def _episode_obs_log() -> object:
    """Return an observation_log-family episode."""
    for seed in range(100):
        ep = generate_episode(seed)
        if ep.template_family is TemplateFamily.OBSERVATION_LOG:
            return ep
    raise RuntimeError("No OBSERVATION_LOG episode in first 100 seeds")  # pragma: no cover


def _episode_t1() -> object:
    """Return a T1 (shift_after_position=2) episode."""
    for seed in range(100):
        ep = generate_episode(seed)
        if ep.template_id is TemplateId.T1:
            return ep
    raise RuntimeError("No T1 episode found in first 100 seeds")  # pragma: no cover


def _episode_t2() -> object:
    """Return a T2 (shift_after_position=3) episode."""
    for seed in range(100):
        ep = generate_episode(seed)
        if ep.template_id is TemplateId.T2:
            return ep
    raise RuntimeError("No T2 episode found in first 100 seeds")  # pragma: no cover


# ---------------------------------------------------------------------------
# PERTURBATION_CLASS_ORDER constant
# ---------------------------------------------------------------------------


def test_perturbation_class_order_has_four_canonical_entries():
    assert PERTURBATION_CLASS_ORDER == (
        "wording_paraphrase",
        "layout_reformat",
        "neutral_renaming",
        "non_causal_ordering",
    )


# ---------------------------------------------------------------------------
# apply_wording_paraphrase
# ---------------------------------------------------------------------------


def test_wording_paraphrase_changes_binary_outro():
    """The binary outro 'Return exactly 4 labels' is replaced with 'Output'."""
    prompt = "Return exactly 4 labels in order, one per probe. Use only attract or repel."
    result = apply_wording_paraphrase(prompt)
    assert "Output exactly 4 labels in order, one per probe." in result
    assert "Return exactly 4 labels in order, one per probe." not in result


def test_wording_paraphrase_changes_canonical_intro_line():
    """'Each labeled line shows' is replaced in canonical prompts."""
    prompt = "Each labeled line shows q1, q2, and the observed result."
    result = apply_wording_paraphrase(prompt)
    assert "Each entry shows q1, q2, and the observed result." in result
    assert "Each labeled line shows" not in result


def test_wording_paraphrase_changes_obs_log_intro_line():
    """'Each entry records' is replaced in observation-log prompts."""
    prompt = "Each entry records q1, q2, and the observed outcome."
    result = apply_wording_paraphrase(prompt)
    assert "Each entry lists q1, q2, and the observed outcome." in result
    assert "Each entry records" not in result


def test_wording_paraphrase_preserves_attract_repel_labels():
    """The label words 'attract' and 'repel' must be preserved exactly."""
    prompt = (
        "1. q1=+1, q2=-2 -> attract\n"
        "2. q1=-3, q2=+1 -> repel\n"
        "Return exactly 4 labels in order, one per probe. Use only attract or repel."
    )
    result = apply_wording_paraphrase(prompt)
    assert "attract" in result
    assert "repel" in result


def test_wording_paraphrase_is_deterministic():
    prompt = "Return exactly 4 labels in order, one per probe. Use only attract or repel."
    assert apply_wording_paraphrase(prompt) == apply_wording_paraphrase(prompt)


def test_wording_paraphrase_changes_full_canonical_prompt():
    """Full rendered canonical binary prompt is changed by the paraphrase."""
    ep = _episode_canonical()
    prompt = render_binary_prompt(ep)
    result = apply_wording_paraphrase(prompt)
    assert result != prompt


def test_wording_paraphrase_changes_full_obs_log_prompt():
    """Full rendered observation-log binary prompt is changed by the paraphrase."""
    ep = _episode_obs_log()
    prompt = render_binary_prompt(ep)
    result = apply_wording_paraphrase(prompt)
    assert result != prompt


# ---------------------------------------------------------------------------
# apply_layout_reformat
# ---------------------------------------------------------------------------


def test_layout_reformat_adds_separator_before_canonical_probes():
    """'---' separator inserted before 'Probes:' heading in canonical format."""
    prompt = "Labeled examples:\n1. q1=+1, q2=-2 -> attract\n\nProbes:\n6. q1=+1, q2=+3 -> ?"
    result = apply_layout_reformat(prompt)
    assert "---\nProbes:" in result
    assert "Probes:" in result


def test_layout_reformat_adds_separator_before_obs_log_probes():
    """'---' separator inserted before 'Unresolved probe entries:' in obs-log format."""
    prompt = (
        "Resolved log entries:\n[01] q1=+1 | q2=-2 | observed=attract\n\n"
        "Unresolved probe entries:\n[06] q1=+1 | q2=+3 | observed=?"
    )
    result = apply_layout_reformat(prompt)
    assert "---\nUnresolved probe entries:" in result


def test_layout_reformat_does_not_change_item_data():
    """Item charge values and labels are unchanged after layout reformat."""
    prompt = "1. q1=+1, q2=-2 -> attract\n\nProbes:\n6. q1=+2, q2=-3 -> ?"
    result = apply_layout_reformat(prompt)
    assert "q1=+1, q2=-2 -> attract" in result
    assert "q1=+2, q2=-3 -> ?" in result


def test_layout_reformat_is_deterministic():
    prompt = "item\n\nProbes:\nprobe"
    assert apply_layout_reformat(prompt) == apply_layout_reformat(prompt)


def test_layout_reformat_changes_full_canonical_prompt():
    ep = _episode_canonical()
    prompt = render_binary_prompt(ep)
    assert apply_layout_reformat(prompt) != prompt


def test_layout_reformat_changes_full_obs_log_prompt():
    ep = _episode_obs_log()
    prompt = render_binary_prompt(ep)
    assert apply_layout_reformat(prompt) != prompt


# ---------------------------------------------------------------------------
# apply_neutral_renaming
# ---------------------------------------------------------------------------


def test_neutral_renaming_renames_canonical_labeled_heading():
    prompt = "Labeled examples:\n1. q1=+1, q2=-2 -> attract"
    result = apply_neutral_renaming(prompt)
    assert "Training examples:" in result
    assert "Labeled examples:" not in result


def test_neutral_renaming_renames_canonical_probe_heading():
    prompt = "Probes:\n6. q1=+1, q2=+3 -> ?"
    result = apply_neutral_renaming(prompt)
    assert "Test cases:" in result
    assert "Probes:" not in result


def test_neutral_renaming_renames_obs_log_resolved_heading():
    prompt = "Resolved log entries:\n[01] q1=+1 | q2=-2 | observed=attract"
    result = apply_neutral_renaming(prompt)
    assert "Confirmed log entries:" in result
    assert "Resolved log entries:" not in result


def test_neutral_renaming_renames_obs_log_unresolved_heading():
    prompt = "Unresolved probe entries:\n[06] q1=+1 | q2=+3 | observed=?"
    result = apply_neutral_renaming(prompt)
    assert "Pending entries:" in result
    assert "Unresolved probe entries:" not in result


def test_neutral_renaming_does_not_change_charge_values():
    """Charge values and labels are unaffected by heading renames."""
    prompt = "Labeled examples:\n1. q1=+1, q2=-2 -> attract\nProbes:\n6. q1=-3, q2=+2 -> ?"
    result = apply_neutral_renaming(prompt)
    assert "q1=+1, q2=-2 -> attract" in result
    assert "q1=-3, q2=+2 -> ?" in result


def test_neutral_renaming_is_deterministic():
    prompt = "Labeled examples:\nProbes:"
    assert apply_neutral_renaming(prompt) == apply_neutral_renaming(prompt)


def test_neutral_renaming_changes_full_canonical_prompt():
    ep = _episode_canonical()
    prompt = render_binary_prompt(ep)
    assert apply_neutral_renaming(prompt) != prompt


def test_neutral_renaming_changes_full_obs_log_prompt():
    ep = _episode_obs_log()
    prompt = render_binary_prompt(ep)
    assert apply_neutral_renaming(prompt) != prompt


# ---------------------------------------------------------------------------
# apply_non_causal_ordering
# ---------------------------------------------------------------------------


def test_non_causal_ordering_swaps_two_preshift_items_canonical():
    """T1 (pre_shift_count=2): items 1 and 2 swap their body content."""
    prompt = (
        "Labeled examples:\n"
        "1. q1=+1, q2=-2 -> attract\n"
        "2. q1=-3, q2=+1 -> repel\n"
        "3. q1=+2, q2=-1 -> attract\n"
        "\nProbes:\n"
        "6. q1=+1, q2=+3 -> ?"
    )
    result = apply_non_causal_ordering(prompt, pre_shift_count=2)
    lines = result.split("\n")
    # Find lines starting with "1." and "2."
    line1 = next(l for l in lines if l.startswith("1."))
    line2 = next(l for l in lines if l.startswith("2."))
    # Swap: position 1 now has the old body of position 2, and vice versa.
    assert "q1=-3, q2=+1 -> repel" in line1
    assert "q1=+1, q2=-2 -> attract" in line2


def test_non_causal_ordering_reverses_three_preshift_items_canonical():
    """T2 (pre_shift_count=3): items 1, 2, 3 are reversed."""
    prompt = (
        "Labeled examples:\n"
        "1. q1=+1, q2=-2 -> attract\n"
        "2. q1=-3, q2=+1 -> repel\n"
        "3. q1=+2, q2=-1 -> attract\n"
        "4. q1=-1, q2=+3 -> repel\n"
        "5. q1=+3, q2=-2 -> attract\n"
        "\nProbes:\n"
        "6. q1=+1, q2=+3 -> ?"
    )
    result = apply_non_causal_ordering(prompt, pre_shift_count=3)
    lines = result.split("\n")
    line1 = next(l for l in lines if l.startswith("1."))
    line2 = next(l for l in lines if l.startswith("2."))
    line3 = next(l for l in lines if l.startswith("3."))
    # Reversed: 1←3, 2 unchanged, 3←1
    assert "q1=+2, q2=-1 -> attract" in line1
    assert "q1=-3, q2=+1 -> repel" in line2
    assert "q1=+1, q2=-2 -> attract" in line3


def test_non_causal_ordering_preserves_post_shift_labeled_items():
    """Items beyond pre_shift_count are not moved."""
    prompt = (
        "Labeled examples:\n"
        "1. q1=+1, q2=-2 -> attract\n"
        "2. q1=-3, q2=+1 -> repel\n"
        "3. q1=+2, q2=-1 -> attract\n"
        "4. q1=-1, q2=+3 -> repel\n"
        "5. q1=+3, q2=-2 -> attract\n"
    )
    result = apply_non_causal_ordering(prompt, pre_shift_count=2)
    lines = result.split("\n")
    line3 = next(l for l in lines if l.startswith("3."))
    line4 = next(l for l in lines if l.startswith("4."))
    line5 = next(l for l in lines if l.startswith("5."))
    assert "q1=+2, q2=-1 -> attract" in line3
    assert "q1=-1, q2=+3 -> repel" in line4
    assert "q1=+3, q2=-2 -> attract" in line5


def test_non_causal_ordering_preserves_probe_items():
    """Probe items (positions 6–9) are not moved."""
    prompt = (
        "1. q1=+1, q2=-2 -> attract\n"
        "2. q1=-3, q2=+1 -> repel\n"
        "\nProbes:\n"
        "6. q1=+1, q2=+3 -> ?\n"
        "7. q1=-2, q2=-1 -> ?\n"
    )
    result = apply_non_causal_ordering(prompt, pre_shift_count=2)
    lines = result.split("\n")
    line6 = next(l for l in lines if l.startswith("6."))
    line7 = next(l for l in lines if l.startswith("7."))
    assert "q1=+1, q2=+3 -> ?" in line6
    assert "q1=-2, q2=-1 -> ?" in line7


def test_non_causal_ordering_swaps_obs_log_items():
    """Observation-log format: [01] and [02] body content swaps for pre_shift_count=2."""
    prompt = (
        "Resolved log entries:\n"
        "[01] q1=+1 | q2=-2 | observed=attract\n"
        "[02] q1=-3 | q2=+1 | observed=repel\n"
        "[03] q1=+2 | q2=-1 | observed=attract\n"
    )
    result = apply_non_causal_ordering(prompt, pre_shift_count=2)
    lines = result.split("\n")
    line01 = next(l for l in lines if l.startswith("[01]"))
    line02 = next(l for l in lines if l.startswith("[02]"))
    assert "q1=-3 | q2=+1 | observed=repel" in line01
    assert "q1=+1 | q2=-2 | observed=attract" in line02


def test_non_causal_ordering_obs_log_preserves_probe_items():
    """Observation-log probe items ([06]–[09]) are not moved."""
    prompt = (
        "[01] q1=+1 | q2=-2 | observed=attract\n"
        "[02] q1=-3 | q2=+1 | observed=repel\n"
        "[06] q1=+1 | q2=+3 | observed=?\n"
    )
    result = apply_non_causal_ordering(prompt, pre_shift_count=2)
    lines = result.split("\n")
    line06 = next(l for l in lines if l.startswith("[06]"))
    assert "q1=+1 | q2=+3 | observed=?" in line06


def test_non_causal_ordering_is_deterministic():
    prompt = (
        "1. q1=+1, q2=-2 -> attract\n"
        "2. q1=-3, q2=+1 -> repel\n"
    )
    r1 = apply_non_causal_ordering(prompt, pre_shift_count=2)
    r2 = apply_non_causal_ordering(prompt, pre_shift_count=2)
    assert r1 == r2


def test_non_causal_ordering_changes_full_t1_prompt():
    """T1 episode canonical prompt is changed by non-causal ordering."""
    ep = _episode_t1()
    assert ep.shift_after_position == 2
    prompt = render_binary_prompt(ep)
    result = apply_non_causal_ordering(prompt, pre_shift_count=ep.shift_after_position)
    assert result != prompt


def test_non_causal_ordering_changes_full_t2_prompt():
    """T2 episode canonical prompt is changed by non-causal ordering."""
    ep = _episode_t2()
    assert ep.shift_after_position == 3
    prompt = render_binary_prompt(ep)
    result = apply_non_causal_ordering(prompt, pre_shift_count=ep.shift_after_position)
    assert result != prompt


# ---------------------------------------------------------------------------
# Minimal-pair invariant property: probe_targets unchanged across perturbations
# ---------------------------------------------------------------------------


def test_minimal_pair_wording_paraphrase_preserves_probe_targets():
    """Wording-paraphrase case has the same probe_targets as the episode."""
    ep = generate_episode(0)
    cases = generate_invariance_cases([ep])
    wp_case = next(c for c in cases if c.perturbation_class is PerturbationClass.WORDING_PARAPHRASE)
    assert wp_case.probe_targets == ep.probe_targets


def test_minimal_pair_layout_reformat_preserves_probe_targets():
    """Layout-reformat case has the same probe_targets as the episode."""
    ep = generate_episode(1)
    cases = generate_invariance_cases([ep])
    lr_case = next(c for c in cases if c.perturbation_class is PerturbationClass.LAYOUT_REFORMAT)
    assert lr_case.probe_targets == ep.probe_targets


def test_minimal_pair_neutral_renaming_preserves_probe_targets():
    """Neutral-renaming case has the same probe_targets as the episode."""
    ep = generate_episode(2)
    cases = generate_invariance_cases([ep])
    nr_case = next(c for c in cases if c.perturbation_class is PerturbationClass.NEUTRAL_RENAMING)
    assert nr_case.probe_targets == ep.probe_targets


def test_minimal_pair_non_causal_ordering_preserves_probe_targets():
    """Non-causal-ordering case has the same probe_targets as the episode."""
    ep = generate_episode(3)
    cases = generate_invariance_cases([ep])
    nco_case = next(c for c in cases if c.perturbation_class is PerturbationClass.NON_CAUSAL_ORDERING)
    assert nco_case.probe_targets == ep.probe_targets


def test_minimal_pair_all_classes_preserve_probe_targets_multiple_episodes():
    """probe_targets preserved for all 4 classes across a batch of episodes."""
    episodes = [generate_episode(seed) for seed in range(8)]
    cases = generate_invariance_cases(episodes)
    for case in cases:
        ep = next(e for e in episodes if e.episode_id == case.episode_id)
        assert case.probe_targets == ep.probe_targets, (
            f"probe_targets mismatch for {case.episode_id} "
            f"under {case.perturbation_class}"
        )


# ---------------------------------------------------------------------------
# generate_invariance_cases
# ---------------------------------------------------------------------------


def test_generate_invariance_cases_produces_four_cases_per_episode():
    ep = generate_episode(0)
    cases = generate_invariance_cases([ep])
    assert len(cases) == 4


def test_generate_invariance_cases_one_case_per_perturbation_class():
    ep = generate_episode(0)
    cases = generate_invariance_cases([ep])
    classes = [c.perturbation_class for c in cases]
    assert set(classes) == set(PerturbationClass)


def test_generate_invariance_cases_class_order_matches_perturbation_class_order():
    ep = generate_episode(0)
    cases = generate_invariance_cases([ep])
    assert [c.perturbation_class.value for c in cases] == list(PERTURBATION_CLASS_ORDER)


def test_generate_invariance_cases_episode_id_matches():
    ep = generate_episode(42)
    cases = generate_invariance_cases([ep])
    assert all(c.episode_id == ep.episode_id for c in cases)


def test_generate_invariance_cases_all_versioned_with_invariance_version():
    ep = generate_episode(0)
    cases = generate_invariance_cases([ep])
    assert all(c.perturbation_version == INVARIANCE_VERSION for c in cases)


def test_generate_invariance_cases_perturbed_differs_from_canonical():
    """Every generated case must have a perturbed_prompt ≠ canonical_prompt."""
    ep = generate_episode(0)
    cases = generate_invariance_cases([ep])
    for case in cases:
        assert case.perturbed_prompt != case.canonical_prompt, (
            f"Perturbation {case.perturbation_class} produced no change for {case.episode_id}"
        )


def test_generate_invariance_cases_deterministic():
    ep = generate_episode(7)
    cases1 = generate_invariance_cases([ep])
    cases2 = generate_invariance_cases([ep])
    for c1, c2 in zip(cases1, cases2):
        assert c1.perturbed_prompt == c2.perturbed_prompt
        assert c1.probe_targets == c2.probe_targets


def test_generate_invariance_cases_canonical_prompt_matches_render():
    ep = generate_episode(5)
    cases = generate_invariance_cases([ep])
    expected_canonical = render_binary_prompt(ep)
    assert all(c.canonical_prompt == expected_canonical for c in cases)


def test_generate_invariance_cases_scales_with_episodes():
    episodes = [generate_episode(s) for s in range(3)]
    cases = generate_invariance_cases(episodes)
    assert len(cases) == 3 * 4  # 3 episodes × 4 perturbation classes


def test_generate_invariance_cases_empty_produces_no_cases():
    cases = generate_invariance_cases([])
    assert cases == []


def test_generate_invariance_cases_is_frozen():
    ep = generate_episode(0)
    cases = generate_invariance_cases([ep])
    case = cases[0]
    with pytest.raises((AttributeError, TypeError)):
        case.episode_id = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# PerturbationClassAccuracy
# ---------------------------------------------------------------------------


def test_perturbation_class_accuracy_property():
    acc = PerturbationClassAccuracy(
        perturbation_class="wording_paraphrase",
        episode_count=4,
        correct_probes=12,
        total_probes=16,
    )
    assert acc.accuracy == pytest.approx(12 / 16)


def test_perturbation_class_accuracy_zero_total_returns_zero():
    acc = PerturbationClassAccuracy(
        perturbation_class="layout_reformat",
        episode_count=0,
        correct_probes=0,
        total_probes=0,
    )
    assert acc.accuracy == 0.0


def test_perturbation_class_accuracy_to_dict():
    acc = PerturbationClassAccuracy(
        perturbation_class="neutral_renaming",
        episode_count=2,
        correct_probes=6,
        total_probes=8,
    )
    d = acc.to_dict()
    assert d["perturbation_class"] == "neutral_renaming"
    assert d["episode_count"] == 2
    assert d["correct_probes"] == 6
    assert d["total_probes"] == 8
    assert d["accuracy"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# build_invariance_report
# ---------------------------------------------------------------------------


def _make_case(
    episode_id: str,
    pc: PerturbationClass,
    targets: tuple[InteractionLabel, ...] = _FOUR_LABELS,
) -> InvarianceCase:
    return InvarianceCase(
        episode_id=episode_id,
        perturbation_class=pc,
        canonical_prompt="canonical",
        perturbed_prompt="perturbed",
        probe_targets=targets,
        perturbation_version=INVARIANCE_VERSION,
    )


def test_build_invariance_report_empty_input():
    report = build_invariance_report([])
    assert report.version == INVARIANCE_VERSION
    for _, acc in report.by_class:
        assert acc.episode_count == 0
        assert acc.correct_probes == 0
        assert acc.total_probes == 0
        assert acc.accuracy == 0.0


def test_build_invariance_report_all_correct():
    cases_preds = [
        (_make_case("ep-1", PerturbationClass.WORDING_PARAPHRASE, _FOUR_LABELS),
         _valid_prediction(*_FOUR_LABELS)),
    ]
    report = build_invariance_report(cases_preds)
    acc_dict = dict(report.by_class)
    wp = acc_dict["wording_paraphrase"]
    assert wp.correct_probes == PROBE_COUNT
    assert wp.total_probes == PROBE_COUNT
    assert wp.accuracy == 1.0


def test_build_invariance_report_invalid_prediction_counts_zero():
    cases_preds = [
        (_make_case("ep-1", PerturbationClass.LAYOUT_REFORMAT, _FOUR_LABELS),
         _invalid_prediction()),
    ]
    report = build_invariance_report(cases_preds)
    acc_dict = dict(report.by_class)
    assert acc_dict["layout_reformat"].correct_probes == 0
    assert acc_dict["layout_reformat"].total_probes == PROBE_COUNT


def test_build_invariance_report_skipped_prediction_counts_zero():
    cases_preds = [
        (_make_case("ep-1", PerturbationClass.NEUTRAL_RENAMING, _FOUR_LABELS),
         _skipped_prediction()),
    ]
    report = build_invariance_report(cases_preds)
    acc_dict = dict(report.by_class)
    assert acc_dict["neutral_renaming"].correct_probes == 0


def test_build_invariance_report_aggregates_multiple_episodes():
    cases_preds = [
        (_make_case("ep-1", PerturbationClass.NON_CAUSAL_ORDERING, _FOUR_LABELS),
         _valid_prediction(*_FOUR_LABELS)),   # all correct
        (_make_case("ep-2", PerturbationClass.NON_CAUSAL_ORDERING, _FOUR_LABELS),
         _valid_prediction(REPEL, REPEL, ATTRACT, REPEL)),  # 3 correct
    ]
    report = build_invariance_report(cases_preds)
    acc_dict = dict(report.by_class)
    nco = acc_dict["non_causal_ordering"]
    assert nco.episode_count == 2
    assert nco.correct_probes == PROBE_COUNT + 3  # 4 + 3
    assert nco.total_probes == 2 * PROBE_COUNT


def test_build_invariance_report_canonical_class_order():
    """by_class entries are in PERTURBATION_CLASS_ORDER."""
    cases_preds = [
        (_make_case("ep-1", pc), _valid_prediction(*_FOUR_LABELS))
        for pc in PerturbationClass
    ]
    report = build_invariance_report(cases_preds)
    keys = [k for k, _ in report.by_class]
    assert keys == list(PERTURBATION_CLASS_ORDER)


def test_build_invariance_report_all_four_classes_always_present():
    """All four perturbation classes appear even when some have no cases."""
    cases_preds = [
        (_make_case("ep-1", PerturbationClass.WORDING_PARAPHRASE), _valid_prediction(*_FOUR_LABELS)),
    ]
    report = build_invariance_report(cases_preds)
    keys = [k for k, _ in report.by_class]
    assert set(keys) == {pc.value for pc in PerturbationClass}


def test_build_invariance_report_version_is_invariance_version():
    report = build_invariance_report([])
    assert report.version == INVARIANCE_VERSION


def test_build_invariance_report_to_dict_structure():
    cases_preds = [
        (_make_case("ep-1", PerturbationClass.WORDING_PARAPHRASE), _valid_prediction(*_FOUR_LABELS)),
    ]
    report = build_invariance_report(cases_preds)
    d = report.to_dict()
    assert "version" in d
    assert "by_class" in d
    assert d["version"] == INVARIANCE_VERSION
    by_class = d["by_class"]
    assert isinstance(by_class, dict)
    for pc in PerturbationClass:
        assert pc.value in by_class


def test_build_invariance_report_is_frozen():
    report = build_invariance_report([])
    with pytest.raises((AttributeError, TypeError)):
        report.version = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# MetricSummary.invariance_report field
# ---------------------------------------------------------------------------


def test_metric_summary_invariance_report_defaults_to_none():
    summary = MetricSummary(
        post_shift_probe_accuracy=0.5,
        binary_parse_valid_rate=1.0,
        narrative_schema_valid_rate=1.0,
        narrative_parse_failure_count=0,
    )
    assert summary.invariance_report is None


def test_metric_summary_invariance_report_can_be_set():
    inv_report = build_invariance_report([])
    summary = MetricSummary(
        post_shift_probe_accuracy=0.75,
        binary_parse_valid_rate=1.0,
        narrative_schema_valid_rate=1.0,
        narrative_parse_failure_count=0,
        invariance_report=inv_report,
    )
    assert summary.invariance_report is inv_report


def test_metric_summary_invariance_report_does_not_affect_leaderboard_metric():
    """Adding an invariance_report must not change post_shift_probe_accuracy."""
    inv_report = build_invariance_report([])
    summary_without = MetricSummary(
        post_shift_probe_accuracy=0.5,
        binary_parse_valid_rate=1.0,
        narrative_schema_valid_rate=1.0,
        narrative_parse_failure_count=0,
    )
    summary_with = MetricSummary(
        post_shift_probe_accuracy=0.5,
        binary_parse_valid_rate=1.0,
        narrative_schema_valid_rate=1.0,
        narrative_parse_failure_count=0,
        invariance_report=inv_report,
    )
    assert summary_without.post_shift_probe_accuracy == summary_with.post_shift_probe_accuracy


# ---------------------------------------------------------------------------
# InvarianceReport: separate from aggregate accuracy
# ---------------------------------------------------------------------------


def test_invariance_report_accuracy_is_separate_from_aggregate():
    """InvarianceReport accuracy is computed independently per class."""
    cases_preds = [
        # All correct for wording_paraphrase
        (_make_case("ep-1", PerturbationClass.WORDING_PARAPHRASE, _FOUR_LABELS),
         _valid_prediction(*_FOUR_LABELS)),
        # All wrong for layout_reformat
        (_make_case("ep-1", PerturbationClass.LAYOUT_REFORMAT, _FOUR_LABELS),
         _valid_prediction(REPEL, ATTRACT, REPEL, ATTRACT)),
    ]
    report = build_invariance_report(cases_preds)
    acc_dict = dict(report.by_class)
    assert acc_dict["wording_paraphrase"].accuracy == 1.0
    assert acc_dict["layout_reformat"].accuracy == 0.0
    # Other classes have no data → zero
    assert acc_dict["neutral_renaming"].accuracy == 0.0
    assert acc_dict["non_causal_ordering"].accuracy == 0.0
