"""Invariance checking for the RuleShift benchmark.

Non-causal perturbations applied to rendered prompts must not change the
correct Binary answer.  Each ``InvarianceCase`` is a (episode, perturbation)
pair: it stores both the canonical and perturbed prompt together with the
original ``probe_targets`` copied verbatim from the episode.

Perturbation classes
--------------------
- ``wording_paraphrase``    : synonym-level wording changes in intro/outro text.
- ``layout_reformat``       : visual separator added before the probe section.
- ``neutral_renaming``      : section headings renamed to equivalent labels.
- ``non_causal_ordering``   : pre-shift labeled items presented in reverse order.

Invariant property: ``case.probe_targets == episode.probe_targets`` for every
generated case, regardless of perturbation class.

Versioning: every case carries ``perturbation_version = INVARIANCE_VERSION``
so stored fixtures remain auditable across code changes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Callable, Iterable, Sequence

from core.parser import ParseStatus, ParsedPrediction
from tasks.ruleshift_benchmark.protocol import PROBE_COUNT, InteractionLabel
from tasks.ruleshift_benchmark.render import render_binary_prompt
from tasks.ruleshift_benchmark.schema import Episode

__all__ = [
    "INVARIANCE_VERSION",
    "PERTURBATION_CLASS_ORDER",
    "PerturbationClass",
    "InvarianceCase",
    "PerturbationClassAccuracy",
    "InvarianceReport",
    "apply_wording_paraphrase",
    "apply_layout_reformat",
    "apply_neutral_renaming",
    "apply_non_causal_ordering",
    "generate_invariance_cases",
    "build_invariance_report",
]

# Bump when perturbation logic changes so stored cases are distinguishable.
INVARIANCE_VERSION = "v1"

# Canonical reporting order — matches PerturbationClass enum definition order.
PERTURBATION_CLASS_ORDER: tuple[str, ...] = (
    "wording_paraphrase",
    "layout_reformat",
    "neutral_renaming",
    "non_causal_ordering",
)

# ---------------------------------------------------------------------------
# Substitution tables for deterministic text perturbations
# ---------------------------------------------------------------------------

# (old, new) pairs applied left-to-right via str.replace.  Each target phrase
# appears at most once in any rendered binary prompt, so plain replace is safe.

_WORDING_PARAPHRASE_SUBS: tuple[tuple[str, str], ...] = (
    # Binary outro — shared across all binary prompts (both template families).
    (
        "Return exactly 4 labels in order, one per probe.",
        "Output exactly 4 labels in order, one per probe.",
    ),
    # Canonical binary: second sentence of the intro paragraph.
    (
        "Each labeled line shows q1, q2, and the observed result.",
        "Each entry shows q1, q2, and the observed result.",
    ),
    # Observation-log binary: second sentence of the intro paragraph.
    (
        "Each entry records q1, q2, and the observed outcome.",
        "Each entry lists q1, q2, and the observed outcome.",
    ),
)

_LAYOUT_REFORMAT_SUBS: tuple[tuple[str, str], ...] = (
    # Add a visual separator line before the canonical probe heading.
    ("\n\nProbes:\n", "\n\n---\nProbes:\n"),
    # Add a visual separator line before the observation-log probe heading.
    ("\n\nUnresolved probe entries:\n", "\n\n---\nUnresolved probe entries:\n"),
)

_NEUTRAL_RENAMING_SUBS: tuple[tuple[str, str], ...] = (
    # Canonical section headings.
    ("Labeled examples:", "Training examples:"),
    ("Probes:", "Test cases:"),
    # Observation-log section headings.
    ("Resolved log entries:", "Confirmed log entries:"),
    ("Unresolved probe entries:", "Pending entries:"),
)


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------


class PerturbationClass(StrEnum):
    """Four canonical non-causal perturbation classes."""

    WORDING_PARAPHRASE = "wording_paraphrase"
    LAYOUT_REFORMAT = "layout_reformat"
    NEUTRAL_RENAMING = "neutral_renaming"
    NON_CAUSAL_ORDERING = "non_causal_ordering"


@dataclass(frozen=True, slots=True)
class InvarianceCase:
    """A minimal-pair perturbation of a single episode.

    ``probe_targets`` is always identical to the originating episode's
    ``probe_targets``; only the rendered prompt surface form changes.
    Both prompts are stored for full auditability.
    """

    episode_id: str
    perturbation_class: PerturbationClass
    canonical_prompt: str
    perturbed_prompt: str
    probe_targets: tuple[InteractionLabel, ...]
    perturbation_version: str  # always INVARIANCE_VERSION


@dataclass(frozen=True, slots=True)
class PerturbationClassAccuracy:
    """Aggregate binary accuracy for one perturbation class over all episodes."""

    perturbation_class: str  # PerturbationClass.value
    episode_count: int
    correct_probes: int
    total_probes: int

    @property
    def accuracy(self) -> float:
        if self.total_probes == 0:
            return 0.0
        return self.correct_probes / self.total_probes

    def to_dict(self) -> dict[str, object]:
        return {
            "perturbation_class": self.perturbation_class,
            "episode_count": self.episode_count,
            "correct_probes": self.correct_probes,
            "total_probes": self.total_probes,
            "accuracy": self.accuracy,
        }


@dataclass(frozen=True, slots=True)
class InvarianceReport:
    """Per-perturbation-class binary accuracy across all invariance cases.

    Binary is the only leaderboard metric; this report is diagnostic only
    and does not affect ``primary_result`` in the canonical Kaggle payload.
    Results are ordered by ``PERTURBATION_CLASS_ORDER``.
    """

    by_class: tuple[tuple[str, PerturbationClassAccuracy], ...]
    version: str  # INVARIANCE_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "by_class": {k: v.to_dict() for k, v in self.by_class},
        }


# ---------------------------------------------------------------------------
# Perturbation functions — pure, deterministic, prompt-text only
# ---------------------------------------------------------------------------


def apply_wording_paraphrase(prompt: str) -> str:
    """Replace selected phrases with semantically equivalent synonyms.

    Targets intro/outro sentences only; ``attract`` / ``repel`` labels and
    all charge values are preserved exactly.
    """
    for old, new in _WORDING_PARAPHRASE_SUBS:
        prompt = prompt.replace(old, new)
    return prompt


def apply_layout_reformat(prompt: str) -> str:
    """Insert a visual ``---`` separator before the probe section heading.

    Works for both the canonical (``Probes:``) and observation-log
    (``Unresolved probe entries:``) template families.
    """
    for old, new in _LAYOUT_REFORMAT_SUBS:
        prompt = prompt.replace(old, new)
    return prompt


def apply_neutral_renaming(prompt: str) -> str:
    """Rename section headings to equivalent labels.

    The renamed headings carry no semantic signal about rule logic, charge
    values, or interaction labels.
    """
    for old, new in _NEUTRAL_RENAMING_SUBS:
        prompt = prompt.replace(old, new)
    return prompt


def apply_non_causal_ordering(prompt: str, *, pre_shift_count: int) -> str:
    """Reverse the presentation order of pre-shift labeled items.

    The first ``pre_shift_count`` labeled items all demonstrate ``rule_A``.
    Their relative order is non-causal: the same rule is derivable from any
    permutation.  ``probe_targets`` depend solely on ``rule_B`` applied to
    fixed charge pairs, so they are unaffected.

    Detects prompt format automatically (canonical vs. observation-log).
    """
    if re.search(r"^\[\d{2}\] ", prompt, re.MULTILINE):
        return _reverse_preshift_obs_log(prompt, pre_shift_count)
    return _reverse_preshift_canonical(prompt, pre_shift_count)


# ---------------------------------------------------------------------------
# Non-causal ordering helpers
# ---------------------------------------------------------------------------


def _reverse_preshift_canonical(prompt: str, pre_shift_count: int) -> str:
    """Reverse the body content of lines ``1..pre_shift_count`` in canonical format.

    Line format: ``{pos}. {body}`` where body = ``q1=..., q2=... -> {label}``.
    Position numbers are kept stable; only the body (charges + label) swaps.
    """
    lines = prompt.split("\n")
    item_map: dict[int, tuple[int, str]] = {}  # pos → (line_index, body)

    for idx, line in enumerate(lines):
        m = re.match(r"^(\d+)\. (.+)$", line)
        if m:
            pos = int(m.group(1))
            if 1 <= pos <= pre_shift_count:
                item_map[pos] = (idx, m.group(2))

    if len(item_map) < 2:
        return prompt  # nothing to reverse (should not happen with current templates)

    positions = sorted(item_map)
    bodies = [item_map[p][1] for p in positions]
    for pos, body in zip(positions, reversed(bodies)):
        line_idx = item_map[pos][0]
        lines[line_idx] = f"{pos}. {body}"

    return "\n".join(lines)


def _reverse_preshift_obs_log(prompt: str, pre_shift_count: int) -> str:
    """Reverse the body content of log lines ``[01]..[N]`` in observation-log format.

    Line format: ``[{pos:02d}] {body}`` where body = ``q1=... | q2=... | observed=...``.
    Zero-padded position tags are preserved; only the body swaps.
    """
    lines = prompt.split("\n")
    item_map: dict[int, tuple[int, str]] = {}  # pos → (line_index, body)

    for idx, line in enumerate(lines):
        m = re.match(r"^\[(\d+)\] (.+)$", line)
        if m:
            pos = int(m.group(1))
            if 1 <= pos <= pre_shift_count:
                item_map[pos] = (idx, m.group(2))

    if len(item_map) < 2:
        return prompt

    positions = sorted(item_map)
    bodies = [item_map[p][1] for p in positions]
    for pos, body in zip(positions, reversed(bodies)):
        line_idx = item_map[pos][0]
        lines[line_idx] = f"[{pos:02d}] {body}"

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Case generation
# ---------------------------------------------------------------------------


def generate_invariance_cases(
    episodes: Iterable[Episode],
    *,
    render_fn: Callable[[Episode], str] = render_binary_prompt,
) -> list[InvarianceCase]:
    """Generate one ``InvarianceCase`` per (episode, perturbation_class) pair.

    All four perturbation classes are applied to every episode, yielding
    ``4 × len(episodes)`` cases in ``PERTURBATION_CLASS_ORDER``.  Each case
    stores both the canonical and perturbed prompt for auditability, and
    carries ``probe_targets`` copied verbatim from the episode.

    Args:
        episodes: Episodes to generate cases from.
        render_fn: Callable that renders an Episode to a prompt string.
            Defaults to ``render_binary_prompt``.

    Returns:
        List of ``InvarianceCase`` objects in episode-major, class-minor order.
    """
    cases: list[InvarianceCase] = []
    for episode in episodes:
        canonical = render_fn(episode)
        for perturb_class in PerturbationClass:
            perturbed = _apply_perturbation(canonical, perturb_class, episode)
            cases.append(
                InvarianceCase(
                    episode_id=episode.episode_id,
                    perturbation_class=perturb_class,
                    canonical_prompt=canonical,
                    perturbed_prompt=perturbed,
                    probe_targets=episode.probe_targets,
                    perturbation_version=INVARIANCE_VERSION,
                )
            )
    return cases


def _apply_perturbation(
    prompt: str,
    perturb_class: PerturbationClass,
    episode: Episode,
) -> str:
    if perturb_class is PerturbationClass.WORDING_PARAPHRASE:
        return apply_wording_paraphrase(prompt)
    if perturb_class is PerturbationClass.LAYOUT_REFORMAT:
        return apply_layout_reformat(prompt)
    if perturb_class is PerturbationClass.NEUTRAL_RENAMING:
        return apply_neutral_renaming(prompt)
    if perturb_class is PerturbationClass.NON_CAUSAL_ORDERING:
        return apply_non_causal_ordering(
            prompt, pre_shift_count=episode.shift_after_position
        )
    raise ValueError(f"Unhandled perturbation class: {perturb_class!r}")  # pragma: no cover


# ---------------------------------------------------------------------------
# Report aggregation
# ---------------------------------------------------------------------------


def build_invariance_report(
    cases_and_predictions: Sequence[tuple[InvarianceCase, ParsedPrediction]],
) -> InvarianceReport:
    """Aggregate per-case predictions into a per-class accuracy report.

    Args:
        cases_and_predictions: Sequence of (InvarianceCase, ParsedPrediction)
            pairs.  Each prediction must correspond to running the LLM on
            ``case.perturbed_prompt`` in Binary mode.

    Returns:
        An ``InvarianceReport`` with one ``PerturbationClassAccuracy`` per
        class, in ``PERTURBATION_CLASS_ORDER``.
    """
    accum: dict[str, dict[str, int]] = {
        pc.value: {"episode_count": 0, "correct_probes": 0, "total_probes": 0}
        for pc in PerturbationClass
    }

    for case, prediction in cases_and_predictions:
        key = case.perturbation_class.value
        correct = _count_correct_probes(prediction, case.probe_targets)
        accum[key]["episode_count"] += 1
        accum[key]["correct_probes"] += correct
        accum[key]["total_probes"] += PROBE_COUNT

    by_class = tuple(
        (
            pc.value,
            PerturbationClassAccuracy(
                perturbation_class=pc.value,
                episode_count=accum[pc.value]["episode_count"],
                correct_probes=accum[pc.value]["correct_probes"],
                total_probes=accum[pc.value]["total_probes"],
            ),
        )
        for pc in PerturbationClass
    )
    return InvarianceReport(by_class=by_class, version=INVARIANCE_VERSION)


def _count_correct_probes(
    prediction: ParsedPrediction,
    targets: tuple[InteractionLabel, ...],
) -> int:
    if prediction.status is not ParseStatus.VALID:
        return 0
    if len(prediction.labels) != PROBE_COUNT:
        return 0
    return sum(p is t for p, t in zip(prediction.labels, targets))
