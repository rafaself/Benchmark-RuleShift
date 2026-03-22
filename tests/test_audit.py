from audit import (
    AuditSliceSummary,
    AuditSource,
    HeuristicAlignmentSummary,
    ModeComparisonSummary,
    run_audit,
)
from baselines import (
    last_evidence_baseline,
    never_update_baseline,
    physics_prior_baseline,
    random_baseline,
    run_baselines,
    template_position_baseline,
)
from parser import ParsedPrediction, ParseStatus
from protocol import (
    LABELED_ITEM_COUNT,
    Difficulty,
    InteractionLabel,
    ItemKind,
    Phase,
    RuleName,
    Split,
    TEMPLATES,
    TemplateId,
    Transition,
)
from rules import label
from schema import DIFFICULTY_VERSION, Episode, EpisodeItem, ProbeMetadata


def _probe_sign_pattern(q1: int, q2: int) -> str:
    if q1 > 0 and q2 > 0:
        return "++"
    if q1 < 0 and q2 < 0:
        return "--"
    if q1 > 0 and q2 < 0:
        return "+-"
    return "-+"


def _effective_probe_targets(
    *,
    probe_items: tuple[EpisodeItem, ...],
    labeled_items: tuple[EpisodeItem, ...],
    pre_count: int,
    rule_a: RuleName,
    rule_b: RuleName,
) -> tuple[InteractionLabel, ...]:
    updated_sign_patterns = frozenset(
        _probe_sign_pattern(item.q1, item.q2) for item in labeled_items[pre_count:]
    )
    return tuple(
        label(
            rule_b if _probe_sign_pattern(item.q1, item.q2) in updated_sign_patterns else rule_a,
            item.q1,
            item.q2,
        )
        for item in probe_items
    )


def _difficulty_for(
    template_id: TemplateId,
    probe_targets: tuple[InteractionLabel, ...],
) -> Difficulty:
    if template_id is TemplateId.T1 and len(set(probe_targets)) >= 2:
        return Difficulty.EASY
    return Difficulty.MEDIUM


def _build_episode(
    *,
    episode_id: str,
    template_id: TemplateId,
    rule_a: RuleName,
    labeled_pairs: tuple[tuple[int, int], ...],
    probe_pairs: tuple[tuple[int, int], ...],
) -> Episode:
    template = TEMPLATES[template_id]
    rule_b = rule_a.opposite

    items = []
    for position, (q1, q2) in enumerate(labeled_pairs + probe_pairs, start=1):
        is_probe = position > LABELED_ITEM_COUNT
        active_rule = rule_a if position <= template.pre_count else rule_b
        items.append(
            EpisodeItem(
                position=position,
                phase=Phase.POST if position > template.pre_count else Phase.PRE,
                kind=ItemKind.PROBE if is_probe else ItemKind.LABELED,
                q1=q1,
                q2=q2,
                label=None if is_probe else label(active_rule, q1, q2),
            )
        )

    item_rows = tuple(items)
    labeled_items = item_rows[:LABELED_ITEM_COUNT]
    probe_items = item_rows[LABELED_ITEM_COUNT:]
    probe_targets = _effective_probe_targets(
        probe_items=probe_items,
        labeled_items=labeled_items,
        pre_count=template.pre_count,
        rule_a=rule_a,
        rule_b=rule_b,
    )
    probe_metadata = tuple(
        ProbeMetadata(
            position=item.position,
            is_disagreement_probe=label(RuleName.R_STD, item.q1, item.q2)
            != label(RuleName.R_INV, item.q1, item.q2),
            old_rule_label=label(rule_a, item.q1, item.q2),
            new_rule_label=label(rule_b, item.q1, item.q2),
        )
        for item in probe_items
    )
    contradiction_count_post = sum(
        label(rule_a, item.q1, item.q2) != label(rule_b, item.q1, item.q2)
        for item in item_rows[template.pre_count:LABELED_ITEM_COUNT]
    )

    return Episode(
        episode_id=episode_id,
        split=Split.DEV,
        difficulty=_difficulty_for(template_id, probe_targets),
        template_id=template_id,
        rule_A=rule_a,
        rule_B=rule_b,
        transition=Transition.from_rules(rule_a, rule_b),
        pre_count=template.pre_count,
        post_labeled_count=template.post_labeled_count,
        shift_after_position=template.shift_after_position,
        contradiction_count_post=contradiction_count_post,
        items=item_rows,
        probe_targets=probe_targets,
        probe_label_counts=(
            (
                InteractionLabel.ATTRACT,
                probe_targets.count(InteractionLabel.ATTRACT),
            ),
            (
                InteractionLabel.REPEL,
                probe_targets.count(InteractionLabel.REPEL),
            ),
        ),
        probe_sign_pattern_counts=(
            ("++", sum(_probe_sign_pattern(item.q1, item.q2) == "++" for item in probe_items)),
            ("--", sum(_probe_sign_pattern(item.q1, item.q2) == "--" for item in probe_items)),
            ("+-", sum(_probe_sign_pattern(item.q1, item.q2) == "+-" for item in probe_items)),
            ("-+", sum(_probe_sign_pattern(item.q1, item.q2) == "-+" for item in probe_items)),
        ),
        probe_metadata=probe_metadata,
        difficulty_version=DIFFICULTY_VERSION,
    )


def _episodes() -> tuple[Episode, ...]:
    return (
        _build_episode(
            episode_id="fixture-t1",
            template_id=TemplateId.T1,
            rule_a=RuleName.R_STD,
            labeled_pairs=((1, 1), (-1, 1), (-2, -3), (2, -2), (-1, -2)),
            probe_pairs=((2, 3), (-3, -2), (3, -1), (-2, 3)),
        ),
        _build_episode(
            episode_id="fixture-t2",
            template_id=TemplateId.T2,
            rule_a=RuleName.R_INV,
            labeled_pairs=((1, 2), (-1, 2), (-2, -3), (2, 3), (-2, 1)),
            probe_pairs=((1, 3), (-1, -3), (2, -1), (-3, 2)),
        ),
    )


def _binary_source(episodes: tuple[Episode, ...]) -> AuditSource:
    return AuditSource.from_parsed_predictions(
        "binary_model",
        (
            ParsedPrediction(
                labels=template_position_baseline(episodes[0]),
                status=ParseStatus.VALID,
            ),
            ParsedPrediction(
                labels=episodes[1].probe_targets,
                status=ParseStatus.VALID,
            ),
        ),
        task_mode="Binary",
    )


def _narrative_source(episodes: tuple[Episode, ...]) -> AuditSource:
    return AuditSource.from_parsed_predictions(
        "narrative_model",
        (
            ParsedPrediction(
                labels=episodes[0].probe_targets,
                status=ParseStatus.VALID,
            ),
            ParsedPrediction(labels=(), status=ParseStatus.INVALID),
        ),
        task_mode="Narrative",
    )


def _baseline_sources(episodes: tuple[Episode, ...]) -> tuple[AuditSource, ...]:
    results = run_baselines(
        episodes,
        (
            ("random", lambda episode: random_baseline(episode, seed=11)),
            ("never_update", never_update_baseline),
            ("last_evidence", last_evidence_baseline),
            ("physics_prior", physics_prior_baseline),
            ("template_position", template_position_baseline),
        ),
    )
    return tuple(AuditSource.from_baseline_run(result) for result in results)


def _report():
    episodes = _episodes()
    sources = (_binary_source(episodes), _narrative_source(episodes), *_baseline_sources(episodes))
    return run_audit(episodes, sources)


def _source_map(report):
    return {summary.name: summary for summary in report.source_summaries}


def _slice_map(summaries):
    return dict(summaries)


def _failure_pattern_map(summary):
    return {pattern.pattern_name: pattern for pattern in summary.failure_patterns}


def test_audit_results_are_deterministic_for_fixed_inputs():
    assert _report() == _report()


def test_audit_correctly_aggregates_overall_accuracy_from_fixture_predictions():
    report = _report()
    source_summaries = _source_map(report)

    assert source_summaries["binary_model"].overall == AuditSliceSummary(
        episode_count=2,
        correct_probe_count=5,
        total_probe_count=8,
        accuracy=0.625,
        valid_prediction_count=2,
        parse_valid_rate=1.0,
    )
    assert source_summaries["narrative_model"].overall == AuditSliceSummary(
        episode_count=2,
        correct_probe_count=4,
        total_probe_count=8,
        accuracy=0.5,
        valid_prediction_count=1,
        parse_valid_rate=0.5,
    )


def test_template_level_and_difficulty_level_summaries_match_hand_checked_fixture():
    report = _report()
    binary_summary = _source_map(report)["binary_model"]

    assert _slice_map(binary_summary.by_template) == {
        "T1": AuditSliceSummary(
            episode_count=1,
            correct_probe_count=1,
            total_probe_count=4,
            accuracy=0.25,
            valid_prediction_count=1,
            parse_valid_rate=1.0,
        ),
        "T2": AuditSliceSummary(
            episode_count=1,
            correct_probe_count=4,
            total_probe_count=4,
            accuracy=1.0,
            valid_prediction_count=1,
            parse_valid_rate=1.0,
        ),
    }
    assert _slice_map(binary_summary.by_difficulty) == {
        "easy": AuditSliceSummary(
            episode_count=1,
            correct_probe_count=1,
            total_probe_count=4,
            accuracy=0.25,
            valid_prediction_count=1,
            parse_valid_rate=1.0,
        ),
        "medium": AuditSliceSummary(
            episode_count=1,
            correct_probe_count=4,
            total_probe_count=4,
            accuracy=1.0,
            valid_prediction_count=1,
            parse_valid_rate=1.0,
        ),
    }


def test_baseline_comparison_summary_is_stable():
    report = _report()

    assert report.baseline_comparison.accuracy_ranking == (
        ("template_position", 0.625),
        ("last_evidence", 0.5),
        ("never_update", 0.5),
        ("physics_prior", 0.5),
        ("random", 0.25),
    )
    assert report.baseline_comparison.best_baseline_name == "template_position"
    assert report.baseline_comparison.best_baseline_accuracy == 0.625


def test_audit_handles_current_absence_of_emitted_hard_episodes_cleanly():
    report = _report()

    assert report.difficulty_labels_present == ("easy", "medium")
    assert report.difficulty_labels_missing == ("hard",)
    assert report.limitations == (
        "No emitted hard episodes in supplied set; hard slice omitted.",
    )
    for summary in report.source_summaries:
        assert "hard" not in _slice_map(summary.by_difficulty)


def test_task_mode_summaries_and_comparison_are_reported_deterministically():
    report = _report()
    task_mode_summaries = _slice_map(report.task_mode_summaries)

    assert task_mode_summaries == {
        "Binary": AuditSliceSummary(
            episode_count=2,
            correct_probe_count=5,
            total_probe_count=8,
            accuracy=0.625,
            valid_prediction_count=2,
            parse_valid_rate=1.0,
        ),
        "Narrative": AuditSliceSummary(
            episode_count=2,
            correct_probe_count=4,
            total_probe_count=8,
            accuracy=0.5,
            valid_prediction_count=1,
            parse_valid_rate=0.5,
        ),
    }
    assert report.mode_comparison == ModeComparisonSummary(
        binary_accuracy=0.625,
        narrative_accuracy=0.5,
        accuracy_gap=0.125,
        binary_parse_valid_rate=1.0,
        narrative_parse_valid_rate=0.5,
        parse_valid_rate_gap=0.5,
    )


def test_failure_pattern_summaries_are_computed_from_observable_prediction_agreement():
    report = _report()
    source_summaries = _source_map(report)
    binary_patterns = _failure_pattern_map(source_summaries["binary_model"])
    narrative_patterns = _failure_pattern_map(source_summaries["narrative_model"])

    assert binary_patterns["template-position"] == HeuristicAlignmentSummary(
        pattern_name="template-position",
        reference_source_name="template_position",
        matching_probe_count=8,
        total_probe_count=8,
        probe_agreement_rate=1.0,
        matching_error_probe_count=3,
        total_error_probes=3,
        error_agreement_rate=1.0,
        matching_episode_count=2,
        episode_count=2,
        episode_agreement_rate=1.0,
    )
    assert binary_patterns["recency / last-evidence-like"] == HeuristicAlignmentSummary(
        pattern_name="recency / last-evidence-like",
        reference_source_name="last_evidence",
        matching_probe_count=3,
        total_probe_count=8,
        probe_agreement_rate=0.375,
        matching_error_probe_count=1,
        total_error_probes=3,
        error_agreement_rate=1 / 3,
        matching_episode_count=0,
        episode_count=2,
        episode_agreement_rate=0.0,
    )
    assert narrative_patterns["persistence-like"] == HeuristicAlignmentSummary(
        pattern_name="persistence-like",
        reference_source_name="never_update",
        matching_probe_count=2,
        total_probe_count=8,
        probe_agreement_rate=0.25,
        matching_error_probe_count=0,
        total_error_probes=4,
        error_agreement_rate=0.0,
        matching_episode_count=0,
        episode_count=2,
        episode_agreement_rate=0.0,
    )

    last_evidence_patterns = tuple(
        pattern.pattern_name for pattern in source_summaries["last_evidence"].failure_patterns
    )
    assert "recency / last-evidence-like" not in last_evidence_patterns
