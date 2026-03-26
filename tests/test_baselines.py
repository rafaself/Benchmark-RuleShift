from functools import partial

from tasks.ruleshift_benchmark.baselines import (
    physics_prior_baseline,
    last_evidence_baseline,
    never_update_baseline,
    random_baseline,
    run_baseline,
    run_baselines,
    template_position_baseline,
)
from core.parser import ParseStatus
from tasks.ruleshift_benchmark.protocol import (
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
from tasks.ruleshift_benchmark.rules import label
from tasks.ruleshift_benchmark.schema import (
    DIFFICULTY_VERSION,
    Episode,
    EpisodeItem,
    ProbeMetadata,
    derive_difficulty_factors,
    derive_difficulty_profile,
)


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
    difficulty_factors = derive_difficulty_factors(item_rows, template.pre_count)
    difficulty, difficulty_profile_id = derive_difficulty_profile(difficulty_factors)

    return Episode(
        episode_id=episode_id,
        split=Split.DEV,
        difficulty=difficulty,
        template_id=template_id,
        template_family=(
            "canonical" if template_id is TemplateId.T1 else "observation_log"
        ),
        rule_A=rule_a,
        rule_B=rule_b,
        transition=Transition.from_rules(rule_a, rule_b),
        pre_count=template.pre_count,
        post_labeled_count=template.post_labeled_count,
        shift_after_position=template.shift_after_position,
        contradiction_count_post=contradiction_count_post,
        difficulty_profile_id=difficulty_profile_id,
        difficulty_factors=difficulty_factors,
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


def _t1_episode() -> Episode:
    return _build_episode(
        episode_id="fixture-t1",
        template_id=TemplateId.T1,
        rule_a=RuleName.R_STD,
        labeled_pairs=((1, 1), (-1, 1), (-2, -3), (2, -2), (-1, -2)),
        probe_pairs=((2, 3), (-3, -2), (3, -1), (-2, 3)),
    )


def _t2_episode() -> Episode:
    return _build_episode(
        episode_id="fixture-t2",
        template_id=TemplateId.T2,
        rule_a=RuleName.R_INV,
        labeled_pairs=((1, 2), (-1, 2), (-2, -3), (2, 3), (-2, 1)),
        probe_pairs=((1, 3), (-1, -3), (2, -1), (-3, 2)),
    )


def test_each_baseline_returns_exactly_four_ordered_labels():
    episode = _t1_episode()
    baselines = (
        partial(random_baseline, seed=11),
        never_update_baseline,
        last_evidence_baseline,
        physics_prior_baseline,
        template_position_baseline,
    )

    for baseline in baselines:
        prediction = baseline(episode)
        assert len(prediction) == 4
        assert all(isinstance(label_value, InteractionLabel) for label_value in prediction)


def test_random_baseline_is_deterministic_for_a_given_seed():
    episode = _t1_episode()

    assert random_baseline(episode, seed=7) == random_baseline(episode, seed=7)
    assert len({random_baseline(episode, seed=seed) for seed in range(8)}) > 1


def test_never_update_baseline_uses_only_pre_shift_evidence():
    episode = _t1_episode()

    assert never_update_baseline(episode) == (
        InteractionLabel.REPEL,
        InteractionLabel.REPEL,
        InteractionLabel.ATTRACT,
        InteractionLabel.ATTRACT,
    )


def test_last_evidence_baseline_depends_only_on_final_labeled_example():
    episode = _t1_episode()

    assert last_evidence_baseline(episode) == (
        InteractionLabel.ATTRACT,
        InteractionLabel.ATTRACT,
        InteractionLabel.REPEL,
        InteractionLabel.REPEL,
    )


def test_physics_prior_baseline_always_matches_r_std_behavior():
    episode = _t1_episode()

    assert physics_prior_baseline(episode) == tuple(
        label(RuleName.R_STD, item.q1, item.q2)
        for item in episode.items[LABELED_ITEM_COUNT:]
    )


def test_template_position_baseline_is_deterministic():
    episode = _t2_episode()

    assert template_position_baseline(episode) == template_position_baseline(episode)
    assert template_position_baseline(episode) == (
        InteractionLabel.REPEL,
        InteractionLabel.ATTRACT,
        InteractionLabel.REPEL,
        InteractionLabel.ATTRACT,
    )


def test_template_position_baseline_does_not_require_hidden_answer_metadata():
    episode = _t2_episode()
    expected = template_position_baseline(episode)

    object.__setattr__(
        episode,
        "probe_targets",
        (
            InteractionLabel.REPEL,
            InteractionLabel.REPEL,
            InteractionLabel.REPEL,
            InteractionLabel.REPEL,
        ),
    )
    object.__setattr__(episode, "probe_metadata", ())

    assert template_position_baseline(episode) == expected


def test_runner_returns_stable_structured_outputs():
    episodes = (_t1_episode(), _t2_episode())
    results = run_baselines(
        episodes,
        (
            ("never_update", never_update_baseline),
            ("template_position", template_position_baseline),
        ),
    )

    assert tuple(result.baseline_name for result in results) == (
        "never_update",
        "template_position",
    )
    assert tuple(row.episode_id for row in results[0].rows) == ("fixture-t1", "fixture-t2")
    assert all(
        row.parsed_prediction.status is ParseStatus.VALID
        for result in results
        for row in result.rows
    )
    assert tuple(row.target for row in results[0].rows) == tuple(
        episode.probe_targets for episode in episodes
    )


def test_run_baseline_preserves_single_baseline_row_order():
    episodes = (_t2_episode(), _t1_episode())
    result = run_baseline("physics_prior", physics_prior_baseline, episodes)

    assert result.baseline_name == "physics_prior"
    assert tuple(row.episode_id for row in result.rows) == ("fixture-t2", "fixture-t1")


def test_hand_built_fixture_shows_baselines_can_disagree():
    episode = _t1_episode()

    assert never_update_baseline(episode) == (
        InteractionLabel.REPEL,
        InteractionLabel.REPEL,
        InteractionLabel.ATTRACT,
        InteractionLabel.ATTRACT,
    )
    assert last_evidence_baseline(episode) == (
        InteractionLabel.ATTRACT,
        InteractionLabel.ATTRACT,
        InteractionLabel.REPEL,
        InteractionLabel.REPEL,
    )
    assert physics_prior_baseline(episode) == (
        InteractionLabel.REPEL,
        InteractionLabel.REPEL,
        InteractionLabel.ATTRACT,
        InteractionLabel.ATTRACT,
    )
    assert template_position_baseline(episode) == (
        InteractionLabel.ATTRACT,
        InteractionLabel.REPEL,
        InteractionLabel.ATTRACT,
        InteractionLabel.ATTRACT,
    )
