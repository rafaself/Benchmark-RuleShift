from tasks.ruleshift_benchmark.generator import generate_episode
from tasks.ruleshift_benchmark.protocol import LABELED_ITEM_COUNT, RuleName, TemplateFamily
from tasks.ruleshift_benchmark.render import render_binary_prompt, render_narrative_prompt
from tasks.ruleshift_benchmark.rules import label


def _format_charge(charge: int) -> str:
    return f"{charge:+d}"


def _binary_line(item) -> str:
    outcome = item.label.value if item.label is not None else "?"
    return f"{item.position}. q1={_format_charge(item.q1)}, q2={_format_charge(item.q2)} -> {outcome}"


def _binary_log_line(item) -> str:
    outcome = item.label.value if item.label is not None else "?"
    return (
        f"[{item.position:02d}] q1={_format_charge(item.q1)} | "
        f"q2={_format_charge(item.q2)} | observed={outcome}"
    )


def _narrative_line(item) -> str:
    outcome = item.label.value if item.label is not None else "?"
    return (
        f"{item.position}. A {_format_charge(item.q1)} charge and a {_format_charge(item.q2)} "
        f"charge were observed to {outcome}."
    )


def _narrative_log_line(item) -> str:
    outcome = item.label.value if item.label is not None else "?"
    return (
        f"[{item.position:02d}] charges({_format_charge(item.q1)}, {_format_charge(item.q2)}) "
        f"=> observed {outcome}."
    )


def _rendered_binary_line(episode, item) -> str:
    if episode.template_family is TemplateFamily.OBSERVATION_LOG:
        return _binary_log_line(item)
    return _binary_line(item)


def _rendered_narrative_line(episode, item) -> str:
    if episode.template_family is TemplateFamily.OBSERVATION_LOG:
        return _narrative_log_line(item)
    return _narrative_line(item)


def _assert_lines_appear_in_order(text: str, lines: tuple[str, ...]) -> None:
    search_start = 0
    for line in lines:
        next_index = text.index(line, search_start)
        assert next_index >= search_start
        search_start = next_index + len(line)


def _first_episode_with_template(template_id: str):
    for seed in range(64):
        episode = generate_episode(seed)
        if episode.template_id.value == template_id:
            return episode
    raise AssertionError(f"could not find an episode with template_id={template_id!r}")


def make_valid_t1_episode():
    return _first_episode_with_template("T1")


def make_valid_t2_episode():
    return _first_episode_with_template("T2")


def test_same_episode_renders_identically_across_repeated_runs():
    episode = make_valid_t1_episode()

    assert render_binary_prompt(episode) == render_binary_prompt(episode)
    assert render_binary_prompt(episode) == render_binary_prompt(episode)
    assert render_narrative_prompt(episode) == render_narrative_prompt(episode)
    assert render_narrative_prompt(episode) == render_narrative_prompt(episode)


def test_binary_render_includes_all_labeled_examples_and_all_probes_in_correct_order():
    episode = make_valid_t1_episode()
    prompt = render_binary_prompt(episode)
    labeled_lines = tuple(
        _rendered_binary_line(episode, item) for item in episode.items[:LABELED_ITEM_COUNT]
    )
    probe_lines = tuple(
        _rendered_binary_line(episode, item)
        for item in episode.items[LABELED_ITEM_COUNT:]
    )

    expected_labeled_heading = (
        "Resolved log entries:"
        if episode.template_family is TemplateFamily.OBSERVATION_LOG
        else "Labeled examples:"
    )
    expected_probe_heading = (
        "Unresolved probe entries:"
        if episode.template_family is TemplateFamily.OBSERVATION_LOG
        else "Probes:"
    )

    assert expected_labeled_heading in prompt
    assert expected_probe_heading in prompt
    assert prompt.index(expected_labeled_heading) < prompt.index(expected_probe_heading)
    _assert_lines_appear_in_order(prompt, labeled_lines + probe_lines)


def test_narrative_render_includes_the_same_underlying_items_and_probe_order():
    episode = make_valid_t2_episode()
    binary_prompt = render_binary_prompt(episode)
    narrative_prompt = render_narrative_prompt(episode)
    binary_lines = tuple(_rendered_binary_line(episode, item) for item in episode.items)
    narrative_lines = tuple(_rendered_narrative_line(episode, item) for item in episode.items)

    _assert_lines_appear_in_order(binary_prompt, binary_lines)
    _assert_lines_appear_in_order(narrative_prompt, narrative_lines)

    for item in episode.items:
        binary_probe_fragment = f"{item.position}. q1={_format_charge(item.q1)}, q2={_format_charge(item.q2)}"
        narrative_probe_fragment = (
            f"{item.position}. A {_format_charge(item.q1)} charge and a {_format_charge(item.q2)} charge"
        )
        assert binary_probe_fragment in binary_prompt
        assert narrative_probe_fragment in narrative_prompt


def test_render_output_does_not_expose_hidden_metadata_or_target_labels():
    episode = make_valid_t1_episode()
    probe_target_sequence = ", ".join(target.value for target in episode.probe_targets)

    for prompt in (render_binary_prompt(episode), render_narrative_prompt(episode)):
        for forbidden_snippet in (
            "rule_A",
            "rule_B",
            "R_std",
            "R_inv",
            "shift_after_position",
            "probe_targets",
            "probe_metadata",
            "old_rule_label",
            "new_rule_label",
            "transition",
            "template_id",
            "pre_count",
            "post_labeled_count",
            "contradiction_count_post",
        ):
            assert forbidden_snippet not in prompt

        assert probe_target_sequence not in prompt

    binary_prompt = render_binary_prompt(episode)
    narrative_prompt = render_narrative_prompt(episode)
    for probe_index, item in enumerate(episode.items[LABELED_ITEM_COUNT:]):
        assert _rendered_binary_line(episode, item) in binary_prompt
        assert _rendered_narrative_line(episode, item) in narrative_prompt
        target_label = episode.probe_targets[probe_index].value
        if episode.template_family is TemplateFamily.OBSERVATION_LOG:
            assert (
                f"[{item.position:02d}] q1={_format_charge(item.q1)} | "
                f"q2={_format_charge(item.q2)} | observed={target_label}"
            ) not in binary_prompt
            assert (
                f"[{item.position:02d}] charges({_format_charge(item.q1)}, {_format_charge(item.q2)}) "
                f"=> observed {target_label}."
            ) not in narrative_prompt
        else:
            assert (
                f"{item.position}. q1={_format_charge(item.q1)}, q2={_format_charge(item.q2)} -> {target_label}"
            ) not in binary_prompt
            assert (
                f"{item.position}. A {_format_charge(item.q1)} charge and a {_format_charge(item.q2)} "
                f"charge were observed to {target_label}."
            ) not in narrative_prompt


def test_binary_and_narrative_preserve_identical_probe_targets_from_the_source_episode():
    episode = make_valid_t2_episode()

    binary_prompt = render_binary_prompt(episode)
    narrative_prompt = render_narrative_prompt(episode)
    for item in episode.items[LABELED_ITEM_COUNT:]:
        binary_fragment = _rendered_binary_line(episode, item)
        narrative_fragment = _rendered_narrative_line(episode, item)
        assert binary_fragment in binary_prompt
        assert narrative_fragment in narrative_prompt

    probe_items = episode.items[LABELED_ITEM_COUNT:]
    global_rule_a_targets = tuple(
        label(episode.rule_A, item.q1, item.q2) for item in probe_items
    )
    global_rule_b_targets = tuple(
        label(episode.rule_B, item.q1, item.q2) for item in probe_items
    )
    assert episode.probe_targets != global_rule_a_targets
    assert episode.probe_targets != global_rule_b_targets


def test_output_instructions_explicitly_require_four_ordered_answers():
    episode = make_valid_t1_episode()
    binary_prompt = render_binary_prompt(episode)
    narrative_prompt = render_narrative_prompt(episode)

    assert (
        "Return exactly 4 labels in order, one per probe. Use only attract or repel."
    ) in binary_prompt
    assert "Narrative is supplemental audit evidence only." in narrative_prompt
    assert "Binary remains the official leaderboard task." in narrative_prompt
    assert "rule_before: <short pre-shift rule>" in narrative_prompt
    assert "shift_evidence: <short shift evidence>" in narrative_prompt
    assert "rule_after: <short post-shift rule>" in narrative_prompt
    assert "final_decision: attract, repel, repel, attract" in narrative_prompt
