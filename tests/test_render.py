from tasks.ruleshift_benchmark.generator import generate_episode
from tasks.ruleshift_benchmark.protocol import LABELED_ITEM_COUNT, TemplateFamily
from tasks.ruleshift_benchmark.render import render_binary_prompt
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


def _binary_ledger_line(item) -> str:
    outcome = item.label.value if item.label is not None else "?"
    return (
        f"row {item.position:02d} | pair=({_format_charge(item.q1)}, {_format_charge(item.q2)}) "
        f"| result={outcome}"
    )


def _rendered_binary_line(episode, item) -> str:
    if episode.template_family is TemplateFamily.OBSERVATION_LOG:
        return _binary_log_line(item)
    if episode.template_family is TemplateFamily.CASE_LEDGER:
        return _binary_ledger_line(item)
    return _binary_line(item)


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

    if episode.template_family is TemplateFamily.OBSERVATION_LOG:
        expected_labeled_heading = "Resolved log entries:"
        expected_probe_heading = "Unresolved probe entries:"
    elif episode.template_family is TemplateFamily.CASE_LEDGER:
        expected_labeled_heading = "Confirmed ledger rows:"
        expected_probe_heading = "Pending ledger rows:"
    else:
        expected_labeled_heading = "Labeled examples:"
        expected_probe_heading = "Probes:"

    assert expected_labeled_heading in prompt
    assert expected_probe_heading in prompt
    assert prompt.index(expected_labeled_heading) < prompt.index(expected_probe_heading)
    _assert_lines_appear_in_order(prompt, labeled_lines + probe_lines)


def test_render_output_does_not_expose_hidden_metadata_or_target_labels():
    episode = make_valid_t1_episode()
    probe_target_sequence = ", ".join(target.value for target in episode.probe_targets)
    binary_prompt = render_binary_prompt(episode)

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
        assert forbidden_snippet not in binary_prompt

    assert probe_target_sequence not in binary_prompt

    for probe_index, item in enumerate(episode.items[LABELED_ITEM_COUNT:]):
        assert _rendered_binary_line(episode, item) in binary_prompt
        target_label = episode.probe_targets[probe_index].value
        if episode.template_family is TemplateFamily.OBSERVATION_LOG:
            assert (
                f"[{item.position:02d}] q1={_format_charge(item.q1)} | "
                f"q2={_format_charge(item.q2)} | observed={target_label}"
            ) not in binary_prompt
        elif episode.template_family is TemplateFamily.CASE_LEDGER:
            assert (
                f"row {item.position:02d} | pair=({_format_charge(item.q1)}, {_format_charge(item.q2)}) "
                f"| result={target_label}"
            ) not in binary_prompt
        else:
            assert (
                f"{item.position}. q1={_format_charge(item.q1)}, q2={_format_charge(item.q2)} -> {target_label}"
            ) not in binary_prompt


def test_binary_render_preserves_probe_targets_from_the_source_episode():
    episode = make_valid_t2_episode()

    binary_prompt = render_binary_prompt(episode)
    for item in episode.items[LABELED_ITEM_COUNT:]:
        binary_fragment = _rendered_binary_line(episode, item)
        assert binary_fragment in binary_prompt

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

    assert (
        "Return exactly 4 labels in order, one per probe. Use only attract or repel."
    ) in binary_prompt
