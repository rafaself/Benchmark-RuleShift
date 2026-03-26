import pytest

from protocol import (
    CASE_SPACE,
    CHARGES,
    DIFFICULTIES,
    EPISODE_LENGTH,
    ITEM_KINDS,
    PHASES,
    PROBE_COUNT,
    RULES,
    SPLITS,
    TEMPLATE_IDS,
    TEMPLATE_FAMILIES,
    TEMPLATES,
    Difficulty,
    InteractionLabel,
    ItemKind,
    Phase,
    RuleName,
    Split,
    TemplateFamily,
    TemplateSpec,
    TemplateId,
    Transition,
    parse_difficulty,
    parse_item_kind,
    parse_label,
    parse_phase,
    parse_rule,
    parse_split,
    parse_template_family,
    parse_template_id,
    parse_transition,
)


def test_rule_parser_accepts_enums_and_canonical_strings():
    assert parse_rule(RuleName.R_STD) is RuleName.R_STD
    assert parse_rule("R_inv") is RuleName.R_INV


@pytest.mark.parametrize(
    ("parser", "enum_value", "string_value", "expected"),
    [
        (parse_label, InteractionLabel.ATTRACT, "attract", InteractionLabel.ATTRACT),
        (parse_template_id, TemplateId.T1, "T2", TemplateId.T2),
        (
            parse_template_family,
            TemplateFamily.CANONICAL,
            "observation_log",
            TemplateFamily.OBSERVATION_LOG,
        ),
        (
            parse_transition,
            Transition.R_STD_TO_R_INV,
            "R_inv_to_R_std",
            Transition.R_INV_TO_R_STD,
        ),
        (parse_split, Split.DEV, "public", Split.PUBLIC),
        (parse_difficulty, Difficulty.EASY, "hard", Difficulty.HARD),
        (parse_phase, Phase.PRE, "post", Phase.POST),
        (parse_item_kind, ItemKind.LABELED, "probe", ItemKind.PROBE),
    ],
)
def test_all_enum_parsers_accept_enum_instances_and_canonical_strings(
    parser, enum_value, string_value, expected
):
    assert parser(enum_value) is enum_value
    assert parser(string_value) is expected


def test_rule_parser_rejects_typos_with_canonical_values():
    with pytest.raises(ValueError, match=r"unknown rule: R-std"):
        parse_rule("R-std")


def test_label_parser_rejects_noncanonical_values():
    with pytest.raises(ValueError, match=r"unknown label: repells"):
        parse_label("repells")


def test_template_parser_rejects_noncanonical_values():
    with pytest.raises(ValueError, match=r"unknown template_id: t1"):
        parse_template_id("t1")


def test_template_family_parser_rejects_noncanonical_values():
    with pytest.raises(ValueError, match=r"unknown template_family: log"):
        parse_template_family("log")


@pytest.mark.parametrize(
    ("parser", "value", "field_name"),
    [
        (parse_transition, "R_std_to_R_std", "transition"),
        (parse_split, "prod", "split"),
        (parse_difficulty, "expert", "difficulty"),
        (parse_phase, "during", "phase"),
        (parse_item_kind, "target", "item_kind"),
    ],
)
def test_other_enum_parsers_reject_noncanonical_values(parser, value, field_name):
    with pytest.raises(ValueError, match=rf"unknown {field_name}: {value}"):
        parser(value)


def test_rules_expose_canonical_opposites_and_transitions():
    assert RuleName.R_STD.opposite is RuleName.R_INV
    assert RuleName.R_INV.opposite is RuleName.R_STD
    assert Transition.from_rules(RuleName.R_STD, RuleName.R_INV) == Transition.R_STD_TO_R_INV
    assert Transition.from_rules(RuleName.R_INV, RuleName.R_STD) == Transition.R_INV_TO_R_STD


@pytest.mark.parametrize(
    ("member", "expected"),
    [
        (RuleName.R_STD, "R_std"),
        (InteractionLabel.REPEL, "repel"),
        (TemplateId.T2, "T2"),
        (TemplateFamily.OBSERVATION_LOG, "observation_log"),
        (Transition.R_INV_TO_R_STD, "R_inv_to_R_std"),
        (Split.PRIVATE, "private"),
        (Difficulty.MEDIUM, "medium"),
        (Phase.POST, "post"),
        (ItemKind.PROBE, "probe"),
    ],
)
def test_protocol_enums_preserve_raw_string_behavior(member, expected):
    assert str(member) == expected
    assert isinstance(member, str)
    assert member == expected


def test_template_specs_match_frozen_counts():
    assert RULES == frozenset(RuleName)
    assert TEMPLATE_IDS == frozenset(TemplateId)
    assert TEMPLATE_FAMILIES == frozenset(TemplateFamily)
    assert SPLITS == frozenset(Split)
    assert DIFFICULTIES == frozenset(Difficulty)
    assert PHASES == frozenset(Phase)
    assert ITEM_KINDS == frozenset(ItemKind)
    assert CHARGES == (-3, -2, -1, 1, 2, 3)
    assert frozenset(TEMPLATES) == TEMPLATE_IDS

    for template_id, spec in TEMPLATES.items():
        assert spec.template_id is template_id
        assert spec.probe_count == PROBE_COUNT
        assert spec.total_items == EPISODE_LENGTH
        assert spec.pre_count + spec.post_labeled_count == 5

    assert TEMPLATES[TemplateId.T1].pre_count == 2
    assert TEMPLATES[TemplateId.T1].post_labeled_count == 3
    assert TEMPLATES[TemplateId.T2].pre_count == 3
    assert TEMPLATES[TemplateId.T2].post_labeled_count == 2


def test_case_space_matches_the_frozen_cartesian_product():
    expected = tuple((q1, q2) for q1 in CHARGES for q2 in CHARGES)
    assert CASE_SPACE == expected


def test_templates_catalog_is_read_only():
    with pytest.raises(TypeError):
        TEMPLATES[TemplateId.T1] = TemplateSpec(
            template_id=TemplateId.T1,
            pre_count=2,
            post_labeled_count=3,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "template_id": "T1",
            "pre_count": 2,
            "post_labeled_count": 3,
            "probe_count": PROBE_COUNT,
        },
        {
            "template_id": TemplateId.T1,
            "pre_count": 0,
            "post_labeled_count": 5,
            "probe_count": PROBE_COUNT,
        },
        {
            "template_id": TemplateId.T1,
            "pre_count": 2,
            "post_labeled_count": 2,
            "probe_count": PROBE_COUNT,
        },
        {
            "template_id": TemplateId.T1,
            "pre_count": 2,
            "post_labeled_count": 3,
            "probe_count": PROBE_COUNT + 1,
        },
    ],
)
def test_template_spec_rejects_invalid_states_early(kwargs):
    with pytest.raises((TypeError, ValueError)):
        TemplateSpec(**kwargs)


def test_transition_rejects_same_rule_endpoints():
    with pytest.raises(ValueError, match="requires two distinct rules"):
        Transition.from_rules(RuleName.R_STD, RuleName.R_STD)
