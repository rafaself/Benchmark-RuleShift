import pytest

from tasks.ruleshift_benchmark.protocol import CHARGES, InteractionLabel, RuleName
from tasks.ruleshift_benchmark.rules import label


@pytest.mark.parametrize(
    ("rule", "q1", "q2", "expected"),
    [
        (RuleName.R_STD, 1, 3, InteractionLabel.REPEL),
        (RuleName.R_STD, -1, -3, InteractionLabel.REPEL),
        (RuleName.R_INV, 1, 3, InteractionLabel.ATTRACT),
        (RuleName.R_INV, -1, -3, InteractionLabel.ATTRACT),
    ],
)
def test_same_sign_behavior(rule, q1, q2, expected):
    assert label(rule, q1, q2) == expected


@pytest.mark.parametrize(
    ("rule", "q1", "q2", "expected"),
    [
        (RuleName.R_STD, 1, -3, InteractionLabel.ATTRACT),
        (RuleName.R_STD, -1, 3, InteractionLabel.ATTRACT),
        (RuleName.R_INV, 1, -3, InteractionLabel.REPEL),
        (RuleName.R_INV, -1, 3, InteractionLabel.REPEL),
    ],
)
def test_opposite_sign_behavior(rule, q1, q2, expected):
    assert label(rule, q1, q2) == expected


def test_swap_invariance():
    for rule in RuleName:
        for i, q1 in enumerate(CHARGES):
            for q2 in CHARGES[i:]:
                assert label(rule, q1, q2) == label(rule, q2, q1)


@pytest.mark.parametrize(
    ("rule", "examples"),
    [
        (
            RuleName.R_STD,
            [
                (1, 2),
                (1, 3),
                (2, 3),
                (-1, -2),
                (-1, -3),
                (-2, -3),
            ],
        ),
        (
            RuleName.R_STD,
            [
                (1, -1),
                (1, -2),
                (1, -3),
                (2, -1),
                (2, -2),
                (2, -3),
                (3, -1),
                (3, -2),
                (3, -3),
            ],
        ),
        (
            RuleName.R_INV,
            [
                (1, 2),
                (1, 3),
                (2, 3),
                (-1, -2),
                (-1, -3),
                (-2, -3),
            ],
        ),
        (
            RuleName.R_INV,
            [
                (1, -1),
                (1, -2),
                (1, -3),
                (2, -1),
                (2, -2),
                (2, -3),
                (3, -1),
                (3, -2),
                (3, -3),
            ],
        ),
    ],
)
def test_magnitude_irrelevance_with_fixed_sign_pattern(rule, examples):
    labels = {label(rule, q1, q2) for q1, q2 in examples}
    assert len(labels) == 1


def test_rules_always_disagree_for_allowed_charge_pairs():
    for q1 in CHARGES:
        for q2 in CHARGES:
            assert label(RuleName.R_STD, q1, q2) != label(RuleName.R_INV, q1, q2)


def test_invalid_rule_name_fails_cleanly():
    with pytest.raises(ValueError, match="unknown rule"):
        label("R_bad", 1, -1)


def test_label_accepts_canonical_rule_string_input():
    assert label("R_std", 1, -1) == InteractionLabel.ATTRACT
