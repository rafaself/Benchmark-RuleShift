from tasks.ruleshift_benchmark.protocol import (
    CHARGES,
    InteractionLabel,
    RuleName,
    parse_rule,
)


def sign(charge: int) -> int:
    _validate_charge(charge)
    return 1 if charge > 0 else -1


def same_sign(q1: int, q2: int) -> bool:
    return sign(q1) == sign(q2)


def label(rule: RuleName | str, q1: int, q2: int) -> InteractionLabel:
    resolved_rule = parse_rule(rule)

    if same_sign(q1, q2):
        return (
            InteractionLabel.REPEL
            if resolved_rule is RuleName.R_STD
            else InteractionLabel.ATTRACT
        )

    return (
        InteractionLabel.ATTRACT
        if resolved_rule is RuleName.R_STD
        else InteractionLabel.REPEL
    )


def _validate_charge(charge: int) -> None:
    if charge not in CHARGES:
        raise ValueError(f"unsupported charge: {charge}")
