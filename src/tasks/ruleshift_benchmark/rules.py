from tasks.ruleshift_benchmark.protocol import (
    MARKER_VALUES,
    InteractionLabel,
    RuleName,
    parse_rule,
)


def sign(marker_value: int) -> int:
    _validate_marker_value(marker_value)
    return 1 if marker_value > 0 else -1


def same_sign(r1: int, r2: int) -> bool:
    return sign(r1) == sign(r2)


def label(rule: RuleName | str, r1: int, r2: int) -> InteractionLabel:
    resolved_rule = parse_rule(rule)

    if same_sign(r1, r2):
        return (
            InteractionLabel.BLIM
            if resolved_rule is RuleName.R_STD
            else InteractionLabel.ZARK
        )

    return (
        InteractionLabel.ZARK
        if resolved_rule is RuleName.R_STD
        else InteractionLabel.BLIM
    )


def _validate_marker_value(marker_value: int) -> None:
    if marker_value not in MARKER_VALUES:
        raise ValueError(f"unsupported marker value: {marker_value}")
