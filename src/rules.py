CHARGES = (-3, -2, -1, 1, 2, 3)
RULES = frozenset({"R_std", "R_inv"})


def sign(charge: int) -> int:
    _validate_charge(charge)
    return 1 if charge > 0 else -1


def same_sign(q1: int, q2: int) -> bool:
    return sign(q1) == sign(q2)


def label(rule: str, q1: int, q2: int) -> str:
    if rule not in RULES:
        raise ValueError(f"unknown rule: {rule}")

    if same_sign(q1, q2):
        return "repel" if rule == "R_std" else "attract"

    return "attract" if rule == "R_std" else "repel"


def _validate_charge(charge: int) -> None:
    if charge not in CHARGES:
        raise ValueError(f"unsupported charge: {charge}")
