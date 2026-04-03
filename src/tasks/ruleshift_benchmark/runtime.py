from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, StrEnum
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Final


class RuleName(StrEnum):
    R_STD = "R_std"
    R_INV = "R_inv"

    @property
    def opposite(self) -> RuleName:
        return RuleName.R_INV if self is RuleName.R_STD else RuleName.R_STD


class InteractionLabel(StrEnum):
    ZARK = "zark"
    BLIM = "blim"


class TemplateId(StrEnum):
    T1 = "T1"
    T2 = "T2"
    T3 = "T3"


class TemplateFamily(StrEnum):
    CANONICAL = "canonical"
    OBSERVATION_LOG = "observation_log"
    CASE_LEDGER = "case_ledger"


class Transition(StrEnum):
    R_STD_TO_R_INV = "R_std_to_R_inv"
    R_INV_TO_R_STD = "R_inv_to_R_std"


class Split(StrEnum):
    PUBLIC = "public"
    PRIVATE = "private"


class Difficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class DifficultyProfileId(StrEnum):
    EASY_ANCHORED = "easy_anchored"
    MEDIUM_BALANCED = "medium_balanced"
    HARD_INTERLEAVED = "hard_interleaved"


class FactorLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Phase(StrEnum):
    PRE = "pre"
    POST = "post"


class ItemKind(StrEnum):
    LABELED = "labeled"
    PROBE = "probe"


class Label(str, Enum):
    type_a = "type_a"
    type_b = "type_b"

MARKER_VALUES: Final[tuple[int, ...]] = (-3, -2, -1, 1, 2, 3)
CASE_SPACE: Final[tuple[tuple[int, int], ...]] = tuple(
    (q1, q2) for q1 in MARKER_VALUES for q2 in MARKER_VALUES
)
PROBE_COUNT: Final[int] = 4
LABELED_ITEM_COUNT: Final[int] = 5
EPISODE_LENGTH: Final[int] = LABELED_ITEM_COUNT + PROBE_COUNT

SPEC_VERSION: Final[str] = "v1"
GENERATOR_VERSION: Final[str] = "R13"
TEMPLATE_SET_VERSION: Final[str] = "v2"
DIFFICULTY_VERSION: Final[str] = "R13"
MANIFEST_VERSION: Final[str] = "R14"
TASK_NAME: Final[str] = "ruleshift_benchmark"
PRIVATE_DATASET_ROOT_ENV_VAR: Final[str] = "RULESHIFT_PRIVATE_DATASET_ROOT"

_MAX_ATTEMPTS: Final[int] = 10_000
_PRIVATE_EPISODES_FILENAME: Final[str] = "private_episodes.json"
_PRIVATE_ARTIFACT_SCHEMA: Final[str] = "private_split_artifact.v1"
_MANIFEST_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "frozen_splits"
_KAGGLE_SEARCH_ROOTS: Final[tuple[Path, ...]] = (Path("/kaggle/input"),)

_PUBLIC_LABEL_MAP: Final[dict[str, InteractionLabel]] = {
    "type_a": InteractionLabel.ZARK,
    "type_b": InteractionLabel.BLIM,
}
_INTERNAL_TO_PUBLIC: Final[dict[InteractionLabel, str]] = {
    InteractionLabel.ZARK: "type_a",
    InteractionLabel.BLIM: "type_b",
}
_FACTOR_SCORE: Final[dict[FactorLevel, int]] = {
    FactorLevel.LOW: 0,
    FactorLevel.MEDIUM: 1,
    FactorLevel.HIGH: 2,
}
_DIFFICULTY_ORDER: Final[tuple[Difficulty, ...]] = (
    Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD,
)
_TEMPLATE_CHOICES: Final[tuple[TemplateId, ...]] = (
    TemplateId.T1, TemplateId.T2, TemplateId.T3,
)
_FAMILY_CHOICES: Final[tuple[TemplateFamily, ...]] = (
    TemplateFamily.CANONICAL, TemplateFamily.OBSERVATION_LOG, TemplateFamily.CASE_LEDGER,
)
_TRANSITION_CHOICES: Final[tuple[Transition, ...]] = (
    Transition.R_STD_TO_R_INV, Transition.R_INV_TO_R_STD,
)
_PROBE_SIGN_ORDER: Final[tuple[str, ...]] = ("++", "--", "+-", "-+")

@dataclass(frozen=True, slots=True)
class TemplateSpec:
    template_id: TemplateId
    pre_count: int
    post_labeled_count: int
    probe_count: int = PROBE_COUNT

    @property
    def shift_after_position(self) -> int:
        return self.pre_count

    @property
    def total_items(self) -> int:
        return self.pre_count + self.post_labeled_count + self.probe_count


TEMPLATES: Final[dict[TemplateId, TemplateSpec]] = {
    TemplateId.T1: TemplateSpec(TemplateId.T1, 2, 3),
    TemplateId.T2: TemplateSpec(TemplateId.T2, 3, 2),
    TemplateId.T3: TemplateSpec(TemplateId.T3, 1, 4),
}

def sign(v: int) -> int:
    return 1 if v > 0 else -1


def same_sign(a: int, b: int) -> bool:
    return sign(a) == sign(b)


def label(rule: RuleName, q1: int, q2: int) -> InteractionLabel:
    if same_sign(q1, q2):
        return InteractionLabel.BLIM if rule is RuleName.R_STD else InteractionLabel.ZARK
    return InteractionLabel.ZARK if rule is RuleName.R_STD else InteractionLabel.BLIM


def format_public_label(lbl: InteractionLabel | str) -> str:
    if isinstance(lbl, str):
        lbl = InteractionLabel(lbl)
    return _INTERNAL_TO_PUBLIC[lbl]


def format_public_state(lbl: InteractionLabel | str) -> str:
    return InteractionLabel(lbl).value if isinstance(lbl, str) else lbl.value


def parse_public_label(value: str) -> InteractionLabel:
    normalized = value.strip().lower()
    if normalized in _PUBLIC_LABEL_MAP:
        return _PUBLIC_LABEL_MAP[normalized]
    raise ValueError(f"unknown public label: {value}")

@dataclass(frozen=True, slots=True)
class DifficultyFactors:
    conflict_strength: FactorLevel
    post_shift_evidence_clarity: FactorLevel
    probe_ambiguity: FactorLevel
    evidence_to_final_probe_distance: FactorLevel
    pre_shift_distractor_pressure: FactorLevel


@dataclass(frozen=True, slots=True)
class EpisodeItem:
    position: int
    phase: Phase
    kind: ItemKind
    q1: int
    q2: int
    label: InteractionLabel | None = None


@dataclass(frozen=True, slots=True)
class ProbeMetadata:
    position: int
    is_disagreement_probe: bool
    old_rule_label: InteractionLabel
    new_rule_label: InteractionLabel


@dataclass(frozen=True, slots=True)
class Episode:
    episode_id: str
    split: Split
    difficulty: Difficulty
    template_id: TemplateId
    template_family: TemplateFamily
    rule_A: RuleName
    rule_B: RuleName
    transition: Transition
    pre_count: int
    post_labeled_count: int
    shift_after_position: int
    contradiction_count_post: int
    difficulty_profile_id: DifficultyProfileId
    difficulty_factors: DifficultyFactors
    items: tuple[EpisodeItem, ...]
    probe_targets: tuple[InteractionLabel, ...]
    probe_label_counts: tuple[tuple[InteractionLabel, int], ...]
    probe_sign_pattern_counts: tuple[tuple[str, int], ...]
    probe_metadata: tuple[ProbeMetadata, ...]
    difficulty_version: str = DIFFICULTY_VERSION
    spec_version: str = SPEC_VERSION
    generator_version: str = GENERATOR_VERSION
    template_set_version: str = TEMPLATE_SET_VERSION


@dataclass(frozen=True, slots=True)
class FrozenSplitManifest:
    partition: str
    episode_split: Split
    manifest_version: str
    seed_bank_version: str
    spec_version: str
    generator_version: str
    template_set_version: str
    difficulty_version: str
    seeds: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class FrozenSplitEpisode:
    partition: str
    seed: int
    manifest_version: str
    seed_bank_version: str
    episode: Episode


@dataclass(frozen=True)
class BinaryResponse:
    probe_6: Label
    probe_7: Label
    probe_8: Label
    probe_9: Label

    def as_tuple(self) -> tuple[str, str, str, str]:
        return (
            _coerce_label(self.probe_6, "probe_6"),
            _coerce_label(self.probe_7, "probe_7"),
            _coerce_label(self.probe_8, "probe_8"),
            _coerce_label(self.probe_9, "probe_9"),
        )


class KaggleExecutionError(RuntimeError):
    pass

def _probe_sign(q1: int, q2: int) -> str:
    if q1 > 0 and q2 > 0:
        return "++"
    if q1 < 0 and q2 < 0:
        return "--"
    if q1 > 0 and q2 < 0:
        return "+-"
    return "-+"


def _updated_signs(post_items: tuple[EpisodeItem, ...]) -> frozenset[str]:
    return frozenset(_probe_sign(i.q1, i.q2) for i in post_items)


def _is_same_sign_pattern(p: str) -> bool:
    return p in {"++", "--"}


def _has_mixed_polarity(patterns: frozenset[str]) -> bool:
    return (
        len(patterns) == 2
        and any(_is_same_sign_pattern(p) for p in patterns)
        and any(not _is_same_sign_pattern(p) for p in patterns)
    )


def _effective_targets(
    probes: tuple[EpisodeItem, ...],
    rule_a: RuleName,
    rule_b: RuleName,
    updated: frozenset[str],
) -> tuple[InteractionLabel, ...]:
    return tuple(
        label(rule_b if _probe_sign(i.q1, i.q2) in updated else rule_a, i.q1, i.q2)
        for i in probes
    )


def _contradiction_count(
    labeled: tuple[EpisodeItem, ...],
    pre_count: int,
    rule_a: RuleName,
    rule_b: RuleName,
) -> int:
    return sum(
        label(rule_a, i.q1, i.q2) != label(rule_b, i.q1, i.q2)
        for i in labeled[pre_count:]
    )


def _probe_label_counts(
    targets: tuple[InteractionLabel, ...],
) -> tuple[tuple[InteractionLabel, int], ...]:
    return (
        (InteractionLabel.ZARK, targets.count(InteractionLabel.ZARK)),
        (InteractionLabel.BLIM, targets.count(InteractionLabel.BLIM)),
    )


def _probe_sign_counts(
    probes: tuple[EpisodeItem, ...],
) -> tuple[tuple[str, int], ...]:
    return tuple(
        (p, sum(_probe_sign(i.q1, i.q2) == p for i in probes))
        for p in _PROBE_SIGN_ORDER
    )


def _probe_metadata(
    probes: tuple[EpisodeItem, ...],
    rule_a: RuleName,
    rule_b: RuleName,
) -> tuple[ProbeMetadata, ...]:
    return tuple(
        ProbeMetadata(
            position=i.position,
            is_disagreement_probe=(
                label(RuleName.R_STD, i.q1, i.q2) != label(RuleName.R_INV, i.q1, i.q2)
            ),
            old_rule_label=label(rule_a, i.q1, i.q2),
            new_rule_label=label(rule_b, i.q1, i.q2),
        )
        for i in probes
    )


def _count_switches(probes: tuple[EpisodeItem, ...], updated: frozenset[str]) -> int:
    rules = tuple(
        "new" if _probe_sign(i.q1, i.q2) in updated else "old" for i in probes
    )
    return sum(a != b for a, b in zip(rules, rules[1:]))


def _post_pattern_counts(post_items: tuple[EpisodeItem, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for i in post_items:
        p = _probe_sign(i.q1, i.q2)
        counts[p] = counts.get(p, 0) + 1
    return counts


def _derive_difficulty_factors(
    items: tuple[EpisodeItem, ...],
    pre_count: int,
) -> DifficultyFactors:
    labeled = items[:LABELED_ITEM_COUNT]
    post_labeled = labeled[pre_count:]
    probes = items[LABELED_ITEM_COUNT:]
    pre_patterns = frozenset(_probe_sign(i.q1, i.q2) for i in labeled[:pre_count])
    updated = _updated_signs(post_labeled)
    switches = _count_switches(probes, updated)

    if switches <= 1:
        cs = FactorLevel.LOW
    elif switches == 2:
        cs = FactorLevel.MEDIUM
    else:
        cs = FactorLevel.HIGH

    retained = sum(
        _probe_sign(i.q1, i.q2) not in updated
        and _probe_sign(i.q1, i.q2) in pre_patterns
        for i in probes
    )
    if retained == 0:
        pa = FactorLevel.LOW
    elif retained == 1:
        pa = FactorLevel.MEDIUM
    else:
        pa = FactorLevel.HIGH

    pre_overlap = sum(_probe_sign(i.q1, i.q2) in pre_patterns for i in probes)
    if pre_overlap <= 1:
        dp = FactorLevel.LOW
    elif pre_overlap == 2:
        dp = FactorLevel.MEDIUM
    else:
        dp = FactorLevel.HIGH

    ppc = _post_pattern_counts(post_labeled)
    final_p = _probe_sign(probes[-1].q1, probes[-1].q2)
    if max(ppc.values()) >= 2 and final_p in updated:
        clarity = FactorLevel.HIGH
    elif max(ppc.values()) >= 2:
        clarity = FactorLevel.MEDIUM
    else:
        clarity = FactorLevel.LOW

    if final_p not in updated:
        dist = FactorLevel.HIGH
    else:
        last = max(
            i.position for i in post_labeled if _probe_sign(i.q1, i.q2) == final_p
        )
        if last == LABELED_ITEM_COUNT:
            dist = FactorLevel.LOW
        elif last == LABELED_ITEM_COUNT - 1:
            dist = FactorLevel.MEDIUM
        else:
            dist = FactorLevel.HIGH

    return DifficultyFactors(
        conflict_strength=cs,
        post_shift_evidence_clarity=clarity,
        probe_ambiguity=pa,
        evidence_to_final_probe_distance=dist,
        pre_shift_distractor_pressure=dp,
    )


def _derive_difficulty_profile(
    f: DifficultyFactors,
) -> tuple[Difficulty, DifficultyProfileId]:
    clarity_load = {
        FactorLevel.HIGH: FactorLevel.LOW,
        FactorLevel.MEDIUM: FactorLevel.MEDIUM,
        FactorLevel.LOW: FactorLevel.HIGH,
    }[f.post_shift_evidence_clarity]
    score = sum(
        _FACTOR_SCORE[lvl]
        for lvl in (
            f.conflict_strength,
            clarity_load,
            f.probe_ambiguity,
            f.evidence_to_final_probe_distance,
            f.pre_shift_distractor_pressure,
        )
    )
    if score <= 3:
        return Difficulty.EASY, DifficultyProfileId.EASY_ANCHORED
    if score >= 6:
        return Difficulty.HARD, DifficultyProfileId.HARD_INTERLEAVED
    return Difficulty.MEDIUM, DifficultyProfileId.MEDIUM_BALANCED

def _build_items(
    pairs: tuple[tuple[int, int], ...],
    pre_count: int,
    rule_a: RuleName,
    rule_b: RuleName,
) -> tuple[EpisodeItem, ...]:
    items: list[EpisodeItem] = []
    for pos, (q1, q2) in enumerate(pairs, 1):
        if pos <= pre_count:
            items.append(EpisodeItem(pos, Phase.PRE, ItemKind.LABELED, q1, q2, label(rule_a, q1, q2)))
        elif pos <= LABELED_ITEM_COUNT:
            items.append(EpisodeItem(pos, Phase.POST, ItemKind.LABELED, q1, q2, label(rule_b, q1, q2)))
        else:
            items.append(EpisodeItem(pos, Phase.POST, ItemKind.PROBE, q1, q2))
    return tuple(items)


def _is_valid_candidate(
    contradiction: int,
    items: tuple[EpisodeItem, ...],
    pre_count: int,
    rule_a: RuleName,
    rule_b: RuleName,
    targets: tuple[InteractionLabel, ...],
) -> bool:
    probes = items[LABELED_ITEM_COUNT:]
    updated = _updated_signs(items[pre_count:LABELED_ITEM_COUNT])
    full_a = tuple(label(rule_a, i.q1, i.q2) for i in probes)
    full_b = tuple(label(rule_b, i.q1, i.q2) for i in probes)
    return (
        contradiction >= 1
        and _has_mixed_polarity(updated)
        and _probe_sign_counts(probes) == tuple((p, 1) for p in _PROBE_SIGN_ORDER)
        and len(set(targets)) >= 2
        and targets != full_a
        and targets != full_b
    )


def _generate_episode(seed: int, split: Split = Split.PUBLIC) -> Episode:
    rng = random.Random(seed)
    target_diff = _DIFFICULTY_ORDER[seed % 3]
    target_tid = _TEMPLATE_CHOICES[(seed // 3) % 3]
    target_fam = _FAMILY_CHOICES[(seed // 9) % 3]
    target_trans = _TRANSITION_CHOICES[(seed // 27) % 2]
    rule_a = RuleName.R_STD if target_trans is Transition.R_STD_TO_R_INV else RuleName.R_INV
    rule_b = rule_a.opposite
    tmpl = TEMPLATES[target_tid]

    for _ in range(_MAX_ATTEMPTS):
        pairs = tuple(rng.sample(CASE_SPACE, k=tmpl.total_items))
        items = _build_items(pairs, tmpl.pre_count, rule_a, rule_b)
        updated = _updated_signs(items[tmpl.pre_count : LABELED_ITEM_COUNT])
        targets = _effective_targets(items[LABELED_ITEM_COUNT:], rule_a, rule_b, updated)
        cc = _contradiction_count(items[:LABELED_ITEM_COUNT], tmpl.pre_count, rule_a, rule_b)
        if not _is_valid_candidate(cc, items, tmpl.pre_count, rule_a, rule_b, targets):
            continue
        df = _derive_difficulty_factors(items, tmpl.pre_count)
        d, dp = _derive_difficulty_profile(df)
        if d is target_diff:
            break
    else:
        raise RuntimeError(f"generate_episode exhausted attempts for seed={seed}")

    return Episode(
        episode_id=f"ife-r13-{seed}",
        split=split,
        difficulty=d,
        template_id=target_tid,
        template_family=target_fam,
        rule_A=rule_a,
        rule_B=rule_b,
        transition=Transition(target_trans),
        pre_count=tmpl.pre_count,
        post_labeled_count=tmpl.post_labeled_count,
        shift_after_position=tmpl.pre_count,
        contradiction_count_post=cc,
        difficulty_profile_id=dp,
        difficulty_factors=df,
        items=items,
        probe_targets=targets,
        probe_label_counts=_probe_label_counts(targets),
        probe_sign_pattern_counts=_probe_sign_counts(items[LABELED_ITEM_COUNT:]),
        probe_metadata=_probe_metadata(items[LABELED_ITEM_COUNT:], rule_a, rule_b),
    )

def load_split_manifest(
    partition: str,
    *,
    private_dataset_root: Path | str | None = None,
) -> FrozenSplitManifest:
    if partition == "private_leaderboard":
        root = resolve_private_dataset_root(private_dataset_root)
        payload = json.loads((root / _PRIVATE_EPISODES_FILENAME).read_text("utf-8"))
        records = _parse_private_episodes(payload)
        return FrozenSplitManifest(
            partition="private_leaderboard",
            episode_split=Split(payload["episode_split"]),
            manifest_version=payload["benchmark_version"],
            seed_bank_version=payload["artifact_checksum"],
            spec_version=SPEC_VERSION,
            generator_version=GENERATOR_VERSION,
            template_set_version=TEMPLATE_SET_VERSION,
            difficulty_version=DIFFICULTY_VERSION,
            seeds=tuple(r.seed for r in records),
        )

    payload = json.loads((_MANIFEST_DIR / f"{partition}.json").read_text("utf-8"))
    return FrozenSplitManifest(
        partition=payload["partition"],
        episode_split=Split(payload["episode_split"]),
        manifest_version=payload["manifest_version"],
        seed_bank_version=payload["seed_bank_version"],
        spec_version=payload["spec_version"],
        generator_version=payload["generator_version"],
        template_set_version=payload["template_set_version"],
        difficulty_version=payload["difficulty_version"],
        seeds=tuple(payload["seeds"]),
    )


def load_frozen_split(
    partition: str,
    *,
    private_dataset_root: Path | str | None = None,
) -> tuple[FrozenSplitEpisode, ...]:
    if partition == "private_leaderboard":
        return _load_private_split(private_dataset_root)
    manifest = load_split_manifest(partition)
    return tuple(
        FrozenSplitEpisode(
            partition=manifest.partition,
            seed=seed,
            manifest_version=manifest.manifest_version,
            seed_bank_version=manifest.seed_bank_version,
            episode=_generate_episode(seed, split=manifest.episode_split),
        )
        for seed in manifest.seeds
    )


def discover_private_dataset_root(
    private_dataset_root: Path | str | None = None,
) -> Path | None:
    if private_dataset_root is not None:
        root = Path(private_dataset_root)
        if (root / _PRIVATE_EPISODES_FILENAME).is_file():
            return root
        raise FileNotFoundError(
            f"{_PRIVATE_EPISODES_FILENAME} not found at {root}"
        )

    env = os.environ.get(PRIVATE_DATASET_ROOT_ENV_VAR)
    if env:
        root = Path(env)
        if (root / _PRIVATE_EPISODES_FILENAME).is_file():
            return root
        raise FileNotFoundError(
            f"{_PRIVATE_EPISODES_FILENAME} not found at {root}"
        )

    for search_root in _KAGGLE_SEARCH_ROOTS:
        if not search_root.exists():
            continue
        for path in search_root.rglob(_PRIVATE_EPISODES_FILENAME):
            return path.parent

    return None


def resolve_private_dataset_root(
    private_dataset_root: Path | str | None = None,
) -> Path:
    root = discover_private_dataset_root(private_dataset_root)
    if root is not None:
        return root
    raise FileNotFoundError(
        f"Private dataset not found. Set {PRIVATE_DATASET_ROOT_ENV_VAR} or "
        "attach the authorized private dataset mount."
    )


def _load_private_split(
    private_dataset_root: Path | str | None = None,
) -> tuple[FrozenSplitEpisode, ...]:
    root = resolve_private_dataset_root(private_dataset_root)
    payload = json.loads((root / _PRIVATE_EPISODES_FILENAME).read_text("utf-8"))
    return _parse_private_episodes(payload)


def _parse_private_episodes(payload: dict) -> tuple[FrozenSplitEpisode, ...]:
    checksum_input = {
        "partition": payload["partition"],
        "episode_split": payload["episode_split"],
        "benchmark_version": payload["benchmark_version"],
        "schema_version": payload["schema_version"],
        "episodes": payload["episodes"],
    }
    expected = hashlib.sha256(
        json.dumps(checksum_input, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if payload["artifact_checksum"] != expected:
        raise ValueError("artifact_checksum mismatch")

    version = payload["benchmark_version"]
    checksum = payload["artifact_checksum"]
    return tuple(
        FrozenSplitEpisode(
            partition="private_leaderboard",
            seed=row["seed"],
            manifest_version=version,
            seed_bank_version=checksum,
            episode=_build_episode_from_json(row["episode"]),
        )
        for row in payload["episodes"]
    )


def _build_episode_from_json(ep: dict) -> Episode:
    items = tuple(
        EpisodeItem(
            position=i["position"],
            phase=Phase(i["phase"]),
            kind=ItemKind(i["kind"]),
            q1=i["q1"],
            q2=i["q2"],
            label=InteractionLabel(i["label"]) if i.get("label") else None,
        )
        for i in ep["items"]
    )
    pre_count = ep["pre_count"]
    df = _derive_difficulty_factors(items, pre_count)
    d, dp = _derive_difficulty_profile(df)
    targets = tuple(InteractionLabel(t) for t in ep["probe_targets"])

    return Episode(
        episode_id=ep["episode_id"],
        split=Split(ep["split"]),
        difficulty=Difficulty(ep.get("difficulty", d.value)),
        template_id=TemplateId(ep["template_id"]),
        template_family=TemplateFamily(ep.get("template_family", "canonical")),
        rule_A=RuleName(ep["rule_A"]),
        rule_B=RuleName(ep["rule_B"]),
        transition=Transition(ep["transition"]),
        pre_count=pre_count,
        post_labeled_count=ep["post_labeled_count"],
        shift_after_position=ep["shift_after_position"],
        contradiction_count_post=ep["contradiction_count_post"],
        difficulty_profile_id=DifficultyProfileId(
            ep.get("difficulty_profile_id", dp.value)
        ),
        difficulty_factors=df,
        items=items,
        probe_targets=targets,
        probe_label_counts=tuple(
            (InteractionLabel(p[0]), p[1]) for p in ep["probe_label_counts"]
        ),
        probe_sign_pattern_counts=tuple(
            (p[0], p[1]) for p in ep["probe_sign_pattern_counts"]
        ),
        probe_metadata=tuple(
            ProbeMetadata(
                position=m["position"],
                is_disagreement_probe=m["is_disagreement_probe"],
                old_rule_label=InteractionLabel(m["old_rule_label"]),
                new_rule_label=InteractionLabel(m["new_rule_label"]),
            )
            for m in ep["probe_metadata"]
        ),
        difficulty_version=ep.get("difficulty_version", DIFFICULTY_VERSION),
        spec_version=ep.get("spec_version", SPEC_VERSION),
        generator_version=ep.get("generator_version", GENERATOR_VERSION),
        template_set_version=ep.get("template_set_version", TEMPLATE_SET_VERSION),
    )

_BINARY_OUTRO: Final[str] = (
    "Return exactly 4 outputs in order, one per probe. "
    "Use only type_a or type_b. Map zark to type_a and blim to type_b."
)

_BINARY_FAMILIES: Final[dict[TemplateFamily, tuple[str, str, str, str]]] = {
    TemplateFamily.CANONICAL: (
        "You are given labeled records for two markers.\n"
        "Each labeled line shows r1, r2, and the observed state.\n"
        "Use the full sequence to infer which sign combinations were revised by the later evidence, "
        "then answer the final unlabeled cases.",
        "Labeled examples:",
        "Probes:",
        "{pos}. r1={q1}, r2={q2} -> {out}",
    ),
    TemplateFamily.OBSERVATION_LOG: (
        "Review the observation log for two markers.\n"
        "Each entry records r1, r2, and the observed state.\n"
        "Use the full log to infer which sign combinations were revised later, "
        "then answer the unlabeled probe entries.",
        "Resolved log entries:",
        "Unresolved probe entries:",
        "[{pos:02d}] r1={q1} | r2={q2} | observed={out}",
    ),
    TemplateFamily.CASE_LEDGER: (
        "Review the case ledger for two markers.\n"
        "Each row records r1, r2, and the observed state.\n"
        "Use the full ledger to infer which sign combinations were revised by the later evidence, "
        "then complete the pending rows.",
        "Confirmed ledger rows:",
        "Pending ledger rows:",
        "row {pos:02d} | r1={q1} | r2={q2} | state={out}",
    ),
}


def render_binary_prompt(episode: Episode) -> str:
    labeled = episode.items[:LABELED_ITEM_COUNT]
    probes = episode.items[LABELED_ITEM_COUNT:]
    intro, lbl_hdr, prb_hdr, fmt = _BINARY_FAMILIES[episode.template_family]

    def _fmt(item: EpisodeItem) -> str:
        out = format_public_state(item.label) if item.label is not None else "?"
        return fmt.format(
            pos=item.position, q1=f"{item.q1:+d}", q2=f"{item.q2:+d}", out=out,
        )

    return "\n".join((
        intro, "",
        lbl_hdr, *(_fmt(i) for i in labeled), "",
        prb_hdr, *(_fmt(i) for i in probes), "",
        _BINARY_OUTRO,
    ))

def build_benchmark_bundle(
    *,
    include_private: bool = True,
    private_dataset_root: Path | str | None = None,
) -> dict[str, object]:
    partitions = [_build_partition("public_leaderboard")]
    if include_private:
        root = (
            resolve_private_dataset_root(private_dataset_root)
            if private_dataset_root is not None
            else discover_private_dataset_root()
        )
        if root is not None:
            partitions.append(
                _build_partition("private_leaderboard", private_dataset_root=root)
            )
    return {
        "task": TASK_NAME,
        "benchmark_version": MANIFEST_VERSION,
        "partitions": partitions,
    }


def _build_partition(
    partition: str,
    *,
    private_dataset_root: Path | str | None = None,
) -> dict[str, object]:
    manifest = load_split_manifest(partition, private_dataset_root=private_dataset_root)
    records = load_frozen_split(partition, private_dataset_root=private_dataset_root)
    return {
        "partition": manifest.partition,
        "episode_split": manifest.episode_split.value,
        "manifest_version": manifest.manifest_version,
        "seed_bank_version": manifest.seed_bank_version,
        "episode_count": len(records),
        "episodes": [
            {
                "seed": r.seed,
                "episode_id": r.episode.episode_id,
                "prompt_binary": render_binary_prompt(r.episode),
                "probe_targets": [
                    format_public_label(lbl) for lbl in r.episode.probe_targets
                ],
            }
            for r in records
        ],
    }


def build_leaderboard_rows(bundle: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for partition in bundle["partitions"]:
        for ep in partition["episodes"]:
            rows.append({
                "episode_id": ep["episode_id"],
                "split": partition["partition"],
                "prompt_binary": ep["prompt_binary"],
                "probe_targets": tuple(ep["probe_targets"]),
            })
    return rows

def run_binary_task(
    *,
    llm: object,
    prompt_binary: str,
    probe_targets: tuple[str, ...] | tuple[InteractionLabel, ...],
) -> tuple[int, int]:
    try:
        response = llm.prompt(prompt_binary, schema=BinaryResponse)
    except Exception as exc:
        raise KaggleExecutionError("llm.prompt failed") from exc

    try:
        normalized = normalize_binary_response(response)
    except ValueError as exc:
        raise KaggleExecutionError(f"invalid binary response: {exc}") from exc

    if normalized is None:
        raise KaggleExecutionError(
            f"unscoreable response of type {type(response).__name__}"
        )
    return score_episode(normalized, probe_targets)


def normalize_binary_response(response: object) -> tuple[str, ...] | None:
    if response is None:
        return None
    if isinstance(response, BinaryResponse):
        return response.as_tuple()
    if isinstance(response, str):
        return _parse_text_response(response)
    br = _try_coerce(response)
    return br.as_tuple() if br is not None else None


def score_episode(
    predictions: tuple[str, ...] | tuple[InteractionLabel, ...] | None,
    targets: tuple[str, ...] | tuple[InteractionLabel, ...],
) -> tuple[int, int]:
    norm_targets = _norm_labels(targets)
    if norm_targets is None:
        raise ValueError(f"targets must contain exactly {PROBE_COUNT} valid labels")
    norm_preds = _norm_labels(predictions)
    if norm_preds is None:
        return (0, PROBE_COUNT)
    return (
        sum(p is t for p, t in zip(norm_preds, norm_targets)),
        PROBE_COUNT,
    )


def _parse_text_response(text: str) -> tuple[str, ...] | None:
    tokens = tuple(
        t.strip().lower()
        for t in text.strip().strip("`").replace("\n", ",").split(",")
        if t.strip()
    )
    if len(tokens) != PROBE_COUNT:
        return None
    try:
        return tuple(format_public_label(parse_public_label(t)) for t in tokens)
    except ValueError:
        return None


_PROBE_FIELDS: Final[tuple[str, ...]] = ("probe_6", "probe_7", "probe_8", "probe_9")


def _try_coerce(response: object) -> BinaryResponse | None:
    if isinstance(response, dict):
        vals = response
    elif hasattr(response, "__getitem__") and hasattr(response, "keys"):
        try:
            vals = {f: response[f] for f in _PROBE_FIELDS}
        except (KeyError, TypeError):
            return None
    elif all(hasattr(response, f) for f in _PROBE_FIELDS):
        vals = {f: getattr(response, f) for f in _PROBE_FIELDS}
    else:
        return None
    try:
        labels = tuple(Label(_coerce_label(vals[f], f)) for f in _PROBE_FIELDS)
    except (KeyError, TypeError):
        return None
    return BinaryResponse(*labels)


def _coerce_label(value: object, field: str) -> str:
    if isinstance(value, str):
        raw = value
    elif isinstance(value, Enum):
        raw = value.value
    elif hasattr(value, "value"):
        raw = str(value.value)
    else:
        raw = str(value)
    try:
        return format_public_label(parse_public_label(raw))
    except ValueError as exc:
        raise ValueError(f"invalid field {field}: {raw!r}") from exc


def _norm_labels(
    labels: tuple[str, ...] | tuple[InteractionLabel, ...] | None,
) -> tuple[InteractionLabel, ...] | None:
    if labels is None:
        return None
    try:
        result = tuple(
            lbl if isinstance(lbl, InteractionLabel) else parse_public_label(lbl)
            for lbl in labels
        )
    except ValueError:
        return None
    return result if len(result) == PROBE_COUNT else None
