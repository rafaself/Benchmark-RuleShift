from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Callable, Final, Mapping

from tasks.ruleshift_benchmark.protocol import (
    CASE_SPACE,
    LABELED_ITEM_COUNT,
    Difficulty,
    InteractionLabel,
    ItemKind,
    Phase,
    RuleName,
    Split,
    TemplateFamily,
    TemplateId,
    TEMPLATES,
    Transition,
    format_public_label,
    format_public_state,
    label,
    parse_split,
)
from tasks.ruleshift_benchmark.schema import (
    DEFAULT_GENERATION_MAX_ATTEMPTS,
    DIFFICULTY_VERSION,
    GENERATOR_VERSION,
    SPEC_VERSION,
    TEMPLATE_SET_VERSION,
    DifficultyFactors,
    Episode,
    EpisodeItem,
    ProbeMetadata,
    build_contradiction_count_post,
    build_effective_probe_targets,
    build_probe_label_counts,
    build_probe_metadata,
    build_probe_sign_pattern_counts,
    build_updated_sign_patterns,
    derive_difficulty_factors,
    derive_difficulty_profile,
    has_both_probe_labels,
    has_mixed_polarity_sign_patterns,
    is_global_rule_probe_block,
)

__all__ = [
    "PARTITIONS",
    "PUBLIC_PARTITIONS",
    "MANIFEST_VERSION",
    "TASK_NAME",
    "FrozenSplitManifest",
    "FrozenSplitEpisode",
    "PRIVATE_EPISODES_FILENAME",
    "PRIVATE_DATASET_ROOT_ENV_VAR",
    "PRIVATE_SPLIT_ARTIFACT_SCHEMA_VERSION",
    "build_benchmark_bundle",
    "build_leaderboard_rows",
    "discover_private_dataset_root",
    "generate_episode",
    "generate_frozen_split",
    "load_frozen_split",
    "load_private_split",
    "load_split_manifest",
    "render_binary_prompt",
    "resolve_private_dataset_root",
]

PARTITIONS: Final[tuple[str, ...]] = (
    "public_leaderboard",
    "private_leaderboard",
)
PUBLIC_PARTITIONS: Final[tuple[str, ...]] = ("public_leaderboard",)
MANIFEST_VERSION: Final[str] = "R14"
PRIVATE_EPISODES_FILENAME: Final[str] = "private_episodes.json"
PRIVATE_DATASET_ROOT_ENV_VAR: Final[str] = "RULESHIFT_PRIVATE_DATASET_ROOT"
PRIVATE_SPLIT_ARTIFACT_SCHEMA_VERSION: Final[str] = "private_split_artifact.v1"
TASK_NAME: Final[str] = "ruleshift_benchmark"

_MANIFEST_FIELD_ORDER: Final[tuple[str, ...]] = (
    "partition",
    "episode_split",
    "manifest_version",
    "seed_bank_version",
    "spec_version",
    "generator_version",
    "template_set_version",
    "difficulty_version",
    "seeds",
)
_DEFAULT_MANIFEST_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "frozen_splits"
_PARTITION_TO_EPISODE_SPLIT: Final[dict[str, Split]] = {
    "public_leaderboard": Split.PUBLIC,
    "private_leaderboard": Split.PRIVATE,
}
_EXPECTED_PRIVATE_PARTITION: Final[str] = "private_leaderboard"
_EXPECTED_PRIVATE_EPISODE_SPLIT: Final[str] = "private"
_KAGGLE_PRIVATE_SEARCH_ROOTS: Final[tuple[Path, ...]] = (Path("/kaggle/input"),)
_TEMPLATE_CHOICES: Final[tuple[TemplateId, ...]] = (
    TemplateId.T1,
    TemplateId.T2,
    TemplateId.T3,
)
_TEMPLATE_FAMILY_CHOICES: Final[tuple[TemplateFamily, ...]] = (
    TemplateFamily.CANONICAL,
    TemplateFamily.OBSERVATION_LOG,
    TemplateFamily.CASE_LEDGER,
)
_TRANSITION_CHOICES: Final[tuple[Transition, ...]] = (
    Transition.R_STD_TO_R_INV,
    Transition.R_INV_TO_R_STD,
)
_DIFFICULTY_ORDER: Final[tuple[Difficulty, ...]] = (
    Difficulty.EASY,
    Difficulty.MEDIUM,
    Difficulty.HARD,
)


def _is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value)


def _sample_pairs(
    rng: random.Random,
    total_items: int,
) -> tuple[tuple[int, int], ...]:
    return tuple(rng.sample(CASE_SPACE, k=total_items))


def _build_items(
    sampled_pairs: tuple[tuple[int, int], ...],
    pre_count: int,
    rule_a: RuleName,
    rule_b: RuleName,
) -> tuple[EpisodeItem, ...]:
    items: list[EpisodeItem] = []
    for position, (q1, q2) in enumerate(sampled_pairs, start=1):
        if position <= pre_count:
            active_rule = rule_a
            phase = Phase.PRE
            kind = ItemKind.LABELED
            item_label = label(active_rule, q1, q2)
        elif position <= LABELED_ITEM_COUNT:
            active_rule = rule_b
            phase = Phase.POST
            kind = ItemKind.LABELED
            item_label = label(active_rule, q1, q2)
        else:
            phase = Phase.POST
            kind = ItemKind.PROBE
            item_label = None

        items.append(
            EpisodeItem(
                position=position,
                phase=phase,
                kind=kind,
                q1=q1,
                q2=q2,
                label=item_label,
            )
        )
    return tuple(items)


def _has_full_probe_sign_pattern_coverage(items: tuple[EpisodeItem, ...]) -> bool:
    return build_probe_sign_pattern_counts(items[LABELED_ITEM_COUNT:]) == (
        ("++", 1),
        ("--", 1),
        ("+-", 1),
        ("-+", 1),
    )


def _is_valid_candidate(
    contradiction_count_post: int,
    items: tuple[EpisodeItem, ...],
    pre_count: int,
    rule_a: RuleName,
    rule_b: RuleName,
    probe_targets: tuple[InteractionLabel, ...],
) -> bool:
    probe_items = items[LABELED_ITEM_COUNT:]
    updated_sign_patterns = build_updated_sign_patterns(items[pre_count:LABELED_ITEM_COUNT])
    return (
        contradiction_count_post >= 1
        and has_mixed_polarity_sign_patterns(updated_sign_patterns)
        and _has_full_probe_sign_pattern_coverage(items)
        and has_both_probe_labels(probe_targets)
        and not is_global_rule_probe_block(probe_items, probe_targets, rule_a)
        and not is_global_rule_probe_block(probe_items, probe_targets, rule_b)
    )


def _target_difficulty_for_seed(seed: int) -> Difficulty:
    return _DIFFICULTY_ORDER[seed % len(_DIFFICULTY_ORDER)]


def _target_template_for_seed(seed: int) -> TemplateId:
    return _TEMPLATE_CHOICES[(seed // len(_DIFFICULTY_ORDER)) % len(_TEMPLATE_CHOICES)]


def _target_template_family_for_seed(seed: int) -> TemplateFamily:
    stride = len(_DIFFICULTY_ORDER) * len(_TEMPLATE_CHOICES)
    return _TEMPLATE_FAMILY_CHOICES[
        (seed // stride) % len(_TEMPLATE_FAMILY_CHOICES)
    ]


def _target_transition_for_seed(seed: int) -> Transition:
    stride = len(_DIFFICULTY_ORDER) * len(_TEMPLATE_CHOICES) * len(
        _TEMPLATE_FAMILY_CHOICES
    )
    return _TRANSITION_CHOICES[(seed // stride) % len(_TRANSITION_CHOICES)]


def _manifest_dir(repo_root: Path | str | None = None) -> Path:
    if repo_root is None:
        return _DEFAULT_MANIFEST_DIR
    return Path(repo_root).resolve() / "src" / "frozen_splits"


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

    def __post_init__(self) -> None:
        if self.partition not in PARTITIONS:
            raise ValueError(f"unknown partition: {self.partition}")

        object.__setattr__(self, "episode_split", parse_split(self.episode_split))
        expected_split = _PARTITION_TO_EPISODE_SPLIT[self.partition]
        if self.episode_split is not expected_split:
            raise ValueError("episode_split does not match the canonical partition mapping")

        if self.manifest_version != MANIFEST_VERSION:
            raise ValueError(f"manifest_version must equal {MANIFEST_VERSION}")
        if not _is_nonempty_string(self.seed_bank_version):
            raise ValueError("seed_bank_version must be a non-empty string")
        if self.spec_version != SPEC_VERSION:
            raise ValueError(f"spec_version must equal {SPEC_VERSION}")
        if self.generator_version != GENERATOR_VERSION:
            raise ValueError(f"generator_version must equal {GENERATOR_VERSION}")
        if self.template_set_version != TEMPLATE_SET_VERSION:
            raise ValueError(f"template_set_version must equal {TEMPLATE_SET_VERSION}")
        if self.difficulty_version != DIFFICULTY_VERSION:
            raise ValueError(f"difficulty_version must equal {DIFFICULTY_VERSION}")

        normalized_seeds = tuple(self.seeds)
        if not normalized_seeds:
            raise ValueError("seeds must not be empty")
        if any(not _is_plain_int(seed) for seed in normalized_seeds):
            raise TypeError("seeds must contain only int values")
        if len(set(normalized_seeds)) != len(normalized_seeds):
            raise ValueError("seeds must contain unique values")
        object.__setattr__(self, "seeds", normalized_seeds)


@dataclass(frozen=True, slots=True)
class FrozenSplitEpisode:
    partition: str
    seed: int
    manifest_version: str
    seed_bank_version: str
    episode: Episode

    def __post_init__(self) -> None:
        if self.partition not in PARTITIONS:
            raise ValueError(f"unknown partition: {self.partition}")
        if not _is_plain_int(self.seed):
            raise TypeError("seed must be an int")
        if self.manifest_version != MANIFEST_VERSION:
            raise ValueError(f"manifest_version must equal {MANIFEST_VERSION}")
        if not _is_nonempty_string(self.seed_bank_version):
            raise ValueError("seed_bank_version must be a non-empty string")
        if not isinstance(self.episode, Episode):
            raise TypeError("episode must be an Episode")
        expected_split = _PARTITION_TO_EPISODE_SPLIT[self.partition]
        if self.episode.split is not expected_split:
            raise ValueError("episode split does not match partition mapping")


def load_split_manifest(
    partition: str,
    repo_root: Path | str | None = None,
    *,
    private_dataset_root: Path | str | None = None,
) -> FrozenSplitManifest:
    if partition not in PARTITIONS:
        raise ValueError(f"unknown partition: {partition}")
    if partition == "private_leaderboard":
        return _load_private_split_manifest(private_dataset_root)

    payload = json.loads(
        (_manifest_dir(repo_root) / f"{partition}.json").read_text(encoding="utf-8")
    )
    actual_fields = tuple(payload)
    if actual_fields != _MANIFEST_FIELD_ORDER:
        raise ValueError(
            "manifest fields must exactly match the canonical order: "
            + ", ".join(_MANIFEST_FIELD_ORDER)
        )

    return FrozenSplitManifest(
        partition=payload["partition"],
        episode_split=payload["episode_split"],
        manifest_version=payload["manifest_version"],
        seed_bank_version=payload["seed_bank_version"],
        spec_version=payload["spec_version"],
        generator_version=payload["generator_version"],
        template_set_version=payload["template_set_version"],
        difficulty_version=payload["difficulty_version"],
        seeds=tuple(payload["seeds"]),
    )


def generate_episode(
    seed: int,
    split: Split | str = Split.PUBLIC,
    *,
    max_attempts: int = DEFAULT_GENERATION_MAX_ATTEMPTS,
) -> Episode:
    if not _is_plain_int(seed):
        raise TypeError("seed must be an int")
    if not _is_plain_int(max_attempts):
        raise TypeError("max_attempts must be an int")
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    rng = random.Random(seed)
    target_difficulty = _target_difficulty_for_seed(seed)
    target_template_id = _target_template_for_seed(seed)
    target_template_family = _target_template_family_for_seed(seed)
    target_transition = _target_transition_for_seed(seed)
    rule_a = (
        RuleName.R_STD
        if target_transition is Transition.R_STD_TO_R_INV
        else RuleName.R_INV
    )
    rule_b = rule_a.opposite
    template = TEMPLATES[target_template_id]
    resolved_split = parse_split(split)

    for _attempt in range(max_attempts):
        sampled_pairs = _sample_pairs(rng, template.total_items)
        items = _build_items(sampled_pairs, template.pre_count, rule_a, rule_b)
        probe_targets = build_effective_probe_targets(
            items[LABELED_ITEM_COUNT:],
            rule_a,
            rule_b,
            build_updated_sign_patterns(items[template.pre_count:LABELED_ITEM_COUNT]),
        )
        contradiction_count_post = build_contradiction_count_post(
            items[:LABELED_ITEM_COUNT],
            template.pre_count,
            rule_a,
            rule_b,
        )
        if _is_valid_candidate(
            contradiction_count_post,
            items,
            template.pre_count,
            rule_a,
            rule_b,
            probe_targets,
        ):
            difficulty_factors = derive_difficulty_factors(items, template.pre_count)
            difficulty, difficulty_profile_id = derive_difficulty_profile(
                difficulty_factors
            )
            if difficulty is target_difficulty:
                break
    else:
        raise RuntimeError(
            "generate_episode exhausted max_attempts "
            f"for seed={seed}, split={resolved_split}, max_attempts={max_attempts}"
        )

    return Episode(
        episode_id=f"ife-r13-{seed}",
        split=resolved_split,
        difficulty=difficulty,
        template_id=target_template_id,
        template_family=target_template_family,
        rule_A=rule_a,
        rule_B=rule_b,
        transition=Transition.from_rules(rule_a, rule_b),
        pre_count=template.pre_count,
        post_labeled_count=template.post_labeled_count,
        shift_after_position=template.shift_after_position,
        contradiction_count_post=contradiction_count_post,
        difficulty_profile_id=difficulty_profile_id,
        difficulty_factors=difficulty_factors,
        items=items,
        probe_targets=probe_targets,
        probe_label_counts=build_probe_label_counts(probe_targets),
        probe_sign_pattern_counts=build_probe_sign_pattern_counts(
            items[LABELED_ITEM_COUNT:]
        ),
        probe_metadata=build_probe_metadata(
            items[LABELED_ITEM_COUNT:],
            rule_a,
            rule_b,
        ),
        difficulty_version=DIFFICULTY_VERSION,
    )


def generate_frozen_split(manifest: FrozenSplitManifest) -> tuple[FrozenSplitEpisode, ...]:
    if manifest.partition == "private_leaderboard":
        raise ValueError(
            "private_leaderboard must be loaded from the authorized private artifact; "
            "runtime regeneration is disabled"
        )
    return tuple(
        FrozenSplitEpisode(
            partition=manifest.partition,
            seed=seed,
            manifest_version=manifest.manifest_version,
            seed_bank_version=manifest.seed_bank_version,
            episode=generate_episode(seed, split=manifest.episode_split),
        )
        for seed in manifest.seeds
    )


def load_frozen_split(
    partition: str,
    *,
    private_dataset_root: Path | str | None = None,
) -> tuple[FrozenSplitEpisode, ...]:
    if partition == "private_leaderboard":
        return load_private_split(private_dataset_root)
    return generate_frozen_split(
        load_split_manifest(partition, private_dataset_root=private_dataset_root)
    )


def load_private_split(
    private_dataset_root: Path | str | None = None,
) -> tuple[FrozenSplitEpisode, ...]:
    payload = json.loads(
        _resolve_private_episodes_path(private_dataset_root).read_text(encoding="utf-8")
    )
    return _parse_private_episodes(payload)


def discover_private_dataset_root(
    private_dataset_root: Path | str | None = None,
) -> Path | None:
    if private_dataset_root is not None:
        return _validate_private_dataset_root(
            Path(private_dataset_root),
            context="explicit private_dataset_root",
        )

    env_value = os.environ.get(PRIVATE_DATASET_ROOT_ENV_VAR)
    if env_value:
        return _validate_private_dataset_root(
            Path(env_value),
            context=f"{PRIVATE_DATASET_ROOT_ENV_VAR}={env_value}",
        )

    for search_root in _KAGGLE_PRIVATE_SEARCH_ROOTS:
        if not search_root.exists():
            continue
        for episodes_path in search_root.rglob(PRIVATE_EPISODES_FILENAME):
            return episodes_path.parent

    return None


def resolve_private_dataset_root(
    private_dataset_root: Path | str | None = None,
) -> Path:
    resolved_root = discover_private_dataset_root(private_dataset_root)
    if resolved_root is not None:
        return resolved_root

    raise FileNotFoundError(
        "Private evaluation dataset is not attached. "
        "Attach the authorized private dataset mount or set "
        f"{PRIVATE_DATASET_ROOT_ENV_VAR} to the mounted dataset root."
    )


def _load_private_split_manifest(
    private_dataset_root: Path | str | None = None,
) -> FrozenSplitManifest:
    private_root = resolve_private_dataset_root(private_dataset_root)
    payload = json.loads(
        (private_root / PRIVATE_EPISODES_FILENAME).read_text(encoding="utf-8")
    )
    records = load_private_split(private_root)
    return FrozenSplitManifest(
        partition="private_leaderboard",
        episode_split=payload["episode_split"],
        manifest_version=payload["benchmark_version"],
        seed_bank_version=payload["artifact_checksum"],
        spec_version=SPEC_VERSION,
        generator_version=GENERATOR_VERSION,
        template_set_version=TEMPLATE_SET_VERSION,
        difficulty_version=DIFFICULTY_VERSION,
        seeds=tuple(record.seed for record in records),
    )


def _resolve_private_episodes_path(
    private_dataset_root: Path | str | None,
) -> Path:
    return resolve_private_dataset_root(private_dataset_root) / PRIVATE_EPISODES_FILENAME


def _validate_private_dataset_root(root: Path, *, context: str) -> Path:
    candidate = root / PRIVATE_EPISODES_FILENAME
    if not candidate.is_file():
        raise FileNotFoundError(
            f"private_episodes.json not found for {context} at {candidate}. "
            "Attach the authorized private dataset mount before running private evaluation."
        )
    return root


def _parse_private_episodes(payload: object) -> tuple[FrozenSplitEpisode, ...]:
    if not isinstance(payload, dict):
        raise ValueError("private_episodes.json must contain a JSON object")

    partition = payload.get("partition")
    if partition != _EXPECTED_PRIVATE_PARTITION:
        raise ValueError(
            f"partition must equal {_EXPECTED_PRIVATE_PARTITION!r}, got {partition!r}"
        )

    episode_split = payload.get("episode_split")
    if episode_split != _EXPECTED_PRIVATE_EPISODE_SPLIT:
        raise ValueError(
            "episode_split must equal "
            f"{_EXPECTED_PRIVATE_EPISODE_SPLIT!r}, got {episode_split!r}"
        )

    benchmark_version = payload.get("benchmark_version")
    if benchmark_version != MANIFEST_VERSION:
        raise ValueError(
            f"benchmark_version must equal {MANIFEST_VERSION!r}, got {benchmark_version!r}"
        )

    schema_version = payload.get("schema_version")
    if schema_version != PRIVATE_SPLIT_ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            "schema_version must equal "
            f"{PRIVATE_SPLIT_ARTIFACT_SCHEMA_VERSION!r}, got {schema_version!r}"
        )

    artifact_checksum = payload.get("artifact_checksum")
    if not isinstance(artifact_checksum, str) or not artifact_checksum:
        raise ValueError("artifact_checksum must be a non-empty string")

    episodes_raw = payload.get("episodes")
    if not isinstance(episodes_raw, list) or not episodes_raw:
        raise ValueError("episodes must be a non-empty list")

    checksum_payload = {
        "partition": partition,
        "episode_split": episode_split,
        "benchmark_version": benchmark_version,
        "schema_version": schema_version,
        "episodes": episodes_raw,
    }
    if artifact_checksum != _compute_private_artifact_checksum(checksum_payload):
        raise ValueError("artifact_checksum does not match the private artifact payload")

    return tuple(
        _parse_private_episode_row(row, benchmark_version, artifact_checksum)
        for row in episodes_raw
    )


def _compute_private_artifact_checksum(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_private_episode_row(
    row: object,
    manifest_version: str,
    seed_bank_version: str,
) -> FrozenSplitEpisode:
    if not isinstance(row, dict):
        raise ValueError("each episode row must be a JSON object")

    seed = row.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("episode row 'seed' must be an int")

    episode_payload = row.get("episode")
    if not isinstance(episode_payload, dict):
        raise ValueError("episode row 'episode' must be a JSON object")

    return FrozenSplitEpisode(
        partition=_EXPECTED_PRIVATE_PARTITION,
        seed=seed,
        manifest_version=manifest_version,
        seed_bank_version=seed_bank_version,
        episode=_build_episode(episode_payload),
    )


def _build_episode(ep: dict[str, object]) -> Episode:
    items = tuple(
        EpisodeItem(
            position=item["position"],
            phase=item["phase"],
            kind=item["kind"],
            q1=item["q1"],
            q2=item["q2"],
            label=item.get("label"),
        )
        for item in ep["items"]
    )
    probe_targets = tuple(ep["probe_targets"])
    probe_label_counts = tuple((pair[0], pair[1]) for pair in ep["probe_label_counts"])
    probe_sign_pattern_counts = tuple(
        (pair[0], pair[1]) for pair in ep["probe_sign_pattern_counts"]
    )
    probe_metadata = tuple(
        ProbeMetadata(
            position=pm["position"],
            is_disagreement_probe=pm["is_disagreement_probe"],
            old_rule_label=pm["old_rule_label"],
            new_rule_label=pm["new_rule_label"],
        )
        for pm in ep["probe_metadata"]
    )
    pre_count = ep["pre_count"]
    difficulty_factors = derive_difficulty_factors(items, pre_count)
    raw_difficulty_factors = ep.get("difficulty_factors", difficulty_factors)
    if isinstance(raw_difficulty_factors, dict):
        raw_difficulty_factors = DifficultyFactors(**raw_difficulty_factors)
    derived_difficulty, difficulty_profile_id = derive_difficulty_profile(
        difficulty_factors
    )
    return Episode(
        episode_id=ep["episode_id"],
        split=ep["split"],
        difficulty=ep.get("difficulty", derived_difficulty.value),
        template_id=ep["template_id"],
        template_family=ep.get("template_family", "canonical"),
        rule_A=ep["rule_A"],
        rule_B=ep["rule_B"],
        transition=ep["transition"],
        pre_count=pre_count,
        post_labeled_count=ep["post_labeled_count"],
        shift_after_position=ep["shift_after_position"],
        contradiction_count_post=ep["contradiction_count_post"],
        difficulty_profile_id=ep.get(
            "difficulty_profile_id",
            difficulty_profile_id.value,
        ),
        difficulty_factors=raw_difficulty_factors,
        items=items,
        probe_targets=probe_targets,
        probe_label_counts=probe_label_counts,
        probe_sign_pattern_counts=probe_sign_pattern_counts,
        probe_metadata=probe_metadata,
        difficulty_version=ep.get("difficulty_version", DIFFICULTY_VERSION),
        spec_version=ep.get("spec_version", SPEC_VERSION),
        generator_version=ep.get("generator_version", GENERATOR_VERSION),
        template_set_version=ep.get("template_set_version", TEMPLATE_SET_VERSION),
    )


# ---------------------------------------------------------------------------
# Prompt rendering & bundle building (merged from benchmark_bundle.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _BinaryPresentation:
    intro: str
    labeled_heading: str
    probe_heading: str
    outro: str
    line_renderer: Callable[[EpisodeItem], str]


def _render_binary_line(item: EpisodeItem) -> str:
    return (
        f"{item.position}. r1={_format_marker_value(item.q1)}, "
        f"r2={_format_marker_value(item.q2)} -> {_render_outcome(item)}"
    )


def _render_binary_log_line(item: EpisodeItem) -> str:
    return (
        f"[{item.position:02d}] r1={_format_marker_value(item.q1)} | "
        f"r2={_format_marker_value(item.q2)} | observed={_render_outcome(item)}"
    )


def _render_binary_ledger_line(item: EpisodeItem) -> str:
    return (
        f"row {item.position:02d} | r1={_format_marker_value(item.q1)} | "
        f"r2={_format_marker_value(item.q2)} | state={_render_outcome(item)}"
    )


def _render_outcome(item: EpisodeItem) -> str:
    return format_public_state(item.label) if item.label is not None else "?"


def _format_marker_value(marker_value: int) -> str:
    return f"{marker_value:+d}"


_BINARY_OUTRO: Final[str] = (
    "Return exactly 4 outputs in order, one per probe. "
    "Use only type_a or type_b. Map zark to type_a and blim to type_b."
)

_BINARY_PRESENTATIONS: Final[dict[TemplateFamily, _BinaryPresentation]] = {
    TemplateFamily.CANONICAL: _BinaryPresentation(
        intro=(
            "You are given labeled records for two markers.\n"
            "Each labeled line shows r1, r2, and the observed state.\n"
            "Use the full sequence to infer which sign combinations were revised by the later evidence, "
            "then answer the final unlabeled cases."
        ),
        labeled_heading="Labeled examples:",
        probe_heading="Probes:",
        outro=_BINARY_OUTRO,
        line_renderer=_render_binary_line,
    ),
    TemplateFamily.OBSERVATION_LOG: _BinaryPresentation(
        intro=(
            "Review the observation log for two markers.\n"
            "Each entry records r1, r2, and the observed state.\n"
            "Use the full log to infer which sign combinations were revised later, then answer the unlabeled probe entries."
        ),
        labeled_heading="Resolved log entries:",
        probe_heading="Unresolved probe entries:",
        outro=_BINARY_OUTRO,
        line_renderer=_render_binary_log_line,
    ),
    TemplateFamily.CASE_LEDGER: _BinaryPresentation(
        intro=(
            "Review the case ledger for two markers.\n"
            "Each row records r1, r2, and the observed state.\n"
            "Use the full ledger to infer which sign combinations were revised by the later evidence, "
            "then complete the pending rows."
        ),
        labeled_heading="Confirmed ledger rows:",
        probe_heading="Pending ledger rows:",
        outro=_BINARY_OUTRO,
        line_renderer=_render_binary_ledger_line,
    ),
}


def render_binary_prompt(episode: Episode) -> str:
    labeled_items = episode.items[:LABELED_ITEM_COUNT]
    probe_items = episode.items[LABELED_ITEM_COUNT:]
    presentation = _BINARY_PRESENTATIONS[episode.template_family]
    return "\n".join(
        (
            presentation.intro,
            "",
            presentation.labeled_heading,
            *(presentation.line_renderer(item) for item in labeled_items),
            "",
            presentation.probe_heading,
            *(presentation.line_renderer(item) for item in probe_items),
            "",
            presentation.outro,
        )
    )


def build_benchmark_bundle(
    *,
    include_private: bool = True,
    private_dataset_root: Path | str | None = None,
) -> dict[str, object]:
    partitions = [_build_partition_bundle("public_leaderboard")]

    if include_private:
        resolved_private_root = _resolve_optional_private_dataset_root(private_dataset_root)
        if resolved_private_root is not None:
            partitions.append(
                _build_partition_bundle(
                    "private_leaderboard",
                    private_dataset_root=resolved_private_root,
                )
            )

    return {
        "task": TASK_NAME,
        "benchmark_version": MANIFEST_VERSION,
        "partitions": partitions,
    }


def _build_partition_bundle(
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
                "seed": record.seed,
                "episode_id": record.episode.episode_id,
                "prompt_binary": render_binary_prompt(record.episode),
                "probe_targets": [
                    format_public_label(label) for label in record.episode.probe_targets
                ],
            }
            for record in records
        ],
    }


def build_leaderboard_rows(bundle: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for partition in bundle["partitions"]:
        split_name = partition["partition"]
        for episode in partition["episodes"]:
            rows.append(
                {
                    "episode_id": episode["episode_id"],
                    "split": split_name,
                    "prompt_binary": episode["prompt_binary"],
                    "probe_targets": tuple(episode["probe_targets"]),
                }
            )
    return rows


def _resolve_optional_private_dataset_root(
    private_dataset_root: Path | str | None,
) -> Path | None:
    if private_dataset_root is not None:
        return resolve_private_dataset_root(private_dataset_root)
    return discover_private_dataset_root()
