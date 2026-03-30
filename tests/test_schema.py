from __future__ import annotations

from tasks.ruleshift_benchmark.generator import generate_episode
from tasks.ruleshift_benchmark.protocol import (
    Difficulty,
    DifficultyProfileId,
    FactorLevel,
)
from tasks.ruleshift_benchmark.schema import (
    DifficultyFactors,
    derive_difficulty_factors,
    derive_difficulty_profile,
)


def test_difficulty_derivation_remains_canonical_for_representative_episode():
    episode = generate_episode(2)

    factors = derive_difficulty_factors(episode.items, episode.pre_count)

    assert factors == DifficultyFactors(
        conflict_strength=FactorLevel.MEDIUM,
        post_shift_evidence_clarity=FactorLevel.MEDIUM,
        probe_ambiguity=FactorLevel.MEDIUM,
        evidence_to_final_probe_distance=FactorLevel.HIGH,
        pre_shift_distractor_pressure=FactorLevel.MEDIUM,
    )
    assert derive_difficulty_profile(factors) == (
        Difficulty.HARD,
        DifficultyProfileId.HARD_INTERLEAVED,
    )
