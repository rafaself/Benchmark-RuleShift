import pandas as pd
import pytest

from core.kaggle import (
    build_kaggle_payload,
    normalize_count_result_df,
    validate_kaggle_payload,
)
from core.kaggle.payload import compute_bootstrap_confidence_interval


def _make_binary_df(rows: list[dict] | None = None) -> pd.DataFrame:
    return pd.DataFrame(rows or [{"num_correct": 3, "total": 4}] * 4)


def test_compute_bootstrap_confidence_interval_is_deterministic_and_valid():
    num_correct = [3, 4, 2, 4, 1] * 10
    total = [4] * 50

    left = compute_bootstrap_confidence_interval(num_correct, total, seed=2025)
    right = compute_bootstrap_confidence_interval(num_correct, total, seed=2025)

    assert left == right
    assert left.lower <= left.mean <= left.upper
    assert left.margin >= 0.0


def test_normalize_count_result_df_supports_runtime_input_shapes():
    tuple_df = pd.DataFrame([{"result": (3, 4), "difficulty": "easy"}])
    legacy_df = pd.DataFrame([{"score_0": 3, "score_1": 4}])
    indexed_df = pd.DataFrame([{0: 3, 1: 4}])

    normalized_tuple = normalize_count_result_df(tuple_df)
    normalized_legacy = normalize_count_result_df(legacy_df)
    normalized_indexed = normalize_count_result_df(indexed_df)

    assert normalized_tuple.loc[0, "num_correct"] == 3
    assert normalized_tuple.loc[0, "total"] == 4
    assert normalized_tuple.loc[0, "difficulty"] == "easy"
    assert normalized_legacy.loc[0, "num_correct"] == 3
    assert normalized_legacy.loc[0, "total"] == 4
    assert normalized_indexed.loc[0, "num_correct"] == 3
    assert normalized_indexed.loc[0, "total"] == 4


def test_build_kaggle_payload_emits_canonical_contract():
    payload = build_kaggle_payload(
        _make_binary_df([{"num_correct": 3, "total": 4, "split": "public_leaderboard"}] * 4)
    )
    validate_kaggle_payload(payload)

    assert payload == {
        "score": payload["score"],
        "numerator": 12,
        "denominator": 16,
        "total_episodes": 4,
        "benchmark_version": "R14",
        "split": "public_leaderboard",
        "manifest_version": "R14",
    }


def test_build_kaggle_payload_rejects_dev_rows():
    with pytest.raises(ValueError, match="dev row"):
        build_kaggle_payload(_make_binary_df([{"num_correct": 3, "total": 4, "split": "dev"}]))


def test_validate_kaggle_payload_rejects_old_or_incomplete_shapes():
    payload = build_kaggle_payload(_make_binary_df())

    with pytest.raises(ValueError, match="old kbench"):
        validate_kaggle_payload({"results": [{"numericResult": {"confidenceInterval": {}}}]})

    with pytest.raises(ValueError, match="exactly these fields"):
        validate_kaggle_payload({**payload, "narrative_result": {}})

    with pytest.raises(ValueError, match="denominator is 0"):
        validate_kaggle_payload({**payload, "denominator": 0})
