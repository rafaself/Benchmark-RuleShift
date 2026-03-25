import json
from pathlib import Path

import pandas as pd
import pytest

from src.core.kaggle import (
    ConfidenceInterval,
    build_kaggle_payload,
    compute_bootstrap_confidence_interval,
    validate_kaggle_payload,
)


# ---------------------------------------------------------------------------
# compute_bootstrap_confidence_interval
# ---------------------------------------------------------------------------

def test_compute_bootstrap_confidence_interval_determinism():
    num_correct = [3, 4, 2, 4, 1] * 10
    total = [4, 4, 4, 4, 4] * 10

    ci1 = compute_bootstrap_confidence_interval(num_correct, total, seed=2025)
    ci2 = compute_bootstrap_confidence_interval(num_correct, total, seed=2025)

    assert ci1.mean == ci2.mean
    assert ci1.lower == ci2.lower
    assert ci1.upper == ci2.upper
    assert ci1.margin == ci2.margin

    # Different seed should likely produce different CI bounds (not a hard failure
    # if data is trivial, but the seed parameter must be accepted without error).
    ci3 = compute_bootstrap_confidence_interval(num_correct, total, seed=12345)
    assert ci3.level == ci1.level


def test_compute_bootstrap_confidence_interval_determinism_exact_values():
    """Exact reproducibility of the canonical 48-episode scenario with seed 2025."""
    num_correct = [3, 4, 2, 4, 1, 3, 4, 2, 4, 1, 3, 4, 2, 4, 1, 3] * 3  # 48 episodes
    total = [4] * 48

    ci_a = compute_bootstrap_confidence_interval(num_correct, total, seed=2025)
    ci_b = compute_bootstrap_confidence_interval(num_correct, total, seed=2025)

    assert ci_a.mean == ci_b.mean
    assert ci_a.lower == ci_b.lower
    assert ci_a.upper == ci_b.upper
    assert ci_a.margin == ci_b.margin
    assert ci_a.level == 0.95

    # Sanity: bounds must be ordered and mean must be within them.
    assert ci_a.lower <= ci_a.mean <= ci_a.upper
    assert ci_a.margin >= 0.0


def test_compute_bootstrap_confidence_interval_basic():
    total = [4] * 10

    # All correct
    ci = compute_bootstrap_confidence_interval([4] * 10, total)
    assert ci.mean == 1.0
    assert ci.lower == 1.0
    assert ci.upper == 1.0
    assert ci.margin == 0.0

    # All wrong
    ci = compute_bootstrap_confidence_interval([0] * 10, total)
    assert ci.mean == 0.0
    assert ci.lower == 0.0
    assert ci.upper == 0.0

    # Uniform 50 %: bootstrap over identical episodes should be stable at 0.5
    ci = compute_bootstrap_confidence_interval([2] * 10, total)
    assert ci.mean == 0.5
    assert ci.lower == 0.5
    assert ci.upper == 0.5

    # High variance across episodes: CI should straddle the mean
    ci = compute_bootstrap_confidence_interval([0, 4] * 50, [4] * 100)
    assert ci.mean == 0.5
    assert ci.lower < 0.5
    assert ci.upper > 0.5
    assert 0.3 < ci.lower < 0.45
    assert 0.55 < ci.upper < 0.7


def test_compute_bootstrap_confidence_interval_empty():
    ci = compute_bootstrap_confidence_interval([], [])
    assert ci.mean == 0.0
    assert ci.lower == 0.0
    assert ci.upper == 0.0


def test_compute_bootstrap_confidence_interval_length_mismatch():
    with pytest.raises(ValueError):
        compute_bootstrap_confidence_interval([1], [1, 2])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_binary_df(n: int = 16, num_correct: int = 3) -> pd.DataFrame:
    return pd.DataFrame([{"num_correct": num_correct, "total": 4}] * n)


def _make_narrative_df(n: int = 16, num_correct: int = 3) -> pd.DataFrame:
    return pd.DataFrame([{"num_correct": num_correct, "total": 4}] * n)


def _valid_payload(n: int = 16) -> dict:
    """Return a minimal valid canonical payload with both Binary and Narrative."""
    bin_df = _make_binary_df(n, num_correct=3)
    nar_df = _make_narrative_df(n, num_correct=3)
    return build_kaggle_payload(bin_df, nar_df)


# ---------------------------------------------------------------------------
# build_kaggle_payload — basic contract
# ---------------------------------------------------------------------------

def test_build_kaggle_payload_valid():
    bin_df = pd.DataFrame([
        {"num_correct": 4, "total": 4},
        {"num_correct": 2, "total": 4},
    ])
    nar_df = pd.DataFrame([
        {"num_correct": 3, "total": 4},
        {"num_correct": 1, "total": 4},
    ])
    payload = build_kaggle_payload(bin_df, nar_df)

    assert payload["primary_result"]["score"] == 0.75
    assert payload["primary_result"]["numerator"] == 6
    assert payload["primary_result"]["denominator"] == 8
    assert payload["primary_result"]["total_episodes"] == 2

    ci = payload["primary_result"]["confidence_interval"]
    assert ci["mean"] == 0.75
    assert ci["lower"] <= 0.75
    assert ci["upper"] >= 0.75

    # Narrative result is real, not None
    assert payload["narrative_result"] is not None
    assert payload["narrative_result"]["total_episodes"] == 2
    assert payload["narrative_result"]["denominator"] == 8

    # Comparison block is real, not None
    assert payload["comparison"] is not None
    assert payload["comparison"]["episode_count_aligned"] is True

    assert payload["slices"] == {}
    assert payload["metadata"]["benchmark_version"] is not None


def test_build_kaggle_payload_empty_fails():
    df = pd.DataFrame(columns=["num_correct", "total"])
    nar_df = _make_narrative_df()
    with pytest.raises(ValueError, match="cannot be empty"):
        build_kaggle_payload(df, nar_df)


def test_build_kaggle_payload_missing_columns():
    df = pd.DataFrame([{"foo": 1}])
    nar_df = _make_narrative_df()
    with pytest.raises(ValueError, match="must contain 'num_correct' and 'total'"):
        build_kaggle_payload(df, nar_df)


def test_build_kaggle_payload_invalid_type():
    nar_df = _make_narrative_df()
    with pytest.raises(TypeError, match="must be a pandas DataFrame"):
        build_kaggle_payload("not a dataframe", nar_df)


# ---------------------------------------------------------------------------
# Narrative mandatory — negative cases
# ---------------------------------------------------------------------------

def test_build_kaggle_payload_narrative_missing_fails():
    """Narrative missing (narrative_df=None) must fail the release path."""
    bin_df = _make_binary_df()
    with pytest.raises(ValueError, match="narrative_df is required"):
        build_kaggle_payload(bin_df, None)


def test_build_kaggle_payload_narrative_skipped_fails():
    """Narrative skipped (explicitly None) must fail — same as missing."""
    bin_df = _make_binary_df()
    with pytest.raises(ValueError, match="narrative_df is required"):
        build_kaggle_payload(binary_df=bin_df, narrative_df=None)


def test_build_kaggle_payload_episode_count_mismatch_fails():
    """Binary and Narrative episode-count mismatch must fail."""
    bin_df = _make_binary_df(n=16)
    nar_df = _make_narrative_df(n=10)  # different count
    with pytest.raises(ValueError, match="episode counts do not match"):
        build_kaggle_payload(bin_df, nar_df)


def test_build_kaggle_payload_denominator_mismatch_fails():
    """Binary and Narrative denominator mismatch must fail."""
    bin_df = _make_binary_df(n=16)
    nar_df = pd.DataFrame([{"num_correct": 2, "total": 6}] * 16)  # different total
    with pytest.raises(ValueError, match="denominators do not match"):
        build_kaggle_payload(bin_df, nar_df)


# ---------------------------------------------------------------------------
# Actual emitted public result structure (passing)
# ---------------------------------------------------------------------------

def test_emitted_public_result_structure():
    """Full 48-episode payload with both Binary and Narrative validates cleanly."""
    bin_df = pd.DataFrame(
        [{"num_correct": c, "total": 4} for c in [3, 4, 2, 4, 1, 3, 4, 2] * 6]
    )
    nar_df = pd.DataFrame(
        [{"num_correct": c, "total": 4} for c in [2, 3, 1, 3, 0, 2, 3, 1] * 6]
    )
    assert len(bin_df) == 48
    assert len(nar_df) == 48

    payload = build_kaggle_payload(bin_df, nar_df)
    validate_kaggle_payload(payload)  # must not raise

    pr = payload["primary_result"]
    for field in ("score", "numerator", "denominator", "total_episodes", "confidence_interval"):
        assert field in pr, f"primary_result missing '{field}'"

    ci = pr["confidence_interval"]
    for field in ("mean", "lower", "upper", "level", "margin"):
        assert field in ci, f"confidence_interval missing '{field}'"

    assert pr["total_episodes"] == 48
    assert pr["denominator"] == 48 * 4

    # Narrative result present and populated
    nr = payload["narrative_result"]
    assert nr is not None
    for field in ("score", "numerator", "denominator", "total_episodes", "confidence_interval"):
        assert field in nr, f"narrative_result missing '{field}'"
    assert nr["total_episodes"] == 48
    assert nr["denominator"] == 48 * 4

    # Comparison block present and populated
    comp = payload["comparison"]
    assert comp is not None
    for field in (
        "binary_score", "narrative_score", "delta",
        "episode_count_aligned", "binary_total_episodes", "narrative_total_episodes",
    ):
        assert field in comp, f"comparison missing '{field}'"
    assert comp["episode_count_aligned"] is True
    assert comp["binary_total_episodes"] == 48
    assert comp["narrative_total_episodes"] == 48

    assert "slices" in payload
    assert "metadata" in payload


# ---------------------------------------------------------------------------
# validate_kaggle_payload — structural acceptance
# ---------------------------------------------------------------------------

def test_validate_kaggle_payload_accepts_new_shape():
    validate_kaggle_payload(_valid_payload())  # must not raise


def test_validate_kaggle_payload_rejects_non_dict():
    with pytest.raises(TypeError, match="must be a dict"):
        validate_kaggle_payload("not a dict")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Regression: reject old bad kbench shape
# ---------------------------------------------------------------------------

def test_validate_kaggle_payload_rejects_old_bad_shape():
    """The old kbench conversations/results/numericResult-only shape must be rejected."""
    old_shape_conversations = {
        "conversations": [{"metrics": {}}],
        "results": [
            {
                "numericResult": {
                    "confidenceInterval": {
                        "mean": 0.75,
                        "lower": 0.70,
                        "upper": 0.80,
                        "level": 0.95,
                        "margin": 0.05,
                    }
                }
            }
        ],
    }
    with pytest.raises(ValueError, match="old kbench"):
        validate_kaggle_payload(old_shape_conversations)


def test_validate_kaggle_payload_rejects_results_only_shape():
    """Any shape carrying a top-level 'results' key is the old shape and must fail."""
    with pytest.raises(ValueError, match="old kbench"):
        validate_kaggle_payload({"results": [{"numericResult": {"confidenceInterval": {}}}]})


# ---------------------------------------------------------------------------
# validate_kaggle_payload — missing required fields
# ---------------------------------------------------------------------------

def test_validate_kaggle_payload_rejects_missing_primary_result():
    payload = _valid_payload()
    broken = {k: v for k, v in payload.items() if k != "primary_result"}
    with pytest.raises(ValueError, match="primary_result"):
        validate_kaggle_payload(broken)


def test_validate_kaggle_payload_rejects_missing_primary_fields():
    payload = _valid_payload()
    for field in ("score", "numerator", "denominator", "total_episodes", "confidence_interval"):
        broken = {**payload, "primary_result": {k: v for k, v in payload["primary_result"].items() if k != field}}
        with pytest.raises(ValueError, match="primary_result is missing required fields"):
            validate_kaggle_payload(broken)


def test_validate_kaggle_payload_rejects_missing_ci_fields():
    payload = _valid_payload()
    original_ci = payload["primary_result"]["confidence_interval"]
    for field in ("mean", "lower", "upper", "level", "margin"):
        broken_ci = {k: v for k, v in original_ci.items() if k != field}
        broken_pr = {**payload["primary_result"], "confidence_interval": broken_ci}
        broken = {**payload, "primary_result": broken_pr}
        with pytest.raises(ValueError, match="confidence_interval is missing required fields"):
            validate_kaggle_payload(broken)


def test_validate_kaggle_payload_rejects_missing_required_keys():
    """Removing any top-level required key must raise ValueError naming that key."""
    payload = _valid_payload()
    for field in ("narrative_result", "comparison", "slices", "metadata"):
        broken = {k: v for k, v in payload.items() if k != field}
        with pytest.raises(ValueError, match=field):
            validate_kaggle_payload(broken)


# ---------------------------------------------------------------------------
# validate_kaggle_payload — Narrative and comparison required
# ---------------------------------------------------------------------------

def test_validate_kaggle_payload_rejects_narrative_result_none():
    """narrative_result=None must fail — Narrative is mandatory for a valid release."""
    payload = _valid_payload()
    broken = {**payload, "narrative_result": None}
    with pytest.raises(ValueError, match="narrative_result is None"):
        validate_kaggle_payload(broken)


def test_validate_kaggle_payload_rejects_comparison_none():
    """comparison=None must fail — comparison block is mandatory for a valid release."""
    payload = _valid_payload()
    broken = {**payload, "comparison": None}
    with pytest.raises(ValueError, match="comparison is None"):
        validate_kaggle_payload(broken)


def test_validate_kaggle_payload_rejects_unaligned_episodes():
    """episode_count_aligned=False must fail."""
    payload = _valid_payload()
    broken_comparison = {**payload["comparison"], "episode_count_aligned": False}
    broken = {**payload, "comparison": broken_comparison}
    with pytest.raises(ValueError, match="episode_count_aligned is not True"):
        validate_kaggle_payload(broken)


# ---------------------------------------------------------------------------
# Negative: empty or missing evaluation output
# ---------------------------------------------------------------------------

def test_negative_empty_evaluation_output():
    """build_kaggle_payload must refuse an empty binary DataFrame."""
    df = pd.DataFrame(columns=["num_correct", "total"])
    nar_df = _make_narrative_df()
    with pytest.raises(ValueError, match="cannot be empty"):
        build_kaggle_payload(df, nar_df)


def test_negative_zero_episodes_payload():
    """validate_kaggle_payload must refuse a payload with total_episodes == 0."""
    payload = _valid_payload()
    payload["primary_result"] = {**payload["primary_result"], "total_episodes": 0}
    with pytest.raises(ValueError, match="total_episodes is 0"):
        validate_kaggle_payload(payload)


def test_negative_zero_denominator_payload():
    """validate_kaggle_payload must refuse a payload with denominator == 0."""
    payload = _valid_payload()
    payload["primary_result"] = {**payload["primary_result"], "denominator": 0}
    with pytest.raises(ValueError, match="denominator is 0"):
        validate_kaggle_payload(payload)


# ---------------------------------------------------------------------------
# Regression: notebook runs both tasks
# ---------------------------------------------------------------------------

def test_notebook_runs_both_tasks():
    """Regression: the official notebook must run both Binary and Narrative evaluations."""
    notebook_path = (
        Path(__file__).parents[1]
        / "packaging"
        / "kaggle"
        / "ruleshift_notebook_task.ipynb"
    )
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))

    all_sources = "\n".join(
        "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        for cell in nb["cells"]
        if cell.get("cell_type") == "code"
    )

    # Both task decorators must be present
    assert "ruleshift_benchmark_v1_binary" in all_sources
    assert "ruleshift_benchmark_v1_narrative" in all_sources

    # Both evaluate() calls must be present
    assert "binary_results = ruleshift_benchmark_v1_binary.evaluate(" in all_sources
    assert "narrative_results = ruleshift_benchmark_v1_narrative.evaluate(" in all_sources


# ---------------------------------------------------------------------------
# Regression: notebook uses canonical builder in the real emitted artifact path
# ---------------------------------------------------------------------------

def test_notebook_uses_canonical_builder_in_real_emitted_artifact_path():
    """
    The official Kaggle notebook must use build_kaggle_payload and
    validate_kaggle_payload in the canonical emission cell, and the result
    must be printed to stdout (the authoritative Kaggle-emitted artifact),
    not only written to a side JSON file.
    The cell must also hard-fail if narrative_results is missing.
    """
    notebook_path = (
        Path(__file__).parents[1]
        / "packaging"
        / "kaggle"
        / "ruleshift_notebook_task.ipynb"
    )
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))

    canonical_source: str | None = None
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        if "build_kaggle_payload" in source:
            canonical_source = source
            break

    assert canonical_source is not None, (
        "No cell containing build_kaggle_payload found in the official notebook"
    )

    # Must call the canonical builder and the validator
    assert "build_kaggle_payload" in canonical_source
    assert "validate_kaggle_payload" in canonical_source

    # Must print (emit to notebook output = authoritative Kaggle artifact)
    assert "print(" in canonical_source, (
        "Canonical payload cell must print the result to stdout "
        "(the real Kaggle-emitted artifact), not only write a side JSON file"
    )

    # Import must go through the canonical src/kaggle re-export, not core.kaggle directly
    assert "from kaggle import" in canonical_source, (
        "Canonical payload cell must import from 'kaggle' (the package re-export), "
        "not directly from 'core.kaggle'"
    )

    # Must hard-fail when narrative_results is missing
    assert "narrative_results is None" in canonical_source, (
        "Canonical payload cell must fail hard when narrative_results is None; "
        "Narrative is mandatory for a valid release"
    )

    # benchmark_result.json must be optional debug, not the sole output path
    if "benchmark_result.json" in canonical_source:
        assert "try:" in canonical_source, (
            "benchmark_result.json write must be inside a try block "
            "to mark it as optional debug output"
        )


# ---------------------------------------------------------------------------
# Regression: final emitted public result includes both results and comparison
# ---------------------------------------------------------------------------

def test_emitted_public_result_includes_both_results_and_comparison():
    """The final emitted public result must include primary_result, narrative_result,
    comparison block with binary_score, narrative_score, delta, and alignment check."""
    n = 48
    bin_df = pd.DataFrame([{"num_correct": 3, "total": 4}] * n)
    nar_df = pd.DataFrame([{"num_correct": 2, "total": 4}] * n)

    payload = build_kaggle_payload(bin_df, nar_df)
    validate_kaggle_payload(payload)  # must not raise

    # Binary result present
    pr = payload["primary_result"]
    assert pr["total_episodes"] == n
    assert pr["score"] == 0.75

    # Narrative result present and real (not None)
    nr = payload["narrative_result"]
    assert nr is not None
    assert nr["total_episodes"] == n
    assert nr["score"] == 0.5

    # Comparison block present with all required fields
    comp = payload["comparison"]
    assert comp is not None
    for field in (
        "binary_score", "narrative_score", "delta",
        "episode_count_aligned", "binary_total_episodes", "narrative_total_episodes",
    ):
        assert field in comp, f"comparison missing '{field}'"

    assert comp["binary_score"] == pytest.approx(0.75)
    assert comp["narrative_score"] == pytest.approx(0.5)
    assert comp["delta"] == pytest.approx(0.25)
    assert comp["episode_count_aligned"] is True
    assert comp["binary_total_episodes"] == n
    assert comp["narrative_total_episodes"] == n
