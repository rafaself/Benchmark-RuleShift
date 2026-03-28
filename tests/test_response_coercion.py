"""Regression tests for LLM response type coercion.

These tests would have caught the 0% benchmark collapse caused by
parse_binary_response and parse_narrative_response not handling dict/mapping
responses from LLM SDKs.

Covers:
  - BinaryResponse dataclass (original happy path)
  - Plain dict with string values
  - Plain dict with Enum values
  - Mapping-like wrapper objects
  - Attribute-bearing wrapper objects (e.g. Pydantic-style)
  - Provider failure (None)
  - Invalid/partial dicts
  - Narrative text extraction from dict/wrapper
  - Score integration for each response type
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

from core.kaggle.types import (
    BinaryResponse,
    Label,
    normalize_binary_response,
    parse_binary_response,
    parse_narrative_response,
    score_episode,
)
from core.parser import (
    NarrativeParseStatus,
    ParseStatus,
)
from tasks.ruleshift_benchmark.protocol import InteractionLabel

ATTRACT = InteractionLabel.ATTRACT
REPEL = InteractionLabel.REPEL

_VALID_TARGETS = ("attract", "repel", "repel", "attract")
_VALID_LABELS = (ATTRACT, REPEL, REPEL, ATTRACT)


def _make_valid_narrative_text() -> str:
    return "\n".join(
        (
            "rule_before: opposite-sign attract, same-sign repel",
            "shift_evidence: observations 3-5 contradict the initial rule",
            "rule_after: same-sign attract, opposite-sign repel",
            "final_decision: attract, repel, repel, attract",
        )
    )


# ---------------------------------------------------------------------------
# Binary: BinaryResponse dataclass (original happy path)
# ---------------------------------------------------------------------------


class TestBinaryDataclassResponse:
    def test_valid_binary_response_dataclass(self):
        response = BinaryResponse(
            Label.attract, Label.repel, Label.repel, Label.attract
        )
        parsed = parse_binary_response(response)
        assert parsed.status is ParseStatus.VALID
        assert parsed.labels == _VALID_LABELS

    def test_normalize_binary_dataclass(self):
        response = BinaryResponse(
            Label.attract, Label.repel, Label.repel, Label.attract
        )
        assert normalize_binary_response(response) == _VALID_TARGETS

    def test_score_with_binary_dataclass(self):
        response = BinaryResponse(
            Label.attract, Label.repel, Label.repel, Label.attract
        )
        parsed = parse_binary_response(response)
        predictions = tuple(label.value for label in parsed.labels)
        assert score_episode(predictions, _VALID_TARGETS) == (4, 4)


# ---------------------------------------------------------------------------
# Binary: plain dict with string values (PRIMARY REGRESSION)
# ---------------------------------------------------------------------------


class TestBinaryDictStringResponse:
    """This is the exact failure mode that caused the 0% collapse.
    LLM SDKs return dicts instead of dataclass instances."""

    def test_dict_with_string_labels(self):
        response = {
            "probe_6": "attract",
            "probe_7": "repel",
            "probe_8": "repel",
            "probe_9": "attract",
        }
        parsed = parse_binary_response(response)
        assert parsed.status is ParseStatus.VALID, (
            f"Dict response must be parsed as VALID, got {parsed.status}"
        )
        assert parsed.labels == _VALID_LABELS

    def test_normalize_dict_string(self):
        response = {
            "probe_6": "attract",
            "probe_7": "repel",
            "probe_8": "repel",
            "probe_9": "attract",
        }
        assert normalize_binary_response(response) == _VALID_TARGETS

    def test_score_with_dict_string(self):
        response = {
            "probe_6": "attract",
            "probe_7": "repel",
            "probe_8": "repel",
            "probe_9": "attract",
        }
        parsed = parse_binary_response(response)
        predictions = tuple(label.value for label in parsed.labels)
        assert score_episode(predictions, _VALID_TARGETS) == (4, 4)

    def test_dict_with_wrong_labels_scores_zero(self):
        response = {
            "probe_6": "repel",
            "probe_7": "attract",
            "probe_8": "attract",
            "probe_9": "repel",
        }
        parsed = parse_binary_response(response)
        assert parsed.status is ParseStatus.VALID
        predictions = tuple(label.value for label in parsed.labels)
        assert score_episode(predictions, _VALID_TARGETS) == (0, 4)


# ---------------------------------------------------------------------------
# Binary: dict with Enum values
# ---------------------------------------------------------------------------


class TestBinaryDictEnumResponse:
    def test_dict_with_label_enum_values(self):
        response = {
            "probe_6": Label.attract,
            "probe_7": Label.repel,
            "probe_8": Label.repel,
            "probe_9": Label.attract,
        }
        parsed = parse_binary_response(response)
        assert parsed.status is ParseStatus.VALID
        assert parsed.labels == _VALID_LABELS

    def test_dict_with_interaction_label_enum_values(self):
        response = {
            "probe_6": InteractionLabel.ATTRACT,
            "probe_7": InteractionLabel.REPEL,
            "probe_8": InteractionLabel.REPEL,
            "probe_9": InteractionLabel.ATTRACT,
        }
        parsed = parse_binary_response(response)
        assert parsed.status is ParseStatus.VALID
        assert parsed.labels == _VALID_LABELS

    def test_dict_with_custom_enum_values(self):
        class CustomLabel(Enum):
            ATTRACT = "attract"
            REPEL = "repel"

        response = {
            "probe_6": CustomLabel.ATTRACT,
            "probe_7": CustomLabel.REPEL,
            "probe_8": CustomLabel.REPEL,
            "probe_9": CustomLabel.ATTRACT,
        }
        parsed = parse_binary_response(response)
        assert parsed.status is ParseStatus.VALID
        assert parsed.labels == _VALID_LABELS


# ---------------------------------------------------------------------------
# Binary: mapping-like wrapper (e.g. MappingProxyType, SDK wrapper)
# ---------------------------------------------------------------------------


class TestBinaryMappingResponse:
    def test_mapping_proxy_type(self):
        response = MappingProxyType({
            "probe_6": "attract",
            "probe_7": "repel",
            "probe_8": "repel",
            "probe_9": "attract",
        })
        parsed = parse_binary_response(response)
        assert parsed.status is ParseStatus.VALID
        assert parsed.labels == _VALID_LABELS


# ---------------------------------------------------------------------------
# Binary: attribute-bearing wrapper (Pydantic model style)
# ---------------------------------------------------------------------------


class TestBinaryAttributeResponse:
    def test_object_with_matching_attributes(self):
        @dataclass(frozen=True)
        class SDKResponse:
            probe_6: str
            probe_7: str
            probe_8: str
            probe_9: str

        response = SDKResponse("attract", "repel", "repel", "attract")
        parsed = parse_binary_response(response)
        assert parsed.status is ParseStatus.VALID
        assert parsed.labels == _VALID_LABELS

    def test_object_with_extra_attributes(self):
        @dataclass(frozen=True)
        class SDKResponse:
            probe_6: str
            probe_7: str
            probe_8: str
            probe_9: str
            model_name: str = "test"

        response = SDKResponse("attract", "repel", "repel", "attract")
        parsed = parse_binary_response(response)
        assert parsed.status is ParseStatus.VALID
        assert parsed.labels == _VALID_LABELS


# ---------------------------------------------------------------------------
# Binary: provider failure (None)
# ---------------------------------------------------------------------------


class TestBinaryProviderFailure:
    def test_none_returns_skipped_provider_failure(self):
        parsed = parse_binary_response(None)
        assert parsed.status is ParseStatus.SKIPPED_PROVIDER_FAILURE

    def test_none_scores_zero(self):
        assert score_episode(None, _VALID_TARGETS) == (0, 4)


# ---------------------------------------------------------------------------
# Binary: string response (text parsing)
# ---------------------------------------------------------------------------


class TestBinaryStringResponse:
    def test_valid_text(self):
        parsed = parse_binary_response("attract, repel, repel, attract")
        assert parsed.status is ParseStatus.VALID
        assert parsed.labels == _VALID_LABELS

    def test_malformed_text(self):
        parsed = parse_binary_response("I don't know")
        assert parsed.status is ParseStatus.INVALID


# ---------------------------------------------------------------------------
# Binary: invalid response shapes
# ---------------------------------------------------------------------------


class TestBinaryInvalidResponses:
    def test_empty_dict(self):
        parsed = parse_binary_response({})
        assert parsed.status is ParseStatus.INVALID

    def test_dict_with_missing_fields(self):
        parsed = parse_binary_response({
            "probe_6": "attract",
            "probe_7": "repel",
        })
        assert parsed.status is ParseStatus.INVALID

    def test_dict_with_invalid_label(self):
        parsed = parse_binary_response({
            "probe_6": "attract",
            "probe_7": "repel",
            "probe_8": "bounce",  # invalid
            "probe_9": "attract",
        })
        assert parsed.status is ParseStatus.INVALID

    def test_list_response(self):
        parsed = parse_binary_response(["attract", "repel", "repel", "attract"])
        assert parsed.status is ParseStatus.INVALID

    def test_integer_response(self):
        parsed = parse_binary_response(42)
        assert parsed.status is ParseStatus.INVALID


# ---------------------------------------------------------------------------
# Narrative: text response paths
# ---------------------------------------------------------------------------


class TestNarrativeTextResponse:
    def test_valid_text_response(self):
        parsed = parse_narrative_response(_make_valid_narrative_text())
        assert parsed.status is NarrativeParseStatus.VALID
        assert parsed.output is not None

    def test_none_returns_skipped(self):
        parsed = parse_narrative_response(None)
        assert parsed.status is NarrativeParseStatus.SKIPPED_PROVIDER_FAILURE


# ---------------------------------------------------------------------------
# Narrative: dict with text/content key
# ---------------------------------------------------------------------------


class TestNarrativeDictResponse:
    def test_dict_with_text_key(self):
        response = {"text": _make_valid_narrative_text()}
        parsed = parse_narrative_response(response)
        assert parsed.status is NarrativeParseStatus.VALID

    def test_dict_with_content_key(self):
        response = {"content": _make_valid_narrative_text()}
        parsed = parse_narrative_response(response)
        assert parsed.status is NarrativeParseStatus.VALID

    def test_dict_without_text_or_content(self):
        response = {"other_key": "some value"}
        parsed = parse_narrative_response(response)
        assert parsed.status is NarrativeParseStatus.INVALID_FORMAT
        assert "unsupported response type" in (parsed.failure_detail or "")


# ---------------------------------------------------------------------------
# Narrative: wrapper object with .text attribute
# ---------------------------------------------------------------------------


class TestNarrativeWrapperResponse:
    def test_wrapper_with_text_attribute(self):
        @dataclass
        class SDKWrapper:
            text: str

        response = SDKWrapper(text=_make_valid_narrative_text())
        parsed = parse_narrative_response(response)
        assert parsed.status is NarrativeParseStatus.VALID

    def test_wrapper_with_content_attribute(self):
        @dataclass
        class SDKWrapper:
            content: str

        response = SDKWrapper(content=_make_valid_narrative_text())
        parsed = parse_narrative_response(response)
        assert parsed.status is NarrativeParseStatus.VALID


# ---------------------------------------------------------------------------
# Integration: dict response through full score path
# ---------------------------------------------------------------------------


class TestDictResponseScoreIntegration:
    """End-to-end: a dict response from the LLM SDK must produce a correct
    score when passed through the complete pipeline."""

    def test_perfect_dict_response_scores_four(self):
        response = {
            "probe_6": "attract",
            "probe_7": "repel",
            "probe_8": "repel",
            "probe_9": "attract",
        }
        parsed = parse_binary_response(response)
        assert parsed.status is ParseStatus.VALID
        predictions = tuple(label.value for label in parsed.labels)
        assert score_episode(predictions, _VALID_TARGETS) == (4, 4)

    def test_half_correct_dict_response_scores_two(self):
        response = {
            "probe_6": "attract",
            "probe_7": "attract",  # wrong
            "probe_8": "repel",
            "probe_9": "repel",    # wrong
        }
        parsed = parse_binary_response(response)
        assert parsed.status is ParseStatus.VALID
        predictions = tuple(label.value for label in parsed.labels)
        assert score_episode(predictions, _VALID_TARGETS) == (2, 4)


# ---------------------------------------------------------------------------
# Contract: split episode count cross-check
# ---------------------------------------------------------------------------


class TestSplitEpisodeCountContract:
    """Regression test: EXPECTED_EPISODES_PER_SPLIT must match actual frozen
    manifests. This would have caught the 16→18 drift."""

    def test_expected_episodes_matches_manifests(self):
        from core.contract_audit import EXPECTED_EPISODES_PER_SPLIT, check_split_episode_counts

        errors = check_split_episode_counts()
        assert errors == [], (
            f"EXPECTED_EPISODES_PER_SPLIT={EXPECTED_EPISODES_PER_SPLIT} "
            f"does not match actual frozen manifests: {errors}"
        )

    def test_expected_episodes_equals_manifest_seed_count(self):
        from core.contract_audit import EXPECTED_EPISODES_PER_SPLIT
        from core.splits import PUBLIC_PARTITIONS, load_split_manifest

        for partition in PUBLIC_PARTITIONS:
            manifest = load_split_manifest(partition)
            assert len(manifest.seeds) == EXPECTED_EPISODES_PER_SPLIT, (
                f"{partition} has {len(manifest.seeds)} seeds but "
                f"EXPECTED_EPISODES_PER_SPLIT is {EXPECTED_EPISODES_PER_SPLIT}"
            )


# ---------------------------------------------------------------------------
# Diagnostic: response type is preserved in parse status
# ---------------------------------------------------------------------------


class TestDiagnosticCategories:
    """Verify that different failure modes produce distinct, informative statuses
    rather than collapsing into a generic 'unknown'."""

    def test_none_is_provider_failure_not_invalid(self):
        parsed = parse_binary_response(None)
        assert parsed.status is ParseStatus.SKIPPED_PROVIDER_FAILURE
        assert parsed.status is not ParseStatus.INVALID

    def test_valid_dict_is_valid_not_invalid(self):
        response = {
            "probe_6": "attract",
            "probe_7": "repel",
            "probe_8": "repel",
            "probe_9": "attract",
        }
        parsed = parse_binary_response(response)
        assert parsed.status is ParseStatus.VALID
        assert parsed.status is not ParseStatus.INVALID

    def test_malformed_dict_is_invalid_not_provider_failure(self):
        parsed = parse_binary_response({"wrong_key": "attract"})
        assert parsed.status is ParseStatus.INVALID
        assert parsed.status is not ParseStatus.SKIPPED_PROVIDER_FAILURE

    def test_narrative_dict_wrapper_preserves_failure_detail(self):
        parsed = parse_narrative_response({"unrecognized": 42})
        assert parsed.status is NarrativeParseStatus.INVALID_FORMAT
        assert parsed.failure_detail is not None
        assert "unsupported response type" in parsed.failure_detail
