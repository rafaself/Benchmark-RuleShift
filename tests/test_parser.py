from core.parser import (
    NarrativeParseStatus,
    NarrativeParsedResult,
    ParsedPrediction,
    ParseStatus,
    parse_binary_output,
    parse_narrative_audit_output,
)
from tasks.ruleshift_benchmark.protocol import InteractionLabel

ATTRACT = InteractionLabel.ATTRACT
REPEL = InteractionLabel.REPEL


def _make_valid_narrative_text(final_decision: str = "attract, repel, repel, attract") -> str:
    return "\n".join(
        (
            "rule_before: opposite-sign attract, same-sign repel",
            "shift_evidence: observations 3-5 contradict the initial rule",
            "rule_after: same-sign attract, opposite-sign repel",
            f"final_decision: {final_decision}",
        )
    )


# ---------------------------------------------------------------------------
# Binary parser tests
# ---------------------------------------------------------------------------


def test_binary_output_parses_exactly_four_labels_in_order():
    parsed = parse_binary_output("attract, repel, repel, attract")

    assert parsed == ParsedPrediction(
        labels=(ATTRACT, REPEL, REPEL, ATTRACT),
        status=ParseStatus.VALID,
    )


def test_safe_formatting_variants_normalize_to_canonical_labels():
    expected = ParsedPrediction(
        labels=(ATTRACT, REPEL, REPEL, ATTRACT),
        status=ParseStatus.VALID,
    )

    assert parse_binary_output("  ATTRACT, repel,\nREPEL,\nattract  ") == expected
    assert parse_binary_output("\nattract\nrepel\nrepel\nattract\n") == expected


def test_malformed_binary_outputs_are_rejected():
    invalid = ParsedPrediction(labels=(), status=ParseStatus.INVALID)

    assert parse_binary_output("attract, repel, repels, attract") == invalid
    assert parse_binary_output("attract, repel, repel, attract because of the shift") == invalid


def test_wrong_length_binary_outputs_use_invalid_result():
    invalid = ParsedPrediction(labels=(), status=ParseStatus.INVALID)

    assert parse_binary_output("attract, repel, repel") == invalid
    assert parse_binary_output("attract, repel, repel, attract, attract") == invalid


# ---------------------------------------------------------------------------
# Narrative audit parser — valid cases
# ---------------------------------------------------------------------------


def test_narrative_audit_parses_valid_four_line_contract():
    result = parse_narrative_audit_output(_make_valid_narrative_text())

    assert result.status is NarrativeParseStatus.VALID
    assert result.output is not None
    assert result.output.rule_before == "opposite-sign attract, same-sign repel"
    assert result.output.shift_evidence == "observations 3-5 contradict the initial rule"
    assert result.output.rule_after == "same-sign attract, opposite-sign repel"
    assert result.output.final_decision == (ATTRACT, REPEL, REPEL, ATTRACT)
    assert result.failure_detail is None


def test_narrative_audit_handles_contract_in_markdown_code_block():
    text = "```\n" + _make_valid_narrative_text() + "\n```"
    result = parse_narrative_audit_output(text)
    assert result.status is NarrativeParseStatus.VALID
    assert result.output.final_decision == (ATTRACT, REPEL, REPEL, ATTRACT)


def test_narrative_audit_normalizes_key_case_and_label_case():
    text = "\n".join(
        (
            "RULE_BEFORE: rule A",
            "SHIFT_EVIDENCE: evidence",
            "RULE_AFTER: rule B",
            "FINAL_DECISION: ATTRACT, Repel, REPEL, Attract",
        )
    )
    result = parse_narrative_audit_output(text)
    assert result.status is NarrativeParseStatus.VALID
    assert result.output.final_decision == (ATTRACT, REPEL, REPEL, ATTRACT)


def test_narrative_audit_strips_whitespace_from_values():
    text = "\n".join(
        (
            "rule_before:   rule A  ",
            "shift_evidence:   evidence  ",
            "rule_after:   rule B  ",
            "final_decision:   attract, repel, repel, attract  ",
        )
    )
    result = parse_narrative_audit_output(text)
    assert result.status is NarrativeParseStatus.VALID
    assert result.output.rule_before == "rule A"
    assert result.output.shift_evidence == "evidence"
    assert result.output.rule_after == "rule B"


# ---------------------------------------------------------------------------
# Narrative audit parser — invalid cases
# ---------------------------------------------------------------------------


def test_narrative_audit_rejects_empty_text():
    result = parse_narrative_audit_output("")
    assert result.status is NarrativeParseStatus.INVALID_FORMAT
    assert result.output is None


def test_narrative_audit_rejects_non_contract_prose():
    result = parse_narrative_audit_output(
        "The rule shifted after observation 3. My answers are attract, repel, repel, attract."
    )
    assert result.status is NarrativeParseStatus.INVALID_FORMAT
    assert result.output is None


def test_narrative_audit_rejects_unknown_field():
    text = "\n".join(
        (
            "rule_before: rule A",
            "shift_evidence: evidence",
            "rule_after: rule B",
            "final_answer: attract, repel, repel, attract",
        )
    )
    result = parse_narrative_audit_output(text)
    assert result.status is NarrativeParseStatus.INVALID_FORMAT
    assert "unknown narrative field" in (result.failure_detail or "")


def test_narrative_audit_rejects_duplicate_field():
    text = "\n".join(
        (
            "rule_before: rule A",
            "shift_evidence: evidence",
            "rule_after: rule B",
            "rule_after: another rule B",
        )
    )
    result = parse_narrative_audit_output(text)
    assert result.status is NarrativeParseStatus.INVALID_FORMAT
    assert "duplicate narrative field" in (result.failure_detail or "")


def test_narrative_audit_rejects_missing_rule_before():
    text = "\n".join(
        (
            "shift_evidence: evidence",
            "rule_after: rule B",
            "final_decision: attract, repel, repel, attract",
        )
    )
    result = parse_narrative_audit_output(text)
    assert result.status is NarrativeParseStatus.INVALID_FORMAT
    assert "exactly 4 non-empty contract lines" in (result.failure_detail or "")


def test_narrative_audit_rejects_empty_field_value():
    text = "\n".join(
        (
            "rule_before: ",
            "shift_evidence: evidence",
            "rule_after: rule B",
            "final_decision: attract, repel, repel, attract",
        )
    )
    result = parse_narrative_audit_output(text)
    assert result.status is NarrativeParseStatus.MISSING_FIELD
    assert "rule_before" in (result.failure_detail or "")


def test_narrative_audit_rejects_extra_prose_line():
    text = _make_valid_narrative_text() + "\nsummary: extra prose"
    result = parse_narrative_audit_output(text)
    assert result.status is NarrativeParseStatus.INVALID_FORMAT
    assert "exactly 4 non-empty contract lines" in (result.failure_detail or "")


def test_narrative_audit_rejects_invalid_label_value():
    result = parse_narrative_audit_output(
        _make_valid_narrative_text("attract, repel, bounce, attract")
    )
    assert result.status is NarrativeParseStatus.INVALID_LABELS
    assert result.output is None


def test_narrative_audit_rejects_too_few_labels():
    result = parse_narrative_audit_output(
        _make_valid_narrative_text("attract, repel, repel")
    )
    assert result.status is NarrativeParseStatus.INVALID_LABELS


def test_narrative_audit_rejects_too_many_labels():
    result = parse_narrative_audit_output(
        _make_valid_narrative_text("attract, repel, repel, attract, attract")
    )
    assert result.status is NarrativeParseStatus.INVALID_LABELS


# ---------------------------------------------------------------------------
# NarrativeParsedResult factory methods
# ---------------------------------------------------------------------------


def test_narrative_parsed_result_skipped_provider_failure_factory():
    result = NarrativeParsedResult.skipped_provider_failure()
    assert result.status is NarrativeParseStatus.SKIPPED_PROVIDER_FAILURE
    assert result.output is None
