from __future__ import annotations

from pathlib import Path

from core.intra_family_comparison import build_comparison, render_comparison_markdown


def test_build_comparison_normalizes_diagnostic_mode_labels_and_preserves_run_paths():
    meta_a = {
        "provider": "gemini",
        "requested_model_id": "gemini-2.5-flash",
        "served_model_id": None,
        "execution_timestamp": "20260323_120000",
        "benchmark_versions": {
            "schema_version": "v1",
            "generator_version": "R13",
            "template_family_version": "v2",
            "parser_version": "v1",
            "metric_version": "v1",
            "difficulty_version": "R13",
            "artifact_schema_version": "v1.1",
        },
        "frozen_artifacts": {
            "split_manifests": [
                {"split_name": "dev", "sha256": "dev-hash"},
                {"split_name": "public_leaderboard", "sha256": "public-hash"},
                {"split_name": "private_leaderboard", "sha256": "private-hash"},
            ]
        },
    }
    meta_b = {
        **meta_a,
        "requested_model_id": "gemini-2.5-flash-lite",
        "served_model_id": "gemini-2.5-flash-lite",
        "execution_timestamp": "20260323_125335",
    }

    def _split_rows(model_name: str):
        return [
            {
                "split_name": "dev",
                "rows": [
                    {
                        "episode_id": "ep-1",
                        "template_id": "T1",
                        "template_family": "canonical",
                        "difficulty": "easy",
                        "transition": "R_std_to_R_inv",
                        "probe_targets": ["attract", "repel", "attract", "repel"],
                        "modes": {
                            "binary": {
                                "parse_status": "valid",
                                "correct_probe_count": 3,
                                "failure_bucket": "adaptation_failure",
                            },
                            "narrative": {
                                "parse_status": "provider_error"
                                if model_name.endswith("lite")
                                else "valid",
                                "correct_probe_count": 0,
                                "failure_bucket": "provider_error"
                                if model_name.endswith("lite")
                                else "adaptation_failure",
                            },
                        },
                    }
                ],
            },
            {
                "split_name": "public_leaderboard",
                "rows": [
                    {
                        "episode_id": "ep-2",
                        "template_id": "T2",
                        "template_family": "observation_log",
                        "difficulty": "medium",
                        "transition": "R_inv_to_R_std",
                        "probe_targets": ["repel", "repel", "attract", "attract"],
                        "modes": {
                            "binary": {
                                "parse_status": "valid",
                                "correct_probe_count": 4,
                                "failure_bucket": "correct",
                            },
                            "narrative": {
                                "parse_status": "valid",
                                "correct_probe_count": 1,
                                "failure_bucket": "adaptation_failure",
                            },
                        },
                    }
                ],
            },
            {
                "split_name": "private_leaderboard",
                "rows": [
                    {
                        "episode_id": "ep-3",
                        "template_id": "T1",
                        "template_family": "canonical",
                        "difficulty": "easy",
                        "transition": "R_std_to_R_inv",
                        "probe_targets": ["repel", "attract", "repel", "attract"],
                        "modes": {
                            "binary": {
                                "parse_status": "valid",
                                "correct_probe_count": 2,
                                "failure_bucket": "adaptation_failure",
                            },
                            "narrative": {
                                "parse_status": "parse_error"
                                if model_name.endswith("lite")
                                else "valid",
                                "correct_probe_count": 0,
                                "failure_bucket": "parse_failure"
                                if model_name.endswith("lite")
                                else "adaptation_failure",
                            },
                        },
                    }
                ],
            },
        ]

    art_a = {
        "release_id": "R18",
        "provider_name": "gemini",
        "model_name": "gemini-2.5-flash",
        "prompt_modes": ["binary", "narrative"],
        "provenance_note": "partial legacy provenance",
        "splits": _split_rows("gemini-2.5-flash"),
        "diagnostic_summary": [
            {
                "scope_type": "overall",
                "scope_label": "overall",
                "mode": "Binary",
                "runtime_error_rate": 0.0,
                "runtime_error_count": 0,
                "parse_failure_rate": 0.0,
                "parse_failure_count": 0,
                "parse_valid_rate": 1.0,
                "correct_rate": 0.75,
                "adaptation_failure_rate": 0.25,
                "adaptation_failure_count": 1,
            },
            {
                "scope_type": "overall",
                "scope_label": "overall",
                "mode": "Narrative",
                "runtime_error_rate": 0.0,
                "runtime_error_count": 0,
                "parse_failure_rate": 0.0,
                "parse_failure_count": 0,
                "parse_valid_rate": 1.0,
                "correct_rate": 0.25,
                "adaptation_failure_rate": 0.75,
                "adaptation_failure_count": 3,
            },
        ],
        "failure_taxonomy": [],
    }
    art_b = {
        **art_a,
        "model_name": "gemini-2.5-flash-lite",
        "provenance_note": None,
        "splits": _split_rows("gemini-2.5-flash-lite"),
        "diagnostic_summary": [
            {
                "scope_type": "overall",
                "scope_label": "overall",
                "mode": "Binary",
                "runtime_error_rate": 0.05,
                "runtime_error_count": 1,
                "parse_failure_rate": 0.0,
                "parse_failure_count": 0,
                "parse_valid_rate": 0.95,
                "correct_rate": 0.5,
                "adaptation_failure_rate": 0.45,
                "adaptation_failure_count": 2,
            },
            {
                "scope_type": "overall",
                "scope_label": "overall",
                "mode": "Narrative",
                "runtime_error_rate": 0.2,
                "runtime_error_count": 1,
                "parse_failure_rate": 0.3,
                "parse_failure_count": 1,
                "parse_valid_rate": 0.5,
                "correct_rate": 0.1,
                "adaptation_failure_rate": 0.4,
                "adaptation_failure_count": 1,
            },
            {
                "scope_type": "split",
                "scope_label": "dev",
                "mode": "Narrative",
                "runtime_error_rate": 0.0,
                "runtime_error_count": 0,
                "parse_failure_rate": 0.0,
                "parse_failure_count": 0,
                "parse_valid_rate": 1.0,
                "correct_rate": 0.25,
                "adaptation_failure_rate": 0.75,
                "adaptation_failure_count": 1,
            },
            {
                "scope_type": "split",
                "scope_label": "public_leaderboard",
                "mode": "Narrative",
                "runtime_error_rate": 0.0,
                "runtime_error_count": 0,
                "parse_failure_rate": 0.0,
                "parse_failure_count": 0,
                "parse_valid_rate": 1.0,
                "correct_rate": 0.25,
                "adaptation_failure_rate": 0.75,
                "adaptation_failure_count": 1,
            },
            {
                "scope_type": "split",
                "scope_label": "private_leaderboard",
                "mode": "Narrative",
                "runtime_error_rate": 0.25,
                "runtime_error_count": 1,
                "parse_failure_rate": 0.5,
                "parse_failure_count": 1,
                "parse_valid_rate": 0.25,
                "correct_rate": 0.0,
                "adaptation_failure_rate": 0.25,
                "adaptation_failure_count": 0,
            },
                {
                    "scope_type": "template",
                    "scope_label": "T1",
                    "mode": "Narrative",
                "runtime_error_rate": 0.2,
                "runtime_error_count": 1,
                "parse_failure_rate": 0.4,
                "parse_failure_count": 1,
                "parse_valid_rate": 0.25,
                "correct_rate": 0.0,
                    "adaptation_failure_rate": 0.2,
                    "adaptation_failure_count": 0,
                },
                {
                    "scope_type": "template_family",
                    "scope_label": "canonical",
                    "mode": "Narrative",
                    "runtime_error_rate": 0.2,
                    "runtime_error_count": 1,
                    "parse_failure_rate": 0.4,
                    "parse_failure_count": 1,
                    "parse_valid_rate": 0.25,
                    "correct_rate": 0.0,
                    "adaptation_failure_rate": 0.2,
                    "adaptation_failure_count": 0,
                },
                {
                    "scope_type": "template_family",
                    "scope_label": "observation_log",
                    "mode": "Narrative",
                    "runtime_error_rate": 0.0,
                    "runtime_error_count": 0,
                    "parse_failure_rate": 0.0,
                    "parse_failure_count": 0,
                    "parse_valid_rate": 1.0,
                    "correct_rate": 0.25,
                    "adaptation_failure_rate": 0.75,
                    "adaptation_failure_count": 1,
                },
                {
                    "scope_type": "template",
                    "scope_label": "T2",
                    "mode": "Narrative",
                "runtime_error_rate": 0.0,
                "runtime_error_count": 0,
                "parse_failure_rate": 0.0,
                "parse_failure_count": 0,
                "parse_valid_rate": 1.0,
                "correct_rate": 0.25,
                "adaptation_failure_rate": 0.75,
                "adaptation_failure_count": 1,
            },
            {
                "scope_type": "transition",
                "scope_label": "R_std_to_R_inv",
                "mode": "Narrative",
                "runtime_error_rate": 0.2,
                "runtime_error_count": 1,
                "parse_failure_rate": 0.4,
                "parse_failure_count": 1,
                "parse_valid_rate": 0.25,
                "correct_rate": 0.0,
                "adaptation_failure_rate": 0.2,
                "adaptation_failure_count": 0,
            },
            {
                "scope_type": "transition",
                "scope_label": "R_inv_to_R_std",
                "mode": "Narrative",
                "runtime_error_rate": 0.0,
                "runtime_error_count": 0,
                "parse_failure_rate": 0.0,
                "parse_failure_count": 0,
                "parse_valid_rate": 1.0,
                "correct_rate": 0.25,
                "adaptation_failure_rate": 0.75,
                "adaptation_failure_count": 1,
            },
        ],
        "failure_taxonomy": [],
    }

    comparison = build_comparison(
        art_a,
        art_b,
        meta_a,
        meta_b,
        artifact_path_a=Path("reports/a_artifact.json"),
        metadata_path_a=Path("reports/a_metadata.json"),
        report_path_a=Path("reports/a_report.md"),
        artifact_path_b=Path("reports/b_artifact.json"),
        metadata_path_b=Path("reports/b_metadata.json"),
        report_path_b=Path("reports/b_report.md"),
    )

    assert comparison["run_a"]["artifact_path"] == "reports/a_artifact.json"
    assert comparison["run_b"]["metadata_path"] == "reports/b_metadata.json"
    assert comparison["run_b"]["served_model"] == "gemini-2.5-flash-lite"

    concentration = comparison["concentration_analysis"]
    assert concentration["by_mode"] == [
        {
            "mode": "binary",
            "runtime_rate": 0.05,
            "parse_failure_rate": 0.0,
            "adaptation_rate": 0.45,
        },
        {
            "mode": "narrative",
            "runtime_rate": 0.2,
            "parse_failure_rate": 0.3,
            "adaptation_rate": 0.4,
        },
    ]
    assert any("private_leaderboard" in finding for finding in concentration["findings"])
    assert any("template T1" in finding for finding in concentration["findings"])
    assert any("transition R_std_to_R_inv" in finding for finding in concentration["findings"])

    report = render_comparison_markdown(comparison)
    assert "| Served model | not recorded | gemini-2.5-flash-lite |" in report
    assert "### By Template (Narrative)" in report
    assert "### By Transition (Narrative)" in report
    assert "`reports/a_artifact.json`" in report
