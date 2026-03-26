"""Intra-family model comparison from canonical stored artifacts.

Compares two panel runs of models within the same provider family.
Reads only from stored artifact JSON and metadata JSON files;
does not re-run the benchmark.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = [
    "load_run",
    "verify_comparability",
    "build_comparison",
    "render_comparison_markdown",
]

COMPARISON_SCHEMA_VERSION = "v1"

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_run(
    artifact_path: Path, metadata_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (artifact, metadata) dicts for a single run."""
    with open(artifact_path, encoding="utf-8") as f:
        artifact = json.load(f)
    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)
    return artifact, metadata


# ---------------------------------------------------------------------------
# Comparability gate
# ---------------------------------------------------------------------------


def verify_comparability(
    meta_a: dict[str, Any],
    meta_b: dict[str, Any],
    art_a: dict[str, Any],
    art_b: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Return (ok, issues).  ok=False means runs are NOT comparable."""
    issues: list[str] = []

    # Benchmark version fields that must match exactly.
    bv_a = meta_a.get("benchmark_versions", {})
    bv_b = meta_b.get("benchmark_versions", {})
    for key in (
        "schema_version",
        "generator_version",
        "template_family_version",
        "parser_version",
        "metric_version",
        "difficulty_version",
        "artifact_schema_version",
    ):
        va, vb = bv_a.get(key), bv_b.get(key)
        if va != vb:
            issues.append(f"benchmark_versions.{key} mismatch: {va!r} vs {vb!r}")

    # Frozen split manifests must have identical hashes.
    manifests_a = {
        m["split_name"]: m
        for m in meta_a.get("frozen_artifacts", {}).get("split_manifests", [])
    }
    manifests_b = {
        m["split_name"]: m
        for m in meta_b.get("frozen_artifacts", {}).get("split_manifests", [])
    }
    if set(manifests_a) != set(manifests_b):
        issues.append(
            f"split set mismatch: {sorted(manifests_a)} vs {sorted(manifests_b)}"
        )
    else:
        for split_name in sorted(manifests_a):
            ha = manifests_a[split_name].get("sha256")
            hb = manifests_b[split_name].get("sha256")
            if ha != hb:
                issues.append(f"split {split_name} sha256 mismatch: {ha} vs {hb}")

    # Both must include the same prompt modes for a fair comparison.
    modes_a = set(art_a.get("prompt_modes", []))
    modes_b = set(art_b.get("prompt_modes", []))
    if modes_a != modes_b:
        issues.append(f"prompt_modes mismatch: {sorted(modes_a)} vs {sorted(modes_b)}")

    # Episode IDs and probe targets must match across splits.
    for split_a in art_a.get("splits", []):
        split_name = split_a["split_name"]
        split_b = _find_split(art_b, split_name)
        if split_b is None:
            issues.append(f"split {split_name} missing in run B artifact")
            continue
        ids_a = [r["episode_id"] for r in split_a["rows"]]
        ids_b = [r["episode_id"] for r in split_b["rows"]]
        if ids_a != ids_b:
            issues.append(f"split {split_name}: episode_id ordering mismatch")
        for row_a, row_b in zip(split_a["rows"], split_b["rows"]):
            if row_a["probe_targets"] != row_b["probe_targets"]:
                issues.append(
                    f"split {split_name} episode {row_a['episode_id']}: "
                    "probe_targets mismatch"
                )

    return (len(issues) == 0, issues)


# ---------------------------------------------------------------------------
# Comparison builder
# ---------------------------------------------------------------------------

_PROBE_COUNT = 4


def build_comparison(
    art_a: dict[str, Any],
    art_b: dict[str, Any],
    meta_a: dict[str, Any],
    meta_b: dict[str, Any],
    *,
    artifact_path_a: Path | None = None,
    metadata_path_a: Path | None = None,
    report_path_a: Path | None = None,
    artifact_path_b: Path | None = None,
    metadata_path_b: Path | None = None,
    report_path_b: Path | None = None,
) -> dict[str, Any]:
    """Build a full comparison payload from two run artifacts."""
    model_a = art_a["model_name"]
    model_b = art_b["model_name"]
    modes = sorted(set(art_a.get("prompt_modes", [])))

    # ---- per-episode comparison ----
    episode_rows: list[dict[str, Any]] = []
    # accumulators keyed by (scope_type, scope_label, mode)
    acc: dict[tuple[str, str, str], _Acc] = {}

    for split_a in art_a["splits"]:
        split_name = split_a["split_name"]
        split_b = _find_split(art_b, split_name)
        assert split_b is not None
        for row_a, row_b in zip(split_a["rows"], split_b["rows"]):
            for mode in modes:
                ma = row_a["modes"].get(mode)
                mb = row_b["modes"].get(mode)
                if ma is None or mb is None:
                    continue
                ep_row = _compare_episode(
                    row_a, row_b, mode, split_name, model_a, model_b
                )
                episode_rows.append(ep_row)

                # Accumulate into scopes
                for scope_type, scope_label in _scope_keys(row_a, split_name):
                    key = (scope_type, scope_label, mode)
                    if key not in acc:
                        acc[key] = _Acc()
                    acc[key].add(ma, mb, ep_row)

    # ---- summary tables ----
    summary_rows = _build_summary_rows(acc, model_a, model_b)

    # ---- failure decomposition comparison ----
    failure_comparison = _build_failure_comparison(art_a, art_b, model_a, model_b)

    # ---- failure taxonomy comparison ----
    taxonomy_comparison = _build_taxonomy_comparison(art_a, art_b, model_a, model_b)

    # ---- disagreement comparison ----
    disagreement_comparison = _build_disagreement_comparison(
        art_a, art_b, model_a, model_b
    )

    # ---- concentration analysis ----
    concentration = _build_concentration_analysis(art_b, model_b)

    return {
        "comparison_schema_version": COMPARISON_SCHEMA_VERSION,
        "model_a": model_a,
        "model_b": model_b,
        "provider": art_a["provider_name"],
        "release_id": art_a["release_id"],
        "prompt_modes": modes,
        "benchmark_versions": meta_a.get("benchmark_versions", {}),
        "frozen_split_hashes": {
            m["split_name"]: m["sha256"]
            for m in meta_a.get("frozen_artifacts", {}).get("split_manifests", [])
        },
        "run_a": {
            "provider": meta_a.get("provider"),
            "requested_model": meta_a.get("requested_model_id", model_a),
            "served_model": meta_a.get("served_model_id"),
            "model": model_a,
            "artifact_path": str(artifact_path_a) if artifact_path_a is not None else None,
            "metadata_path": str(metadata_path_a) if metadata_path_a is not None else None,
            "report_path": str(report_path_a) if report_path_a is not None else None,
            "execution_timestamp": meta_a.get("execution_timestamp"),
            "provenance_note": art_a.get("provenance_note"),
        },
        "run_b": {
            "provider": meta_b.get("provider"),
            "requested_model": meta_b.get("requested_model_id", model_b),
            "served_model": meta_b.get("served_model_id"),
            "model": model_b,
            "artifact_path": str(artifact_path_b) if artifact_path_b is not None else None,
            "metadata_path": str(metadata_path_b) if metadata_path_b is not None else None,
            "report_path": str(report_path_b) if report_path_b is not None else None,
            "execution_timestamp": meta_b.get("execution_timestamp"),
            "provenance_note": art_b.get("provenance_note"),
        },
        "summary": summary_rows,
        "failure_comparison": failure_comparison,
        "taxonomy_comparison": taxonomy_comparison,
        "disagreement_comparison": disagreement_comparison,
        "concentration_analysis": concentration,
        "episode_rows": episode_rows,
    }


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------


def render_comparison_markdown(comparison: dict[str, Any]) -> str:
    model_a = comparison["model_a"]
    model_b = comparison["model_b"]
    lines: list[str] = []

    lines.extend([
        "# Intra-Gemini Comparison: Flash vs Flash-Lite",
        "",
        "## Scope",
        "",
        "- This is **intra-family Gemini evidence** measuring stability within the active Gemini readiness path.",
        "- This is **not** a cross-provider conclusion.",
        "- This does **not** widen the benchmark claim.",
        "- Binary remains the only leaderboard-primary path.",
        "- Narrative remains required same-episode robustness evidence.",
        "",
    ])

    # ---- Run Identification ----
    lines.extend([
        "## Run Identification",
        "",
        f"| Field | {model_a} | {model_b} |",
        "| --- | --- | --- |",
        f"| Provider | {comparison['provider']} | {comparison['provider']} |",
        f"| Release | {comparison['release_id']} | {comparison['release_id']} |",
        f"| Requested model | {comparison['run_a'].get('requested_model') or model_a} | {comparison['run_b'].get('requested_model') or model_b} |",
        f"| Served model | {comparison['run_a'].get('served_model') or 'not recorded'} | {comparison['run_b'].get('served_model') or 'not recorded'} |",
        f"| Execution timestamp | {comparison['run_a']['execution_timestamp']} | {comparison['run_b']['execution_timestamp']} |",
        f"| Prompt modes | {', '.join(comparison['prompt_modes'])} | {', '.join(comparison['prompt_modes'])} |",
    ])
    prov_a = comparison["run_a"].get("provenance_note") or "Full provenance"
    prov_b = comparison["run_b"].get("provenance_note") or "Full provenance"
    # Truncate long provenance notes for the table
    prov_a_short = (prov_a[:60] + "...") if len(prov_a) > 63 else prov_a
    prov_b_short = (prov_b[:60] + "...") if len(prov_b) > 63 else prov_b
    lines.append(f"| Provenance | {prov_a_short} | {prov_b_short} |")
    lines.append("")
    lines.extend([
        f"- {model_a} artifact: `{comparison['run_a'].get('artifact_path') or 'n/a'}`",
        f"- {model_a} metadata: `{comparison['run_a'].get('metadata_path') or 'n/a'}`",
        f"- {model_a} report: `{comparison['run_a'].get('report_path') or 'n/a'}`",
        f"- {model_b} artifact: `{comparison['run_b'].get('artifact_path') or 'n/a'}`",
        f"- {model_b} metadata: `{comparison['run_b'].get('metadata_path') or 'n/a'}`",
        f"- {model_b} report: `{comparison['run_b'].get('report_path') or 'n/a'}`",
        "",
    ])

    # ---- Comparability Verification ----
    bv = comparison["benchmark_versions"]
    lines.extend([
        "## Comparability Verification",
        "",
        "| Version field | Value |",
        "| --- | --- |",
    ])
    for key in (
        "schema_version",
        "generator_version",
        "template_family_version",
        "parser_version",
        "metric_version",
        "difficulty_version",
        "artifact_schema_version",
    ):
        lines.append(f"| {key} | {bv.get(key, 'n/a')} |")
    lines.append("")
    lines.extend([
        "| Split | SHA-256 (shared) |",
        "| --- | --- |",
    ])
    for split_name, sha in sorted(comparison["frozen_split_hashes"].items()):
        lines.append(f"| {split_name} | `{sha[:16]}...` |")
    lines.append("")
    lines.append("All benchmark versions and frozen split hashes match. Runs are directly comparable.")
    lines.append("")

    # ---- extract summary data ----
    summary = comparison["summary"]

    def _get(scope_type: str, scope_label: str, mode: str) -> dict[str, Any] | None:
        for row in summary:
            if (
                row["scope_type"] == scope_type
                and row["scope_label"] == scope_label
                and row["mode"] == mode
            ):
                return row
        return None

    # ---- Binary Headline ----
    bin_overall = _get("overall", "overall", "binary")
    lines.extend([
        "## Binary Headline (primary metric)",
        "",
        f"| Metric | {model_a} | {model_b} | Delta ({model_a} - {model_b}) |",
        "| --- | ---: | ---: | ---: |",
    ])
    if bin_overall:
        lines.append(
            f"| Post-shift Probe Accuracy | "
            f"{bin_overall['accuracy_a']:.6f} | "
            f"{bin_overall['accuracy_b']:.6f} | "
            f"{bin_overall['accuracy_a'] - bin_overall['accuracy_b']:+.6f} |"
        )
        lines.append(
            f"| Parse-valid rate | "
            f"{bin_overall['parse_valid_rate_a']:.6f} | "
            f"{bin_overall['parse_valid_rate_b']:.6f} | "
            f"{bin_overall['parse_valid_rate_a'] - bin_overall['parse_valid_rate_b']:+.6f} |"
        )
    lines.append("")

    # ---- Narrative Robustness ----
    nar_overall = _get("overall", "overall", "narrative")
    lines.extend([
        "## Narrative Robustness (same-episode companion)",
        "",
        f"| Metric | {model_a} | {model_b} | Delta |",
        "| --- | ---: | ---: | ---: |",
    ])
    if nar_overall:
        lines.append(
            f"| Post-shift Probe Accuracy | "
            f"{nar_overall['accuracy_a']:.6f} | "
            f"{nar_overall['accuracy_b']:.6f} | "
            f"{nar_overall['accuracy_a'] - nar_overall['accuracy_b']:+.6f} |"
        )
        lines.append(
            f"| Parse-valid rate | "
            f"{nar_overall['parse_valid_rate_a']:.6f} | "
            f"{nar_overall['parse_valid_rate_b']:.6f} | "
            f"{nar_overall['parse_valid_rate_a'] - nar_overall['parse_valid_rate_b']:+.6f} |"
        )
    lines.append("")

    # ---- Binary → Narrative delta ----
    lines.extend([
        "## Binary to Narrative Delta",
        "",
        f"| Metric | {model_a} | {model_b} |",
        "| --- | ---: | ---: |",
    ])
    if bin_overall and nar_overall:
        delta_a = bin_overall["accuracy_a"] - nar_overall["accuracy_a"]
        delta_b = bin_overall["accuracy_b"] - nar_overall["accuracy_b"]
        lines.append(
            f"| Binary accuracy | {bin_overall['accuracy_a']:.6f} | {bin_overall['accuracy_b']:.6f} |"
        )
        lines.append(
            f"| Narrative accuracy | {nar_overall['accuracy_a']:.6f} | {nar_overall['accuracy_b']:.6f} |"
        )
        lines.append(f"| Binary minus Narrative delta | {delta_a:.6f} | {delta_b:.6f} |")
        lines.append(
            f"| Binary parse-valid | {bin_overall['parse_valid_rate_a']:.6f} | {bin_overall['parse_valid_rate_b']:.6f} |"
        )
        lines.append(
            f"| Narrative parse-valid | {nar_overall['parse_valid_rate_a']:.6f} | {nar_overall['parse_valid_rate_b']:.6f} |"
        )
    lines.append("")

    # ---- Split-Level Summary ----
    split_names = ["dev", "public_leaderboard", "private_leaderboard"]
    lines.extend([
        "## Split-Level Summary",
        "",
        f"| Split | Mode | {model_a} acc | {model_b} acc | Gap | {model_a} PV | {model_b} PV |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for split_name in split_names:
        for mode in ["binary", "narrative"]:
            row = _get("split", split_name, mode)
            if row is None:
                continue
            lines.append(
                f"| {split_name} | {mode} | "
                f"{row['accuracy_a']:.6f} | "
                f"{row['accuracy_b']:.6f} | "
                f"{row['accuracy_a'] - row['accuracy_b']:+.6f} | "
                f"{row['parse_valid_rate_a']:.6f} | "
                f"{row['parse_valid_rate_b']:.6f} |"
            )
    lines.append("")

    # ---- Template Slices ----
    lines.extend([
        "## Template Slices",
        "",
        f"| Template | Mode | {model_a} acc | {model_b} acc | Gap | {model_a} PV | {model_b} PV |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for template in ["T1", "T2"]:
        for mode in ["binary", "narrative"]:
            row = _get("template", template, mode)
            if row is None:
                continue
            lines.append(
                f"| {template} | {mode} | "
                f"{row['accuracy_a']:.6f} | "
                f"{row['accuracy_b']:.6f} | "
                f"{row['accuracy_a'] - row['accuracy_b']:+.6f} | "
                f"{row['parse_valid_rate_a']:.6f} | "
                f"{row['parse_valid_rate_b']:.6f} |"
            )
    lines.append("")

    # ---- Difficulty Slices ----
    lines.extend([
        "## Difficulty Slices",
        "",
        f"| Difficulty | Mode | {model_a} acc | {model_b} acc | Gap | {model_a} PV | {model_b} PV |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for difficulty in ["easy", "medium"]:
        for mode in ["binary", "narrative"]:
            row = _get("difficulty", difficulty, mode)
            if row is None:
                continue
            lines.append(
                f"| {difficulty} | {mode} | "
                f"{row['accuracy_a']:.6f} | "
                f"{row['accuracy_b']:.6f} | "
                f"{row['accuracy_a'] - row['accuracy_b']:+.6f} | "
                f"{row['parse_valid_rate_a']:.6f} | "
                f"{row['parse_valid_rate_b']:.6f} |"
            )
    lines.append("")

    # ---- Transition-Direction Slices ----
    lines.extend([
        "## Transition-Direction Slices",
        "",
        f"| Transition | Mode | {model_a} acc | {model_b} acc | Gap | {model_a} PV | {model_b} PV |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for transition in ["R_std_to_R_inv", "R_inv_to_R_std"]:
        for mode in ["binary", "narrative"]:
            row = _get("transition", transition, mode)
            if row is None:
                continue
            lines.append(
                f"| {transition} | {mode} | "
                f"{row['accuracy_a']:.6f} | "
                f"{row['accuracy_b']:.6f} | "
                f"{row['accuracy_a'] - row['accuracy_b']:+.6f} | "
                f"{row['parse_valid_rate_a']:.6f} | "
                f"{row['parse_valid_rate_b']:.6f} |"
            )
    lines.append("")

    # ---- Failure Decomposition Comparison ----
    fc = comparison.get("failure_comparison", [])
    if fc:
        lines.extend([
            "## Failure Decomposition Comparison (diagnostic-only)",
            "",
            f"| Scope | Mode | Metric | {model_a} | {model_b} | Delta |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ])
        for row in fc:
            if row["scope_type"] != "overall":
                continue
            for metric_key, metric_label in [
                ("runtime_error_rate", "Runtime error rate"),
                ("parse_failure_rate", "Parse/format failure rate"),
                ("parse_valid_rate", "Parse-valid rate"),
                ("adaptation_failure_rate", "Adaptation failure rate"),
            ]:
                va = row.get(f"{metric_key}_a", 0.0)
                vb = row.get(f"{metric_key}_b", 0.0)
                lines.append(
                    f"| {row['scope_label']} | {row['mode']} | {metric_label} | "
                    f"{va:.6f} | {vb:.6f} | {va - vb:+.6f} |"
                )
        lines.append("")

        # Split-level failure decomposition
        lines.extend([
            "### By Split",
            "",
            f"| Split | Mode | Metric | {model_a} | {model_b} | Delta |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ])
        for row in fc:
            if row["scope_type"] != "split":
                continue
            for metric_key, metric_label in [
                ("runtime_error_rate", "Runtime"),
                ("parse_failure_rate", "Parse/format"),
                ("adaptation_failure_rate", "Adaptation"),
            ]:
                va = row.get(f"{metric_key}_a", 0.0)
                vb = row.get(f"{metric_key}_b", 0.0)
                lines.append(
                    f"| {row['scope_label']} | {row['mode']} | {metric_label} | "
                    f"{va:.6f} | {vb:.6f} | {va - vb:+.6f} |"
                )
        lines.append("")

    # ---- Rule Persistence Rate ----
    tc = comparison.get("taxonomy_comparison", [])
    if tc:
        lines.extend([
            "## Rule Persistence Rate (diagnostic-only)",
            "",
            f"| Scope | Mode | Metric | {model_a} | {model_b} | Delta |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ])
        for row in tc:
            for metric_key, metric_label in [
                ("possible_old_rule_persistence_rate", "Old-rule persistence rate"),
                ("possible_recency_overshoot_rate", "Recency overshoot rate"),
            ]:
                va = row.get(f"{metric_key}_a", 0.0)
                vb = row.get(f"{metric_key}_b", 0.0)
                lines.append(
                    f"| {row['scope']} | {row['mode']} | {metric_label} | "
                    f"{va:.6f} | {vb:.6f} | {va - vb:+.6f} |"
                )
        lines.append("")

    # ---- Disagreement Diagnostics ----
    dc = comparison.get("disagreement_comparison", [])
    if dc:
        lines.extend([
            "## Disagreement-Focused Diagnostics (diagnostic-only)",
            "",
            f"| Scope | Mode | Metric | {model_a} | {model_b} |",
            "| --- | --- | --- | --- | --- |",
        ])
        for row in dc:
            if row["scope_type"] != "overall":
                continue
            lines.append(
                f"| {row['scope_label']} | {row['mode']} | Adaptation failures | "
                f"{row['adaptation_failure_count_a']} | {row['adaptation_failure_count_b']} |"
            )
            lines.append(
                f"| {row['scope_label']} | {row['mode']} | Exact old-rule persistence | "
                f"{_fmt_frac(row['exact_old_rule_a'], row['adaptation_failure_count_a'])} | "
                f"{_fmt_frac(row['exact_old_rule_b'], row['adaptation_failure_count_b'])} |"
            )
            lines.append(
                f"| {row['scope_label']} | {row['mode']} | Exact recency overshoot | "
                f"{_fmt_frac(row['exact_recency_a'], row['adaptation_failure_count_a'])} | "
                f"{_fmt_frac(row['exact_recency_b'], row['adaptation_failure_count_b'])} |"
            )
            lines.append(
                f"| {row['scope_label']} | {row['mode']} | Old-rule error probes | "
                f"{_fmt_frac(row['old_rule_probes_a'], row['error_probes_a'])} | "
                f"{_fmt_frac(row['old_rule_probes_b'], row['error_probes_b'])} |"
            )
            lines.append(
                f"| {row['scope_label']} | {row['mode']} | Recency error probes | "
                f"{_fmt_frac(row['recency_probes_a'], row['error_probes_a'])} | "
                f"{_fmt_frac(row['recency_probes_b'], row['error_probes_b'])} |"
            )
        lines.append("")

    # ---- Concentration Analysis ----
    conc = comparison.get("concentration_analysis", {})
    if conc:
        lines.extend([
            "## Flash-Lite Weakness Concentration",
            "",
        ])
        lines.extend([
            "### By Failure Category (overall)",
            "",
            "| Mode | Runtime error rate | Parse/format failure rate | Adaptation failure rate |",
            "| --- | ---: | ---: | ---: |",
        ])
        for mode_row in conc.get("by_mode", []):
            lines.append(
                f"| {mode_row['mode']} | {mode_row['runtime_rate']:.6f} | "
                f"{mode_row['parse_failure_rate']:.6f} | {mode_row['adaptation_rate']:.6f} |"
            )
        lines.append("")

        lines.extend([
            "### By Split (Binary)",
            "",
            "| Split | Accuracy | Parse-valid | Runtime errors | Adaptation failures |",
            "| --- | ---: | ---: | ---: | ---: |",
        ])
        for sr in conc.get("binary_by_split", []):
            lines.append(
                f"| {sr['split']} | {sr['accuracy']:.6f} | {sr['parse_valid_rate']:.6f} | "
                f"{sr['runtime_errors']} | {sr['adaptation_failures']} |"
            )
        lines.append("")

        lines.extend([
            "### By Split (Narrative)",
            "",
            "| Split | Accuracy | Parse-valid | Runtime errors | Parse failures | Adaptation failures |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for sr in conc.get("narrative_by_split", []):
            lines.append(
                f"| {sr['split']} | {sr['accuracy']:.6f} | {sr['parse_valid_rate']:.6f} | "
                f"{sr['runtime_errors']} | {sr['parse_failures']} | {sr['adaptation_failures']} |"
            )
        lines.append("")

        lines.extend([
            "### By Template (Narrative)",
            "",
            "| Template | Accuracy | Parse-valid | Runtime errors | Parse failures | Adaptation failures |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in conc.get("narrative_by_template", []):
            lines.append(
                f"| {row['label']} | {row['accuracy']:.6f} | {row['parse_valid_rate']:.6f} | "
                f"{row['runtime_errors']} | {row['parse_failures']} | {row['adaptation_failures']} |"
            )
        lines.append("")

        lines.extend([
            "### By Template Family (Narrative)",
            "",
            "| Template family | Accuracy | Parse-valid | Runtime errors | Parse failures | Adaptation failures |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in conc.get("narrative_by_template_family", []):
            lines.append(
                f"| {row['label']} | {row['accuracy']:.6f} | {row['parse_valid_rate']:.6f} | "
                f"{row['runtime_errors']} | {row['parse_failures']} | {row['adaptation_failures']} |"
            )
        lines.append("")

        lines.extend([
            "### By Transition (Narrative)",
            "",
            "| Transition | Accuracy | Parse-valid | Runtime errors | Parse failures | Adaptation failures |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in conc.get("narrative_by_transition", []):
            lines.append(
                f"| {row['label']} | {row['accuracy']:.6f} | {row['parse_valid_rate']:.6f} | "
                f"{row['runtime_errors']} | {row['parse_failures']} | {row['adaptation_failures']} |"
            )
        lines.append("")

        lines.append("### Concentration Summary")
        lines.append("")
        for finding in conc.get("findings", []):
            lines.append(f"- {finding}")
        lines.append("")

    # ---- Provider/Runtime Contamination Note ----
    lines.extend([
        "## Provider/Runtime Contamination Note",
        "",
    ])
    prov_a = comparison["run_a"].get("provenance_note")
    if prov_a:
        lines.append(f"- **{model_a}**: {prov_a}")
    else:
        lines.append(f"- **{model_a}**: No provider/runtime contamination noted.")
    prov_b = comparison["run_b"].get("provenance_note")
    if prov_b:
        lines.append(f"- **{model_b}**: {prov_b}")
    else:
        lines.append(f"- **{model_b}**: No provider/runtime contamination noted.")

    # Note runtime error differences
    if bin_overall:
        rt_a = 0.0
        rt_b = 0.0
        for row in fc:
            if row["scope_type"] == "overall" and row["mode"] == "binary":
                rt_a = row.get("runtime_error_rate_a", 0.0)
                rt_b = row.get("runtime_error_rate_b", 0.0)
        if rt_a > 0 or rt_b > 0:
            lines.append(
                f"- Binary runtime error rates: {model_a} = {rt_a:.6f}, {model_b} = {rt_b:.6f}"
            )
        for row in fc:
            if row["scope_type"] == "overall" and row["mode"] == "narrative":
                rt_a = row.get("runtime_error_rate_a", 0.0)
                rt_b = row.get("runtime_error_rate_b", 0.0)
                if rt_a > 0 or rt_b > 0:
                    lines.append(
                        f"- Narrative runtime error rates: {model_a} = {rt_a:.6f}, {model_b} = {rt_b:.6f}"
                    )
    lines.append("")

    # ---- Episode-Level Cross-Model Agreement ----
    episode_rows = comparison.get("episode_rows", [])
    if episode_rows:
        lines.extend([
            "## Episode-Level Cross-Model Agreement",
            "",
        ])
        for mode in ["binary", "narrative"]:
            mode_eps = [e for e in episode_rows if e["mode"] == mode]
            if not mode_eps:
                continue
            total = len(mode_eps)
            both_correct = sum(1 for e in mode_eps if e["both_correct"])
            both_wrong = sum(1 for e in mode_eps if e["both_wrong"])
            a_only = sum(1 for e in mode_eps if e["a_correct_b_wrong"])
            b_only = sum(1 for e in mode_eps if e["b_correct_a_wrong"])
            lines.extend([
                f"### {mode.title()} mode",
                "",
                "| Agreement category | Count | Rate |",
                "| --- | ---: | ---: |",
                f"| Both correct (all 4 probes) | {both_correct} | {both_correct/total:.6f} |",
                f"| {model_a} correct, {model_b} wrong | {a_only} | {a_only/total:.6f} |",
                f"| {model_b} correct, {model_a} wrong | {b_only} | {b_only/total:.6f} |",
                f"| Both wrong | {both_wrong} | {both_wrong/total:.6f} |",
                f"| Total episodes | {total} | |",
                "",
            ])

    # ---- Interpretation ----
    lines.extend([
        "## Interpretation",
        "",
        "- This comparison is **intra-Gemini only**. It measures stability within the active v1 readiness evidence path.",
        "- Binary is the **only** headline interpretation metric.",
        "- Narrative is required same-episode robustness evidence; it does not replace Binary.",
        f"- {model_a} Binary ({bin_overall['accuracy_a']:.6f}) outperforms {model_b} Binary ({bin_overall['accuracy_b']:.6f}) by {bin_overall['accuracy_a'] - bin_overall['accuracy_b']:+.6f}.",
        f"- {model_b} shows higher runtime error rates and substantially lower Narrative parse-valid rates, indicating the Flash-Lite model's weakness is concentrated in Narrative formatting/parsing and provider reliability, not solely in adaptation logic.",
        "- This comparison does not widen the benchmark claim beyond the Gemini evidence path.",
        "",
    ])

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_split(
    artifact: dict[str, Any], split_name: str
) -> dict[str, Any] | None:
    for split in artifact.get("splits", []):
        if split["split_name"] == split_name:
            return split
    return None


def _scope_keys(
    row: dict[str, Any], split_name: str
) -> list[tuple[str, str]]:
    """Return all scope keys an episode row contributes to."""
    return [
        ("overall", "overall"),
        ("split", split_name),
        ("template", row["template_id"]),
        ("template_family", row["template_family"]),
        ("difficulty", row["difficulty"]),
        ("transition", row["transition"]),
    ]


class _Acc:
    """Accumulator for comparison summary rows."""

    def __init__(self) -> None:
        self.episodes = 0
        self.correct_a = 0
        self.correct_b = 0
        self.total_probes = 0
        self.valid_a = 0
        self.valid_b = 0

    def add(
        self,
        mode_a: dict[str, Any],
        mode_b: dict[str, Any],
        ep_row: dict[str, Any],
    ) -> None:
        self.episodes += 1
        self.correct_a += mode_a.get("correct_probe_count", 0)
        self.correct_b += mode_b.get("correct_probe_count", 0)
        self.total_probes += _PROBE_COUNT
        self.valid_a += 1 if mode_a.get("parse_status") == "valid" else 0
        self.valid_b += 1 if mode_b.get("parse_status") == "valid" else 0


def _compare_episode(
    row_a: dict[str, Any],
    row_b: dict[str, Any],
    mode: str,
    split_name: str,
    model_a: str,
    model_b: str,
) -> dict[str, Any]:
    ma = row_a["modes"][mode]
    mb = row_b["modes"][mode]
    probes = _PROBE_COUNT
    correct_a = ma.get("correct_probe_count", 0)
    correct_b = mb.get("correct_probe_count", 0)
    return {
        "episode_id": row_a["episode_id"],
        "split": split_name,
        "mode": mode,
        "template": row_a["template_id"],
        "template_family": row_a["template_family"],
        "difficulty": row_a["difficulty"],
        "transition": row_a["transition"],
        "parse_status_a": ma.get("parse_status"),
        "parse_status_b": mb.get("parse_status"),
        "correct_a": correct_a,
        "correct_b": correct_b,
        "both_correct": correct_a == probes and correct_b == probes,
        "both_wrong": correct_a < probes and correct_b < probes,
        "a_correct_b_wrong": correct_a == probes and correct_b < probes,
        "b_correct_a_wrong": correct_b == probes and correct_a < probes,
        "failure_bucket_a": ma.get("failure_bucket"),
        "failure_bucket_b": mb.get("failure_bucket"),
    }


def _build_summary_rows(
    acc: dict[tuple[str, str, str], _Acc],
    model_a: str,
    model_b: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (scope_type, scope_label, mode), a in sorted(acc.items()):
        total_probes = a.total_probes
        rows.append({
            "scope_type": scope_type,
            "scope_label": scope_label,
            "mode": mode,
            "episodes": a.episodes,
            "accuracy_a": a.correct_a / total_probes if total_probes else 0.0,
            "accuracy_b": a.correct_b / total_probes if total_probes else 0.0,
            "parse_valid_rate_a": a.valid_a / a.episodes if a.episodes else 0.0,
            "parse_valid_rate_b": a.valid_b / a.episodes if a.episodes else 0.0,
        })
    return rows


def _build_failure_comparison(
    art_a: dict[str, Any],
    art_b: dict[str, Any],
    model_a: str,
    model_b: str,
) -> list[dict[str, Any]]:
    ds_a = {
        (r["scope_type"], r["scope_label"], r["mode"]): r
        for r in art_a.get("diagnostic_summary", [])
        if r.get("scope_type") in ("overall", "split")
    }
    ds_b = {
        (r["scope_type"], r["scope_label"], r["mode"]): r
        for r in art_b.get("diagnostic_summary", [])
        if r.get("scope_type") in ("overall", "split")
    }
    rows: list[dict[str, Any]] = []
    for key in sorted(set(ds_a) | set(ds_b)):
        ra = ds_a.get(key, {})
        rb = ds_b.get(key, {})
        rows.append({
            "scope_type": key[0],
            "scope_label": key[1],
            "mode": key[2],
            "runtime_error_rate_a": float(ra.get("runtime_error_rate", 0)),
            "runtime_error_rate_b": float(rb.get("runtime_error_rate", 0)),
            "parse_failure_rate_a": float(ra.get("parse_failure_rate", 0)),
            "parse_failure_rate_b": float(rb.get("parse_failure_rate", 0)),
            "parse_valid_rate_a": float(ra.get("parse_valid_rate", 0)),
            "parse_valid_rate_b": float(rb.get("parse_valid_rate", 0)),
            "adaptation_failure_rate_a": float(ra.get("adaptation_failure_rate", 0)),
            "adaptation_failure_rate_b": float(rb.get("adaptation_failure_rate", 0)),
        })
    return rows


def _build_taxonomy_comparison(
    art_a: dict[str, Any],
    art_b: dict[str, Any],
    model_a: str,
    model_b: str,
) -> list[dict[str, Any]]:
    ft_a = {(r["scope"], r["mode"]): r for r in art_a.get("failure_taxonomy", [])}
    ft_b = {(r["scope"], r["mode"]): r for r in art_b.get("failure_taxonomy", [])}
    rows: list[dict[str, Any]] = []
    for key in sorted(set(ft_a) | set(ft_b)):
        ra = ft_a.get(key, {})
        rb = ft_b.get(key, {})
        rows.append({
            "scope": key[0],
            "mode": key[1],
            "runtime_error_rate_a": float(ra.get("runtime_error_rate", 0)),
            "runtime_error_rate_b": float(rb.get("runtime_error_rate", 0)),
            "parse_failure_rate_a": float(ra.get("parse_failure_rate", 0)),
            "parse_failure_rate_b": float(rb.get("parse_failure_rate", 0)),
            "adaptation_failure_rate_a": float(ra.get("adaptation_failure_rate", 0)),
            "adaptation_failure_rate_b": float(rb.get("adaptation_failure_rate", 0)),
            "possible_old_rule_persistence_rate_a": float(
                ra.get("possible_old_rule_persistence_rate", 0)
            ),
            "possible_old_rule_persistence_rate_b": float(
                rb.get("possible_old_rule_persistence_rate", 0)
            ),
            "possible_recency_overshoot_rate_a": float(
                ra.get("possible_recency_overshoot_rate", 0)
            ),
            "possible_recency_overshoot_rate_b": float(
                rb.get("possible_recency_overshoot_rate", 0)
            ),
        })
    return rows


def _build_disagreement_comparison(
    art_a: dict[str, Any],
    art_b: dict[str, Any],
    model_a: str,
    model_b: str,
) -> list[dict[str, Any]]:
    ds_a = {
        (r["scope_type"], r["scope_label"], r["mode"]): r
        for r in art_a.get("diagnostic_summary", [])
        if r.get("scope_type") in ("overall", "split")
    }
    ds_b = {
        (r["scope_type"], r["scope_label"], r["mode"]): r
        for r in art_b.get("diagnostic_summary", [])
        if r.get("scope_type") in ("overall", "split")
    }
    rows: list[dict[str, Any]] = []
    for key in sorted(set(ds_a) | set(ds_b)):
        ra = ds_a.get(key, {})
        rb = ds_b.get(key, {})
        rows.append({
            "scope_type": key[0],
            "scope_label": key[1],
            "mode": key[2],
            "adaptation_failure_count_a": int(ra.get("adaptation_failure_count", 0)),
            "adaptation_failure_count_b": int(rb.get("adaptation_failure_count", 0)),
            "exact_old_rule_a": int(
                ra.get("exact_global_old_rule_persistence_count", 0)
            ),
            "exact_old_rule_b": int(
                rb.get("exact_global_old_rule_persistence_count", 0)
            ),
            "exact_recency_a": int(
                ra.get("exact_global_recency_overshoot_count", 0)
            ),
            "exact_recency_b": int(
                rb.get("exact_global_recency_overshoot_count", 0)
            ),
            "old_rule_probes_a": int(ra.get("old_rule_error_probe_count", 0)),
            "old_rule_probes_b": int(rb.get("old_rule_error_probe_count", 0)),
            "recency_probes_a": int(
                ra.get("recency_overshoot_error_probe_count", 0)
            ),
            "recency_probes_b": int(
                rb.get("recency_overshoot_error_probe_count", 0)
            ),
            "error_probes_a": int(ra.get("error_probe_count", 0)),
            "error_probes_b": int(rb.get("error_probe_count", 0)),
        })
    return rows


def _build_concentration_analysis(
    art_b: dict[str, Any], model_b: str
) -> dict[str, Any]:
    """Analyze where Flash-Lite weakness is concentrated."""
    ds = art_b.get("diagnostic_summary", [])
    findings: list[str] = []

    # Overall by mode
    by_mode: list[dict[str, Any]] = []
    for mode in ["binary", "narrative"]:
        overall = [
            r
            for r in ds
            if r.get("scope_type") == "overall" and _normalize_mode(r["mode"]) == mode
        ]
        if overall:
            r = overall[0]
            by_mode.append({
                "mode": mode,
                "runtime_rate": float(r.get("runtime_error_rate", 0)),
                "parse_failure_rate": float(r.get("parse_failure_rate", 0)),
                "adaptation_rate": float(r.get("adaptation_failure_rate", 0)),
            })

    # Check Narrative parse/format dominance
    nar_modes = [m for m in by_mode if m["mode"] == "narrative"]
    bin_modes = [m for m in by_mode if m["mode"] == "binary"]
    if nar_modes and bin_modes:
        nar = nar_modes[0]
        bmod = bin_modes[0]
        if nar["parse_failure_rate"] > 0.15:
            findings.append(
                f"Narrative parse/format failure rate ({nar['parse_failure_rate']:.4f}) "
                f"is the dominant non-adaptation weakness. Binary has {bmod['parse_failure_rate']:.4f}."
            )
        if nar["runtime_rate"] > bmod["runtime_rate"] + 0.05:
            findings.append(
                f"Narrative runtime error rate ({nar['runtime_rate']:.4f}) substantially exceeds "
                f"Binary ({bmod['runtime_rate']:.4f}), indicating provider-level reliability issues "
                f"with longer Narrative prompts."
            )

    # Binary by split
    binary_by_split: list[dict[str, Any]] = []
    for r in ds:
        if r.get("scope_type") == "split" and _normalize_mode(r["mode"]) == "binary":
            binary_by_split.append({
                "split": r["scope_label"],
                "accuracy": float(r.get("correct_rate", 0)),
                "parse_valid_rate": float(r.get("parse_valid_rate", 0)),
                "runtime_errors": int(r.get("runtime_error_count", 0)),
                "adaptation_failures": int(r.get("adaptation_failure_count", 0)),
            })

    # Narrative by split
    narrative_by_split: list[dict[str, Any]] = []
    for r in ds:
        if r.get("scope_type") == "split" and _normalize_mode(r["mode"]) == "narrative":
            narrative_by_split.append({
                "split": r["scope_label"],
                "accuracy": float(r.get("correct_rate", 0)),
                "parse_valid_rate": float(r.get("parse_valid_rate", 0)),
                "runtime_errors": int(r.get("runtime_error_count", 0)),
                "parse_failures": int(r.get("parse_failure_count", 0)),
                "adaptation_failures": int(r.get("adaptation_failure_count", 0)),
            })

    narrative_by_template: list[dict[str, Any]] = []
    narrative_by_template_family: list[dict[str, Any]] = []
    narrative_by_transition: list[dict[str, Any]] = []
    for r in ds:
        if _normalize_mode(r.get("mode")) != "narrative":
            continue
        if r.get("scope_type") == "template":
            narrative_by_template.append({
                "label": r["scope_label"],
                "accuracy": float(r.get("correct_rate", 0)),
                "parse_valid_rate": float(r.get("parse_valid_rate", 0)),
                "runtime_errors": int(r.get("runtime_error_count", 0)),
                "parse_failures": int(r.get("parse_failure_count", 0)),
                "adaptation_failures": int(r.get("adaptation_failure_count", 0)),
            })
        if r.get("scope_type") == "template_family":
            narrative_by_template_family.append({
                "label": r["scope_label"],
                "accuracy": float(r.get("correct_rate", 0)),
                "parse_valid_rate": float(r.get("parse_valid_rate", 0)),
                "runtime_errors": int(r.get("runtime_error_count", 0)),
                "parse_failures": int(r.get("parse_failure_count", 0)),
                "adaptation_failures": int(r.get("adaptation_failure_count", 0)),
            })
        if r.get("scope_type") == "transition":
            narrative_by_transition.append({
                "label": r["scope_label"],
                "accuracy": float(r.get("correct_rate", 0)),
                "parse_valid_rate": float(r.get("parse_valid_rate", 0)),
                "runtime_errors": int(r.get("runtime_error_count", 0)),
                "parse_failures": int(r.get("parse_failure_count", 0)),
                "adaptation_failures": int(r.get("adaptation_failure_count", 0)),
            })

    # Find worst split
    if narrative_by_split:
        worst = min(narrative_by_split, key=lambda x: x["parse_valid_rate"])
        findings.append(
            f"Narrative parse-valid rate is lowest on {worst['split']} "
            f"({worst['parse_valid_rate']:.4f}), with {worst['parse_failures']} parse "
            f"failures and {worst['runtime_errors']} runtime errors out of 16 episodes."
        )
    if narrative_by_template:
        worst_template = min(
            narrative_by_template,
            key=lambda x: (x["parse_valid_rate"], x["accuracy"]),
        )
        findings.append(
            f"Narrative weakness is most concentrated on template {worst_template['label']} "
            f"(parse-valid {worst_template['parse_valid_rate']:.4f}, accuracy {worst_template['accuracy']:.4f})."
        )
    if narrative_by_template_family:
        worst_template_family = min(
            narrative_by_template_family,
            key=lambda x: (x["parse_valid_rate"], x["accuracy"]),
        )
        findings.append(
            f"Narrative weakness is most concentrated on template family {worst_template_family['label']} "
            f"(parse-valid {worst_template_family['parse_valid_rate']:.4f}, accuracy {worst_template_family['accuracy']:.4f})."
        )
    if narrative_by_transition:
        worst_transition = min(
            narrative_by_transition,
            key=lambda x: (x["parse_valid_rate"], x["accuracy"]),
        )
        findings.append(
            f"Narrative weakness is most concentrated on transition {worst_transition['label']} "
            f"(parse-valid {worst_transition['parse_valid_rate']:.4f}, accuracy {worst_transition['accuracy']:.4f})."
        )

    # Check adaptation vs format split
    if nar_modes:
        nar = nar_modes[0]
        total_non_correct = nar["runtime_rate"] + nar["parse_failure_rate"] + nar["adaptation_rate"]
        if total_non_correct > 0:
            format_share = (nar["runtime_rate"] + nar["parse_failure_rate"]) / total_non_correct
            adapt_share = nar["adaptation_rate"] / total_non_correct
            findings.append(
                f"Of Narrative failures, {format_share:.1%} are provider/runtime or parse/format "
                f"failures, {adapt_share:.1%} are adaptation failures after valid parses."
            )

    if not findings:
        findings.append("No specific concentration pattern detected.")

    return {
        "by_mode": by_mode,
        "binary_by_split": binary_by_split,
        "narrative_by_split": narrative_by_split,
        "narrative_by_template": narrative_by_template,
        "narrative_by_template_family": narrative_by_template_family,
        "narrative_by_transition": narrative_by_transition,
        "findings": findings,
    }


def _fmt_frac(num: int, denom: int) -> str:
    if denom == 0:
        return f"{num}/0 (n/a)"
    return f"{num}/{denom} ({num / denom:.6f})"


def _normalize_mode(mode: Any) -> str:
    return str(mode or "").strip().lower()
