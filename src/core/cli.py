from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from core.audit import run_release_r15_reaudit, serialize_release_r15_reaudit_report
from core.gemini_panel import (
    DEFAULT_GEMINI_MODEL,
    default_gemini_first_panel_report_path,
    run_gemini_first_panel,
)
from core.kaggle import validate_kaggle_staging_manifest
from core.model_execution import ModelMode
from core.providers.gemini import GeminiConfigurationError
from core.report_outputs import write_text_with_timestamped_snapshot
from core.splits import (
    PARTITIONS,
    assert_no_partition_overlap,
    audit_frozen_splits,
    load_all_frozen_splits,
    load_split_manifest,
)
from core.validate import (
    R13_VALIDITY_GATE,
    run_benchmark_validity_report,
    serialize_benchmark_validity_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ife",
        description="Run local Iron Find Electric benchmark utilities.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    test_parser = subparsers.add_parser(
        "test",
        help="Run the local pytest suite.",
    )
    test_parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Optional extra pytest arguments.",
    )
    test_parser.set_defaults(func=_command_test)

    validity_parser = subparsers.add_parser(
        "validity",
        help="Run the current R13 anti-shortcut validity gate.",
    )
    _add_output_argument(validity_parser)
    validity_parser.set_defaults(func=_command_validity)

    reaudit_parser = subparsers.add_parser(
        "reaudit",
        help="Run the current R15 deterministic re-audit.",
    )
    _add_output_argument(reaudit_parser)
    reaudit_parser.set_defaults(func=_command_reaudit)

    integrity_parser = subparsers.add_parser(
        "integrity",
        help="Validate frozen split and packaging artifact integrity.",
    )
    _add_output_argument(integrity_parser)
    integrity_parser.set_defaults(func=_command_integrity)

    evidence_parser = subparsers.add_parser(
        "evidence-pass",
        help="Run tests, the R13 gate, the R15 re-audit, and integrity checks.",
    )
    evidence_parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Optional extra pytest arguments forwarded to pytest.",
    )
    _add_output_argument(evidence_parser)
    evidence_parser.set_defaults(func=_command_evidence_pass)

    gemini_panel_parser = subparsers.add_parser(
        "gemini-first-panel",
        help="Run the first real Gemini evaluation panel and write a markdown report.",
    )
    gemini_panel_parser.add_argument(
        "--model",
        default=DEFAULT_GEMINI_MODEL,
        help="Gemini model name to run.",
    )
    gemini_panel_parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Markdown report output path. Defaults to the canonical latest path under reports/live/gemini-first-panel/.",
    )
    gemini_panel_parser.add_argument(
        "--include-narrative",
        action="store_true",
        help="Also run the Narrative prompt mode.",
    )
    gemini_panel_parser.set_defaults(func=_command_gemini_first_panel)

    return parser


def entrypoint() -> int:
    return main()


def test_entrypoint() -> int:
    return main(["test"])


def validity_entrypoint() -> int:
    return main(["validity"])


def reaudit_entrypoint() -> int:
    return main(["reaudit"])


def integrity_entrypoint() -> int:
    return main(["integrity"])


def evidence_pass_entrypoint() -> int:
    return main(["evidence-pass"])


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    if args.command in {"test", "evidence-pass"}:
        args.pytest_args.extend(unknown)
    elif unknown:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130


def _add_output_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional JSON output path.",
    )


def _command_test(args: argparse.Namespace) -> int:
    completed = _run_pytest(args.pytest_args)
    return completed.returncode


def _command_validity(args: argparse.Namespace) -> int:
    payload = serialize_benchmark_validity_report(
        run_benchmark_validity_report(gate=R13_VALIDITY_GATE)
    )
    _emit_payload(payload, output_path=args.output)
    return 0


def _command_reaudit(args: argparse.Namespace) -> int:
    payload = serialize_release_r15_reaudit_report(run_release_r15_reaudit())
    _emit_payload(payload, output_path=args.output)
    return 0


def _command_integrity(args: argparse.Namespace) -> int:
    payload = _build_integrity_payload()
    _emit_payload(payload, output_path=args.output)
    return 0


def _command_evidence_pass(args: argparse.Namespace) -> int:
    completed = _run_pytest(args.pytest_args)
    payload: dict[str, Any] = {
        "tests": {
            "passed": completed.returncode == 0,
            "exit_code": completed.returncode,
            "command": [sys.executable, "-m", "pytest", *args.pytest_args],
        }
    }
    if completed.returncode != 0:
        _emit_payload(payload, output_path=args.output)
        return completed.returncode

    payload["validity"] = serialize_benchmark_validity_report(
        run_benchmark_validity_report(gate=R13_VALIDITY_GATE)
    )
    payload["reaudit"] = serialize_release_r15_reaudit_report(run_release_r15_reaudit())
    payload["integrity"] = _build_integrity_payload()
    _emit_payload(payload, output_path=args.output)
    return 0


def _command_gemini_first_panel(args: argparse.Namespace) -> int:
    modes = (
        (ModelMode.BINARY, ModelMode.NARRATIVE)
        if args.include_narrative
        else (ModelMode.BINARY,)
    )
    try:
        report_path = (
            args.report_path
            if args.report_path is not None
            else default_gemini_first_panel_report_path(
                include_narrative=args.include_narrative
            )
        )
        artifacts = run_gemini_first_panel(
            model_name=args.model,
            report_path=report_path,
            modes=modes,
        )
    except GeminiConfigurationError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    payload = {
        "release_id": artifacts.release_report.release_id,
        "provider_name": artifacts.provider_name,
        "model_name": artifacts.model_name,
        "prompt_modes": [mode.value for mode in artifacts.prompt_modes],
        "report_path": str(artifacts.report_path),
        "artifact_path": (
            str(artifacts.artifact_path)
            if artifacts.artifact_path is not None
            else None
        ),
        "snapshot_report_path": (
            str(artifacts.snapshot_report_path)
            if artifacts.snapshot_report_path is not None
            else None
        ),
        "snapshot_artifact_path": (
            str(artifacts.snapshot_artifact_path)
            if artifacts.snapshot_artifact_path is not None
            else None
        ),
    }
    print(json.dumps(payload, indent=2))
    return 0


def _run_pytest(pytest_args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pytest", *pytest_args],
        cwd=_repo_root(),
        check=False,
        text=True,
    )


def _build_integrity_payload() -> dict[str, Any]:
    split_records = load_all_frozen_splits()
    assert_no_partition_overlap(split_records)
    split_audit = audit_frozen_splits(split_records)
    validate_kaggle_staging_manifest(repo_root=_repo_root())

    manifests = {
        partition: {
            "manifest_version": manifest.manifest_version,
            "seed_bank_version": manifest.seed_bank_version,
            "episode_split": manifest.episode_split.value,
            "seed_count": len(manifest.seeds),
            "loaded_episode_count": len(split_records[partition]),
        }
        for partition in PARTITIONS
        for manifest in (load_split_manifest(partition),)
    }

    return {
        "overlap_check_passed": True,
        "kaggle_manifest_valid": True,
        "split_counts": {
            partition: len(records) for partition, records in split_records.items()
        },
        "manifests": manifests,
        "audit_issue_count": len(split_audit.issues),
        "audit_issues": [
            {"code": issue.code, "message": issue.message}
            for issue in split_audit.issues
        ],
    }


def _emit_payload(payload: dict[str, Any], *, output_path: Path | None) -> None:
    text = json.dumps(payload, indent=2) + "\n"
    if output_path is not None:
        write_text_with_timestamped_snapshot(output_path, text)
    print(text, end="")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]
