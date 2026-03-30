from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from maintainer.audit import run_release_r15_reaudit, serialize_release_r15_reaudit_report
from maintainer.contract_audit import run_contract_audit
from maintainer.report_outputs import write_text_with_timestamped_snapshot
from maintainer.validate import (
    R13_VALIDITY_GATE,
    run_benchmark_validity_report,
    serialize_benchmark_validity_report,
)
from core.kaggle import validate_kaggle_staging_manifest
from core.splits import (
    PARTITIONS,
    assert_no_partition_overlap,
    audit_frozen_splits,
    load_all_frozen_splits,
    load_split_manifest,
)

_PRIVATE_REQUIRED_ERROR = (
    "Error: private_leaderboard split is not mounted.\n"
    "\n"
    "This command requires the private evaluation dataset (private_episodes.json).\n"
    "The private_leaderboard split is used by the R13 validity gate, R15 re-audit,\n"
    "and split integrity checks.\n"
    "\n"
    "To mount the private split locally:\n"
    "  export {env_var}=/path/to/private-dataset\n"
    "\n"
    "  The directory must contain private_episodes.json, generated offline with:\n"
    "    python scripts/generate_private_split_artifact.py --help\n"
    "\n"
    "  See packaging/kaggle/PRIVATE_SPLIT_RUNBOOK.md for the full workflow.\n"
    "\n"
    "Commands that work without the private split (public-only environment):\n"
    "  make test            -- run the test suite\n"
    "  make contract-audit  -- validate public artifact contracts\n"
    "  make doctor          -- report environment status\n"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ruleshift-benchmark",
        description="Run local RuleShift Benchmark utilities.",
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

    contract_audit_parser = subparsers.add_parser(
        "contract-audit",
        help="Run the P0 public artifact contract audit.",
    )
    contract_audit_parser.add_argument(
        "--run-artifact",
        type=Path,
        default=None,
        help="Path to a run artifact (artifact.json) to validate.",
    )
    _add_output_argument(contract_audit_parser)
    contract_audit_parser.set_defaults(func=_command_contract_audit)

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Report environment status: private split availability and command matrix.",
    )
    doctor_parser.set_defaults(func=_command_doctor)

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


def contract_audit_entrypoint() -> int:
    return main(["contract-audit"])


def doctor_entrypoint() -> int:
    return main(["doctor"])


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
    if (rc := _preflight_private_split_or_exit()) is not None:
        return rc
    payload = serialize_benchmark_validity_report(
        run_benchmark_validity_report(gate=R13_VALIDITY_GATE)
    )
    _emit_payload(payload, output_path=args.output)
    return 0


def _command_reaudit(args: argparse.Namespace) -> int:
    if (rc := _preflight_private_split_or_exit()) is not None:
        return rc
    payload = serialize_release_r15_reaudit_report(run_release_r15_reaudit())
    _emit_payload(payload, output_path=args.output)
    return 0


def _command_integrity(args: argparse.Namespace) -> int:
    if (rc := _preflight_private_split_or_exit()) is not None:
        return rc
    payload = _build_integrity_payload()
    _emit_payload(payload, output_path=args.output)
    return 0


def _command_evidence_pass(args: argparse.Namespace) -> int:
    if (rc := _preflight_private_split_or_exit()) is not None:
        return rc
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


def _command_doctor(args: argparse.Namespace) -> int:
    from core.private_split import (
        PRIVATE_DATASET_ROOT_ENV_VAR,
        discover_private_dataset_root,
    )

    private_root = discover_private_dataset_root()
    lines: list[str] = ["RuleShift Benchmark — Environment Check", ""]

    if private_root is not None:
        lines.append(f"Private split:  mounted at {private_root}")
        private_status = "ok"
    else:
        lines.append("Private split:  not mounted (public-only environment)")
        lines.append(
            f"  To enable:    export {PRIVATE_DATASET_ROOT_ENV_VAR}=/path/to/private-dataset"
        )
        lines.append(
            "  See packaging/kaggle/PRIVATE_SPLIT_RUNBOOK.md for the generation workflow."
        )
        private_status = "unavailable"

    lines.append("")
    lines.append(f"{'Command':<22}  {'Requirement':<16}  Status")
    lines.append(f"{'-' * 22}  {'-' * 16}  {'-' * 11}")

    _COMMAND_MATRIX = (
        ("make test",           "public-safe",  "ok"),
        ("make contract-audit", "public-safe",  "ok"),
        ("make doctor",         "public-safe",  "ok"),
        ("make validity",       "private split", private_status),
        ("make reaudit",        "private split", private_status),
        ("make integrity",      "private split", private_status),
        ("make evidence-pass",  "private split", private_status),
    )
    for cmd, req, status in _COMMAND_MATRIX:
        lines.append(f"{cmd:<22}  {req:<16}  {status}")

    print("\n".join(lines))
    return 0


def _command_contract_audit(args: argparse.Namespace) -> int:
    payload = run_contract_audit(
        repo_root=_repo_root(),
        run_artifact_path=args.run_artifact,
    )
    _emit_payload(payload, output_path=args.output)
    return 0 if payload["passed"] else 1


def _preflight_private_split_or_exit() -> int | None:
    """Return 1 with a clear error if the private split is not mounted, else None."""
    from core.private_split import PRIVATE_DATASET_ROOT_ENV_VAR, discover_private_dataset_root

    if discover_private_dataset_root() is not None:
        return None
    print(
        _PRIVATE_REQUIRED_ERROR.format(env_var=PRIVATE_DATASET_ROOT_ENV_VAR),
        file=sys.stderr,
    )
    return 1


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


if __name__ == "__main__":
    raise SystemExit(main())
