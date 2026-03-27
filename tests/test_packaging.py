from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tomllib

from core.kaggle import (
    BinaryResponse,
    Label,
    load_kaggle_staging_manifest,
    normalize_binary_response,
    normalize_narrative_response,
    score_episode,
    validate_kaggle_staging_manifest,
)
from tasks.ruleshift_benchmark.schema import (
    DIFFICULTY_VERSION,
    GENERATOR_VERSION,
    SPEC_VERSION,
    TEMPLATE_SET_VERSION,
)
from core.splits import MANIFEST_VERSION, PARTITIONS, load_split_manifest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_KAGGLE_DIR = _REPO_ROOT / "packaging" / "kaggle"
_PYPROJECT_PATH = _REPO_ROOT / "pyproject.toml"
_KBENCH_NOTEBOOK_PATH = _KAGGLE_DIR / "ruleshift_notebook_task.ipynb"
_KERNEL_METADATA_PATH = _KAGGLE_DIR / "kernel-metadata.json"
_CARD_PATH = _KAGGLE_DIR / "BENCHMARK_CARD.md"


def _read_notebook_sources(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", ()))
        for cell in notebook["cells"]
    )


def _load_pyproject() -> dict[str, object]:
    return tomllib.loads(_PYPROJECT_PATH.read_text(encoding="utf-8"))


def _load_kernel_metadata() -> dict[str, object]:
    return json.loads(_KERNEL_METADATA_PATH.read_text(encoding="utf-8"))


def test_kaggle_staging_manifest_resolves_current_frozen_artifacts():
    manifest = load_kaggle_staging_manifest()

    validate_kaggle_staging_manifest()

    assert manifest["bundle_version"] == "R16"
    assert manifest["task_id"] == "ruleshift_benchmark_v1"
    assert manifest["task_name"] == "RuleShift Benchmark v1"

    benchmark_versions = manifest["benchmark_versions"]
    assert benchmark_versions == {
        "manifest_version": MANIFEST_VERSION,
        "spec_version": SPEC_VERSION,
        "generator_version": GENERATOR_VERSION,
        "template_set_version": TEMPLATE_SET_VERSION,
        "difficulty_version": DIFFICULTY_VERSION,
    }

    _RUNTIME_PARTITIONS = ("dev", "public_leaderboard")
    assert tuple(manifest["frozen_split_manifests"]) == _RUNTIME_PARTITIONS
    assert "private_leaderboard" not in manifest["frozen_split_manifests"]
    assert tuple(manifest["entry_points"]) == ("kbench_notebook", "kernel_metadata")
    assert manifest["current_emitted_difficulty_labels"] == ["easy", "medium", "hard"]
    assert manifest["reserved_difficulty_labels"] == []

    for partition in _RUNTIME_PARTITIONS:
        artifact = manifest["frozen_split_manifests"][partition]
        manifest_path = _REPO_ROOT / artifact["path"]
        split_manifest = load_split_manifest(partition)

        assert manifest_path.is_file()
        assert manifest_path.name == f"{partition}.json"
        assert artifact["manifest_version"] == split_manifest.manifest_version
        assert artifact["seed_bank_version"] == split_manifest.seed_bank_version
        assert artifact["episode_split"] == split_manifest.episode_split.value


def test_official_kaggle_submission_flow_is_consistent_across_surface():
    manifest = load_kaggle_staging_manifest()
    kernel_metadata = _load_kernel_metadata()
    readme_text = _REPO_ROOT.joinpath("README.md").read_text(encoding="utf-8")
    card_text = _CARD_PATH.read_text(encoding="utf-8")

    official_notebook_relpath = "packaging/kaggle/ruleshift_notebook_task.ipynb"
    official_notebook_name = "ruleshift_notebook_task.ipynb"
    official_entry_points = tuple(
        artifact["path"]
        for artifact in manifest["entry_points"].values()
        if artifact["path"].endswith(".ipynb")
    )

    assert manifest["entry_points"]["kbench_notebook"]["path"] == official_notebook_relpath
    assert manifest["entry_points"]["kernel_metadata"]["path"] == "packaging/kaggle/kernel-metadata.json"
    assert official_entry_points == (official_notebook_relpath,)
    assert kernel_metadata["code_file"] == official_notebook_name
    assert official_notebook_relpath in readme_text
    assert official_notebook_relpath in card_text


def test_non_official_packaged_paths_are_removed():
    card_text = _CARD_PATH.read_text(encoding="utf-8")

    assert "staging" not in card_text.lower()
    assert not (_KAGGLE_DIR / "staging" / "ruleshift_benchmark_v1_kaggle_staging.ipynb").exists()
    assert not (_KAGGLE_DIR / "staging").exists()
    assert not (_KAGGLE_DIR / "archive").exists()


def test_kaggle_directory_layout_contains_only_official_entry_surfaces():
    top_level_files = sorted(path.name for path in _KAGGLE_DIR.iterdir() if path.is_file())

    assert top_level_files == [
        "BENCHMARK_CARD.md",
        "DEPLOY_RUNBOOK.md",
        "PRIVATE_SPLIT_RUNBOOK.md",
        "frozen_artifacts_manifest.json",
        "kernel-metadata.json",
        "ruleshift_notebook_task.ipynb",
    ]


def test_kaggle_runbook_documents_the_minimum_runtime_subset():
    readme_text = _REPO_ROOT.joinpath("README.md").read_text(encoding="utf-8")

    required_runtime_paths = (
        "packaging/kaggle/ruleshift_notebook_task.ipynb",
        "src/",
        "src/frozen_splits/dev.json",
        "src/frozen_splits/public_leaderboard.json",
    )

    for path in required_runtime_paths:
        assert path in readme_text
    assert "src/frozen_splits/private_leaderboard.json" not in readme_text
    assert "private_episodes.json" in readme_text


def test_packaging_docs_describe_private_split_as_mount_only():
    readme_text = _REPO_ROOT.joinpath("README.md").read_text(encoding="utf-8")
    card_text = _CARD_PATH.read_text(encoding="utf-8")
    notebook_text = _read_notebook_sources(_KBENCH_NOTEBOOK_PATH)

    assert "authorized private dataset mount" in readme_text
    assert "private dataset mount" in card_text
    assert "src/frozen_splits/private_leaderboard.json" not in card_text
    assert "packaging/kaggle/private/private_episodes.json" not in notebook_text
    assert "resolve_private_dataset_root" in notebook_text


def test_packaging_docs_describe_separate_private_flow():
    runbook_text = (_KAGGLE_DIR / "PRIVATE_SPLIT_RUNBOOK.md").read_text(encoding="utf-8")

    assert "scripts/generate_private_split_artifact.py" in runbook_text
    assert "scripts/cd/build_private_dataset_package.py" in runbook_text
    assert "scripts/check_public_private_isolation.py" in runbook_text
    assert "Never commit the artifact" in runbook_text


def test_active_docs_label_frozen_artifacts_manifest_by_runtime_role():
    texts = (
        _REPO_ROOT.joinpath("README.md").read_text(encoding="utf-8"),
        _CARD_PATH.read_text(encoding="utf-8"),
    )
    joined = "\n".join(texts).lower()

    assert "kaggle runtime-contract manifest" in joined
    assert "frozen artifacts index" not in joined


def test_pyproject_exposes_local_console_entrypoints():
    pyproject = _load_pyproject()

    assert pyproject["project"]["name"] == "ruleshift-benchmark"
    assert pyproject["project"]["scripts"] == {
        "ruleshift-benchmark": "core.cli:entrypoint",
        "ruleshift-benchmark-test": "core.cli:test_entrypoint",
        "ruleshift-benchmark-validity": "core.cli:validity_entrypoint",
        "ruleshift-benchmark-reaudit": "core.cli:reaudit_entrypoint",
        "ruleshift-benchmark-integrity": "core.cli:integrity_entrypoint",
        "ruleshift-benchmark-evidence-pass": "core.cli:evidence_pass_entrypoint",
        "ruleshift-benchmark-contract-audit": "core.cli:contract_audit_entrypoint",
    }
    assert pyproject["project"]["dependencies"] == []
    assert "optional-dependencies" not in pyproject["project"]


def test_benchmark_card_matches_current_implementation_state():
    text = _CARD_PATH.read_text(encoding="utf-8")

    assert "RuleShift Benchmark v1" in text
    assert "narrow Executive Functions benchmark for cognitive flexibility" in text
    assert "controlled substrate" in text
    assert "frozen episodes" in text
    assert "Binary" in text
    assert "only leaderboard-primary" in text
    assert "Narrative" in text
    assert "same frozen episodes and probe targets as Binary" in text
    assert "only the final four labels are scored" in text
    assert "Post-shift Probe Accuracy" in text
    assert "Binary task" in text
    assert "`random`" in text
    assert "`never_update`" in text
    assert "`last_evidence`" in text
    assert "`physics_prior`" in text
    assert "`template_position`" in text
    assert "easy" in text
    assert "medium" in text
    assert "hard" in text
    assert "template_family" in text
    assert "Invariance reporting is diagnostic-only" in text
    assert "recency shortcut was materially reduced" in text
    assert "R13 anti-shortcut validity gate" in text
    assert "R15 empirical re-audit" in text


def test_docs_do_not_reference_removed_parallel_sources():
    readme_text = _REPO_ROOT.joinpath("README.md").read_text(encoding="utf-8")
    card_text = _CARD_PATH.read_text(encoding="utf-8")
    runbook_text = (_KAGGLE_DIR / "PRIVATE_SPLIT_RUNBOOK.md").read_text(encoding="utf-8")

    for text in (readme_text, card_text, runbook_text):
        assert "FROZEN_BENCHMARK_SPEC.md" not in text
        assert "packaging/kaggle/README.md" not in text
        assert "src/README.md" not in text
        assert "deploy/kaggle-runtime" not in text
        assert "deploy/kaggle-notebook" not in text
        assert "scripts/build_deploy.py" not in text


def test_active_docs_identify_one_official_packaged_readiness_anchor():
    texts = (
        _REPO_ROOT.joinpath("README.md").read_text(encoding="utf-8"),
        _CARD_PATH.read_text(encoding="utf-8"),
    )
    joined = "\n".join(texts)
    lowered = joined.lower()

    assert "current readiness evidence" not in lowered
    assert "reports/live/gemini-first-panel" not in joined
    assert "m1_binary_vs_narrative_robustness_report.md" not in joined
    assert "supporting comparison material" not in lowered


def test_packaging_docs_do_not_claim_unsupported_features():
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            _REPO_ROOT / "README.md",
            _CARD_PATH,
            _KAGGLE_DIR / "PRIVATE_SPLIT_RUNBOOK.md",
        )
    )
    lowered = text.lower()

    assert "does **not** claim" in text

    explicit_non_claims = (
        "physics skill",
        "full executive-function decomposition",
        "online detection latency",
        "switch cost",
        "recovery length",
        "immediate post-shift drop",
        "broad agi capability",
    )
    for phrase in explicit_non_claims:
        assert phrase in lowered

    assert "post-shift probe accuracy as the sole headline metric" in lowered
    assert "binary as the only leaderboard-primary path" in lowered
    assert "narrative" in lowered and "supplementary same-episode robustness evidence" in lowered
    assert "only the final four labels are scored" in lowered

    forbidden_positive_claims = (
        "hard leaderboard slice",
        "emitted hard slice",
        "hard episodes are included",
        "submission score claim",
        "measures switch cost",
        "measures online detection latency",
        "full executive-function decomposition benchmark",
        "broad adaptation benchmark",
        "physics benchmark",
    )

    for phrase in forbidden_positive_claims:
        assert phrase not in lowered


def test_kaggle_packaging_text_stays_focused_on_benchmark_runtime_path():
    text = _REPO_ROOT.joinpath("README.md").read_text(encoding="utf-8").lower()

    assert "optional local provider runs" not in text
    assert ".[gemini]" not in text
    assert ".[anthropic]" not in text
    assert ".[openai]" not in text
    assert "production dependency installation" not in text


def test_official_kbench_notebook_imports_package_owned_benchmark_logic():
    sources = _read_notebook_sources(_KBENCH_NOTEBOOK_PATH)

    assert (
        "from core.kaggle import BinaryResponse, Label, normalize_binary_response, "
        "normalize_narrative_response, score_episode"
    ) in sources
    assert "load_kaggle_staging_manifest" not in sources
    assert "validate_kaggle_staging_manifest" not in sources
    assert "resolve_kaggle_artifact_path" not in sources
    assert "release_r13_validity_report.json" not in sources
    assert "release_r15_reaudit_report.json" not in sources
    assert "def normalize_binary_response(" not in sources
    assert "def normalize_narrative_response(" not in sources
    assert "def score_binary_episode(" not in sources
    assert "class Label(" not in sources
    assert "class BinaryResponse:" not in sources


def test_kaggle_package_helpers_preserve_binary_and_narrative_scoring_contract():
    structured = BinaryResponse(
        Label.attract,
        Label.repel,
        Label.repel,
        Label.attract,
    )

    assert normalize_binary_response(structured) == (
        "attract",
        "repel",
        "repel",
        "attract",
    )
    assert normalize_binary_response("attract, repel, repel, attract") == (
        "attract",
        "repel",
        "repel",
        "attract",
    )
    import json as _json
    assert normalize_narrative_response(
        "\n".join(
            (
                "rule_before: pre-shift rule",
                "shift_evidence: post-shift observations contradict",
                "rule_after: post-shift rule",
                "final_decision: attract, repel, repel, attract",
            )
        )
    ) == ("attract", "repel", "repel", "attract")
    # Free-form text without the contract is rejected by the parser.
    assert normalize_narrative_response(
        "Reasoning about the later rule.\nattract, repel, repel, attract"
    ) is None
    assert score_episode(
        ("attract", "repel", "repel", "attract"),
        ("attract", "repel", "repel", "attract"),
    ) == (4, 4)
    assert score_episode(None, ("attract", "repel", "repel", "attract")) == (0, 4)


def test_public_packaging_does_not_reference_private_assets():
    """Guardrail: frozen_artifacts_manifest.json must not reference any private-only asset."""
    private_only_filenames = (
        "private_leaderboard.json",
        "private_episodes.json",
    )
    manifest_text = (_KAGGLE_DIR / "frozen_artifacts_manifest.json").read_text(encoding="utf-8")
    for name in private_only_filenames:
        assert name not in manifest_text, (
            f"Public packaging artifact frozen_artifacts_manifest.json references "
            f"private-only asset '{name}'"
        )

def test_repo_structure_uses_single_kaggle_packaging_path():
    assert not (_REPO_ROOT / "deploy").exists()
    assert not (_REPO_ROOT / "scripts" / "build_deploy.py").exists()


def test_private_dataset_packaging_script_builds_private_only_package(tmp_path):
    seeds_path = tmp_path / "private_seeds.json"
    seeds_path.write_text(json.dumps([33000, 33001, 33002]), encoding="utf-8")
    artifact_root = tmp_path / "private-artifact"
    artifact_path = artifact_root / "private_episodes.json"
    private_package_dir = tmp_path / "private-package"

    generate = subprocess.run(
        [
            sys.executable,
            "scripts/generate_private_split_artifact.py",
            "--benchmark-version",
            "R14",
            "--seeds-file",
            str(seeds_path),
            "--output",
            str(artifact_path),
        ],
        cwd=_REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert generate.returncode == 0, generate.stderr or generate.stdout

    packaged = subprocess.run(
        [
            sys.executable,
            "scripts/cd/build_private_dataset_package.py",
            "--artifact",
            str(artifact_path),
            "--output-dir",
            str(private_package_dir),
            "--dataset-id",
            "owner/ruleshift-private-runtime",
        ],
        cwd=_REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert packaged.returncode == 0, packaged.stderr or packaged.stdout
    assert Path(packaged.stdout.strip()) == private_package_dir
    assert (private_package_dir / "private_episodes.json").is_file()
    assert (private_package_dir / "dataset-metadata.json").is_file()
    assert not (private_package_dir / "src").exists()
    assert not (private_package_dir / "packaging").exists()


def test_private_dataset_packaging_script_rejects_repo_output(tmp_path):
    seeds_path = tmp_path / "private_seeds.json"
    seeds_path.write_text(json.dumps([34000, 34001]), encoding="utf-8")
    artifact_path = tmp_path / "private_episodes.json"

    generate = subprocess.run(
        [
            sys.executable,
            "scripts/generate_private_split_artifact.py",
            "--benchmark-version",
            "R14",
            "--seeds-file",
            str(seeds_path),
            "--output",
            str(artifact_path),
        ],
        cwd=_REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert generate.returncode == 0, generate.stderr or generate.stdout

    packaged = subprocess.run(
        [
            sys.executable,
            "scripts/cd/build_private_dataset_package.py",
            "--artifact",
            str(artifact_path),
            "--output-dir",
            str(_REPO_ROOT / "deploy" / "private-runtime"),
            "--dataset-id",
            "owner/ruleshift-private-runtime",
        ],
        cwd=_REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert packaged.returncode != 0
    assert "outside the public repository tree" in (packaged.stderr or packaged.stdout)
