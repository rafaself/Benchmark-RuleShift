from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tomllib

from kaggle import (
    BinaryResponse,
    Label,
    load_kaggle_staging_manifest,
    normalize_binary_response,
    normalize_narrative_response,
    score_episode,
    validate_kaggle_staging_manifest,
)
from schema import DIFFICULTY_VERSION, GENERATOR_VERSION, SPEC_VERSION, TEMPLATE_SET_VERSION
from splits import MANIFEST_VERSION, PARTITIONS, load_split_manifest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_KAGGLE_DIR = _REPO_ROOT / "packaging" / "kaggle"
_PYPROJECT_PATH = _REPO_ROOT / "pyproject.toml"
_KBENCH_NOTEBOOK_PATH = _KAGGLE_DIR / "ruleshift_notebook_task.ipynb"
_STAGING_DIR = _KAGGLE_DIR / "staging"
_ARCHIVE_DIR = _KAGGLE_DIR / "archive"
_STAGING_NOTEBOOK_PATH = _STAGING_DIR / "ruleshift_benchmark_v1_kaggle_staging.ipynb"
_KERNEL_METADATA_PATH = _KAGGLE_DIR / "kernel-metadata.json"
_CARD_PATH = _KAGGLE_DIR / "BENCHMARK_CARD.md"
_USAGE_PATH = _KAGGLE_DIR / "README.md"
_PACKAGING_NOTE_PATH = _ARCHIVE_DIR / "PACKAGING_NOTE.md"


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
    assert manifest["current_emitted_difficulty_labels"] == ["easy", "medium"]
    assert manifest["reserved_difficulty_labels"] == ["hard"]

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
    usage_text = _USAGE_PATH.read_text(encoding="utf-8")
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
    assert official_notebook_name in usage_text
    assert official_notebook_relpath in card_text
    assert "No other notebook or local runtime path is an official Kaggle leaderboard submission surface." in usage_text


def test_non_official_packaged_paths_are_explicitly_marked_non_active():
    usage_text = _USAGE_PATH.read_text(encoding="utf-8")
    card_text = _CARD_PATH.read_text(encoding="utf-8")
    packaging_note_text = _PACKAGING_NOTE_PATH.read_text(encoding="utf-8")

    assert "`staging/ruleshift_benchmark_v1_kaggle_staging.ipynb`: optional package-validation and dry-run notebook" in usage_text
    assert "staging-only" in card_text
    assert "ARCHIVE RELEASE NOTE" in packaging_note_text
    assert "not an authoritative benchmark contract or an operational runbook" in packaging_note_text


def test_kaggle_directory_layout_separates_active_staging_and_archive_files():
    top_level_files = sorted(path.name for path in _KAGGLE_DIR.iterdir() if path.is_file())

    assert top_level_files == [
        "BENCHMARK_CARD.md",
        "PRIVATE_SPLIT_RUNBOOK.md",
        "README.md",
        "frozen_artifacts_manifest.json",
        "kernel-metadata.json",
        "ruleshift_notebook_task.ipynb",
    ]
    assert _STAGING_NOTEBOOK_PATH.is_file()
    assert _PACKAGING_NOTE_PATH.is_file()


def test_kaggle_runbook_documents_the_minimum_runtime_subset():
    usage_text = _USAGE_PATH.read_text(encoding="utf-8")

    required_runtime_paths = (
        "packaging/kaggle/ruleshift_notebook_task.ipynb",
        "packaging/kaggle/kernel-metadata.json",
        "packaging/kaggle/frozen_artifacts_manifest.json",
        "src/",
        "src/frozen_splits/dev.json",
        "src/frozen_splits/public_leaderboard.json",
    )
    non_runtime_paths = (
        "BENCHMARK_CARD.md",
        "this runbook",
        "staging notebooks",
        "archive files",
        "reports/",
        "tests/fixtures/",
    )

    assert "minimum packaged subset" in usage_text.lower()
    assert "kaggle runtime-contract manifest" in usage_text.lower()
    assert "upload the minimum runtime package needed by the official notebook" in usage_text.lower()
    for path in required_runtime_paths:
        assert path in usage_text
    for path in non_runtime_paths:
        assert path in usage_text
    assert "keeping `src/`, `tests/fixtures/`, `reports/`, and `packaging/kaggle/` together" not in usage_text
    assert "src/frozen_splits/private_leaderboard.json" not in usage_text
    assert "packaging/kaggle/private/private_episodes.json" not in usage_text


def test_packaging_docs_describe_private_split_as_mount_only():
    usage_text = _USAGE_PATH.read_text(encoding="utf-8")
    card_text = _CARD_PATH.read_text(encoding="utf-8")
    notebook_text = _read_notebook_sources(_KBENCH_NOTEBOOK_PATH)

    assert "authorized private dataset mount" in usage_text
    assert "repo-local default" not in usage_text
    assert "src/frozen_splits/private_leaderboard.json" not in card_text
    assert "public partitions from the stored seed banks" in card_text
    assert "packaging/kaggle/private/private_episodes.json" not in notebook_text
    assert "resolve_private_dataset_root" in notebook_text


def test_packaging_docs_describe_separate_private_flow():
    usage_text = _USAGE_PATH.read_text(encoding="utf-8")

    assert "Public And Private Packaging Boundary" in usage_text
    assert "scripts/build_deploy.py" in usage_text
    assert "scripts/generate_private_split_artifact.py" in usage_text
    assert "scripts/cd/build_private_dataset_package.py" in usage_text
    assert "must never package, version, or upload `private_episodes.json`" in usage_text


def test_active_docs_label_frozen_artifacts_manifest_by_runtime_role():
    texts = (
        _REPO_ROOT.joinpath("README.md").read_text(encoding="utf-8"),
        _USAGE_PATH.read_text(encoding="utf-8"),
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
    assert pyproject["project"]["optional-dependencies"]["gemini"] == [
        "google-genai>=1,<2",
    ]
    assert pyproject["project"]["optional-dependencies"]["anthropic"] == [
        "anthropic>=0.40,<1",
    ]
    assert pyproject["project"]["optional-dependencies"]["openai"] == [
        "openai>=1.0,<2",
    ]


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
    assert "reserved and not emitted" in text
    assert "recency shortcut was materially reduced" in text
    assert "R13 anti-shortcut validity gate" in text
    assert "R15 empirical re-audit" in text


def test_active_docs_identify_one_official_packaged_readiness_anchor():
    texts = (
        _REPO_ROOT.joinpath("README.md").read_text(encoding="utf-8"),
        _USAGE_PATH.read_text(encoding="utf-8"),
        _CARD_PATH.read_text(encoding="utf-8"),
    )
    joined = "\n".join(texts)
    lowered = joined.lower()
    anchor_path = "reports/m1_binary_vs_narrative_robustness_report.md"
    source_report_path = (
        "reports/live/gemini-first-panel/binary-vs-narrative/history/"
        "report__20260323_120000.md"
    )
    latest_report_path = "reports/live/gemini-first-panel/binary-vs-narrative/latest/report.md"
    comparison_report_path = "reports/live/gemini-first-panel/comparison/latest/report.md"

    assert "gemini-2.5-flash" in joined
    assert joined.count(anchor_path) >= 3
    assert joined.count(source_report_path) >= 3
    assert joined.count(latest_report_path) >= 1
    assert joined.count(comparison_report_path) >= 1
    assert "single current packaged readiness anchor" in lowered
    assert "not a second active readiness anchor" in lowered
    assert "supporting comparison material" in lowered
    assert "current paired Gemini Flash-Lite run is canonical".lower() not in lowered
    assert "direct Flash vs Flash-Lite comparison is canonical".lower() not in lowered
    assert f"current Gemini evidence anchor: `{latest_report_path}`".lower() not in lowered


def test_packaging_docs_do_not_claim_unsupported_features():
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (_REPO_ROOT / "README.md", _CARD_PATH, _USAGE_PATH)
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


def test_kaggle_packaging_text_keeps_optional_provider_sdks_out_of_base_path():
    text = _USAGE_PATH.read_text(encoding="utf-8").lower()

    assert "optional local-only provider sdks" in text
    assert "no production dependency installation is needed" in text


def test_kaggle_staging_path_does_not_depend_on_openai_runtime():
    notebook_text = _read_notebook_sources(_STAGING_NOTEBOOK_PATH).lower()
    usage_text = _USAGE_PATH.read_text(encoding="utf-8").lower()

    assert "openai_api_key" not in notebook_text
    assert "ife openai-panel" not in notebook_text
    assert "openai_api_key" not in usage_text


def test_official_kbench_notebook_imports_package_owned_benchmark_logic():
    sources = _read_notebook_sources(_KBENCH_NOTEBOOK_PATH)

    assert "from kaggle import BinaryResponse, Label, normalize_binary_response, normalize_narrative_response, score_episode" in sources
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
    assert normalize_narrative_response(
        "Reasoning about the later rule.\nattract, repel, repel, attract"
    ) == ("attract", "repel", "repel", "attract")
    assert score_episode(
        ("attract", "repel", "repel", "attract"),
        ("attract", "repel", "repel", "attract"),
    ) == (4, 4)
    assert score_episode(None, ("attract", "repel", "repel", "attract")) == (0, 4)


_PRIVATE_ONLY_FILENAMES = (
    "private_leaderboard.json",
    "private_episodes.json",
)

_BUILD_DEPLOY_DIR = _REPO_ROOT / "scripts"


def test_build_deploy_guardrail_rejects_private_asset_in_runtime(tmp_path):
    """Verify _verify_no_private_in_runtime() exits when a private-only file is present."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("build_deploy", _BUILD_DEPLOY_DIR / "build_deploy.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    fake_runtime = tmp_path / "kaggle-runtime"
    fake_runtime.mkdir()
    (fake_runtime / "private_episodes.json").write_text("{}", encoding="utf-8")

    orig_runtime = mod.DEPLOY_RUNTIME_DIR
    orig_repo = mod.REPO_ROOT
    mod.DEPLOY_RUNTIME_DIR = fake_runtime
    mod.REPO_ROOT = tmp_path
    try:
        import pytest
        with pytest.raises(SystemExit):
            mod._verify_no_private_in_runtime()
    finally:
        mod.DEPLOY_RUNTIME_DIR = orig_runtime
        mod.REPO_ROOT = orig_repo


def test_build_deploy_guardrail_rejects_non_public_split_whitelist(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("build_deploy", _BUILD_DEPLOY_DIR / "build_deploy.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    fake_runtime = tmp_path / "kaggle-runtime"
    frozen_splits = fake_runtime / "src" / "frozen_splits"
    frozen_splits.mkdir(parents=True)
    (frozen_splits / "dev.json").write_text("{}", encoding="utf-8")
    (frozen_splits / "public_leaderboard.json").write_text("{}", encoding="utf-8")
    (frozen_splits / "extra.json").write_text("{}", encoding="utf-8")

    orig_runtime = mod.DEPLOY_RUNTIME_DIR
    mod.DEPLOY_RUNTIME_DIR = fake_runtime
    try:
        import pytest
        with pytest.raises(SystemExit):
            mod._verify_public_runtime_split_whitelist()
    finally:
        mod.DEPLOY_RUNTIME_DIR = orig_runtime


def test_kaggle_runtime_deploy_does_not_contain_private_leaderboard():
    """Guardrail: private_leaderboard.json must never appear in the runtime package."""
    deploy_runtime = _REPO_ROOT / "deploy" / "kaggle-runtime"
    if not deploy_runtime.exists():
        import pytest
        pytest.skip("deploy/kaggle-runtime/ not yet built; run scripts/build_deploy.py first")
    private_files = list(deploy_runtime.rglob("private_leaderboard.json"))
    assert private_files == [], (
        f"private_leaderboard.json must not be shipped in deploy/kaggle-runtime/; "
        f"found: {[str(p) for p in private_files]}"
    )


def test_runtime_deploy_contains_no_private_only_assets():
    """Guardrail: no private-only asset filename may appear anywhere in deploy/kaggle-runtime/."""
    deploy_runtime = _REPO_ROOT / "deploy" / "kaggle-runtime"
    if not deploy_runtime.exists():
        import pytest
        pytest.skip("deploy/kaggle-runtime/ not yet built; run scripts/build_deploy.py first")
    violations = []
    for name in _PRIVATE_ONLY_FILENAMES:
        violations.extend(deploy_runtime.rglob(name))
    assert violations == [], (
        f"Private-only asset(s) found in deploy/kaggle-runtime/: "
        f"{[str(p) for p in violations]}"
    )


def test_public_packaging_does_not_reference_private_assets():
    """Guardrail: frozen_artifacts_manifest.json must not reference any private-only asset."""
    manifest_text = (_KAGGLE_DIR / "frozen_artifacts_manifest.json").read_text(encoding="utf-8")
    for name in _PRIVATE_ONLY_FILENAMES:
        assert name not in manifest_text, (
            f"Public packaging artifact frozen_artifacts_manifest.json references "
            f"private-only asset '{name}'"
        )


def test_frozen_splits_runtime_package_contains_only_public_splits():
    """Guardrail: the frozen_splits/ dir in the runtime package must contain only dev and
    public_leaderboard — no private-only split files."""
    deploy_runtime = _REPO_ROOT / "deploy" / "kaggle-runtime"
    if not deploy_runtime.exists():
        import pytest
        pytest.skip("deploy/kaggle-runtime/ not yet built; run scripts/build_deploy.py first")
    frozen_splits_dir = deploy_runtime / "src" / "frozen_splits"
    if not frozen_splits_dir.exists():
        return
    shipped = {p.name for p in frozen_splits_dir.iterdir() if p.is_file()}
    allowed = {"dev.json", "public_leaderboard.json"}
    unexpected = shipped - allowed
    assert unexpected == set(), (
        f"Unexpected file(s) in runtime frozen_splits/: {unexpected}. "
        f"Only dev.json and public_leaderboard.json are permitted."
    )


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
