import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.build_private_cogflex_dataset import build_private_bundle
from scripts.private_release_paths import (
    PRIVATE_REPO_ROOT_ENV_VAR,
    PRIVATE_ROWS_DIR_ENV_VAR,
    PRIVATE_ROWS_RELATIVE_DIR,
    PRIVATE_SCORING_DIR_ENV_VAR,
    PRIVATE_SCORING_RELATIVE_DIR,
    default_private_release_dirs,
    resolve_private_release_dirs,
)
from scripts.verify_cogflex import resolve_private_bundle_dirs


ROOT = Path(__file__).resolve().parents[1]


class PrivateReleasePathTests(unittest.TestCase):
    def test_default_private_release_dirs_fall_back_to_local_gitignored_surfaces(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            rows_dir, scoring_dir = default_private_release_dirs(ROOT)
        self.assertEqual(rows_dir, ROOT / PRIVATE_ROWS_RELATIVE_DIR)
        self.assertEqual(scoring_dir, ROOT / PRIVATE_SCORING_RELATIVE_DIR)

    def test_default_private_release_dirs_honor_private_repo_root(self) -> None:
        with TemporaryDirectory() as tmpdir, patch.dict(
            "os.environ",
            {PRIVATE_REPO_ROOT_ENV_VAR: str(Path(tmpdir) / "private-repo")},
            clear=True,
        ):
            rows_dir, scoring_dir = default_private_release_dirs(ROOT)
        self.assertEqual(rows_dir, Path(tmpdir) / "private-repo" / PRIVATE_ROWS_RELATIVE_DIR)
        self.assertEqual(scoring_dir, Path(tmpdir) / "private-repo" / PRIVATE_SCORING_RELATIVE_DIR)

    def test_default_private_release_dirs_honor_split_env_overrides(self) -> None:
        with TemporaryDirectory() as tmpdir, patch.dict(
            "os.environ",
            {
                PRIVATE_REPO_ROOT_ENV_VAR: str(Path(tmpdir) / "private-repo"),
                PRIVATE_ROWS_DIR_ENV_VAR: str(Path(tmpdir) / "rows-only"),
                PRIVATE_SCORING_DIR_ENV_VAR: str(Path(tmpdir) / "scoring-only"),
            },
            clear=True,
        ):
            rows_dir, scoring_dir = default_private_release_dirs(ROOT)
        self.assertEqual(rows_dir, Path(tmpdir) / "rows-only")
        self.assertEqual(scoring_dir, Path(tmpdir) / "scoring-only")

    def test_build_private_bundle_honors_private_repo_root(self) -> None:
        with TemporaryDirectory() as tmpdir, patch.dict(
            "os.environ",
            {PRIVATE_REPO_ROOT_ENV_VAR: str(Path(tmpdir) / "private-repo")},
            clear=True,
        ):
            bundle_paths = build_private_bundle()
            self.assertEqual(bundle_paths["rows"].parent, Path(tmpdir) / "private-repo" / PRIVATE_ROWS_RELATIVE_DIR)
            self.assertEqual(
                bundle_paths["answer_key"].parent,
                Path(tmpdir) / "private-repo" / PRIVATE_SCORING_RELATIVE_DIR,
            )
            self.assertTrue(bundle_paths["rows"].exists())
            self.assertTrue(bundle_paths["answer_key"].exists())

    def test_verify_private_bundle_dir_resolution_honors_split_env_overrides(self) -> None:
        with TemporaryDirectory() as tmpdir:
            rows_dir = Path(tmpdir) / "rows"
            scoring_dir = Path(tmpdir) / "scoring"
            rows_dir.mkdir(parents=True, exist_ok=True)
            scoring_dir.mkdir(parents=True, exist_ok=True)
            with patch.dict(
                "os.environ",
                {
                    PRIVATE_ROWS_DIR_ENV_VAR: str(rows_dir),
                    PRIVATE_SCORING_DIR_ENV_VAR: str(scoring_dir),
                },
                clear=True,
            ):
                resolved_rows_dir, resolved_scoring_dir = resolve_private_bundle_dirs(None, None, None)
        self.assertEqual(resolved_rows_dir, rows_dir.resolve())
        self.assertEqual(resolved_scoring_dir, scoring_dir.resolve())

    def test_explicit_private_release_dirs_override_environment_defaults(self) -> None:
        with TemporaryDirectory() as tmpdir, patch.dict(
            "os.environ",
            {PRIVATE_REPO_ROOT_ENV_VAR: str(Path(tmpdir) / "private-repo")},
            clear=True,
        ):
            rows_dir, scoring_dir = resolve_private_release_dirs(
                ROOT,
                rows_dir=Path(tmpdir) / "explicit-rows",
                scoring_dir=Path(tmpdir) / "explicit-scoring",
            )
        self.assertEqual(rows_dir, (Path(tmpdir) / "explicit-rows").resolve())
        self.assertEqual(scoring_dir, (Path(tmpdir) / "explicit-scoring").resolve())
