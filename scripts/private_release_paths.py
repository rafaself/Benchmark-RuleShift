from __future__ import annotations

import os
from pathlib import Path
from typing import Final

PRIVATE_REPO_ROOT_ENV_VAR: Final[str] = "COGFLEX_PRIVATE_REPO_ROOT"
PRIVATE_ROWS_DIR_ENV_VAR: Final[str] = "COGFLEX_PRIVATE_ROWS_DIR"
PRIVATE_SCORING_DIR_ENV_VAR: Final[str] = "COGFLEX_PRIVATE_SCORING_DIR"

PRIVATE_ROWS_RELATIVE_DIR: Final[Path] = Path("kaggle/dataset/private")
PRIVATE_SCORING_RELATIVE_DIR: Final[Path] = Path("kaggle/dataset/private-scoring")


def _optional_env_path(env_var: str) -> Path | None:
    """Resolve an optional path from the environment."""

    path_raw = os.environ.get(env_var)
    if not path_raw:
        return None
    return Path(path_raw).expanduser().resolve()


def default_private_repo_root() -> Path | None:
    """Resolve the optional external private repository root from the environment."""

    return _optional_env_path(PRIVATE_REPO_ROOT_ENV_VAR)


def default_private_release_dirs(public_repo_root: Path) -> tuple[Path, Path]:
    """Resolve the default private rows/scoring directories for this environment.

    Resolution order:
    1. ``COGFLEX_PRIVATE_ROWS_DIR`` / ``COGFLEX_PRIVATE_SCORING_DIR``
    2. ``COGFLEX_PRIVATE_REPO_ROOT`` joined with the standard split paths
    3. Local gitignored working surfaces inside the public repo
    """

    private_repo_root = default_private_repo_root()
    base_root = public_repo_root if private_repo_root is None else private_repo_root
    default_rows_dir = base_root / PRIVATE_ROWS_RELATIVE_DIR
    default_scoring_dir = base_root / PRIVATE_SCORING_RELATIVE_DIR
    return (
        _optional_env_path(PRIVATE_ROWS_DIR_ENV_VAR) or default_rows_dir,
        _optional_env_path(PRIVATE_SCORING_DIR_ENV_VAR) or default_scoring_dir,
    )


def resolve_private_release_dirs(
    public_repo_root: Path,
    *,
    rows_dir: Path | None = None,
    scoring_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Resolve private rows/scoring directories from explicit args or environment."""

    default_rows_dir, default_scoring_dir = default_private_release_dirs(public_repo_root)
    resolved_rows_dir = default_rows_dir if rows_dir is None else rows_dir.expanduser().resolve()
    resolved_scoring_dir = default_scoring_dir if scoring_dir is None else scoring_dir.expanduser().resolve()
    return resolved_rows_dir, resolved_scoring_dir
