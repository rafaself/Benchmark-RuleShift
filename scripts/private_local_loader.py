from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_LOCAL_SCRIPTS_DIR = ROOT / "scripts" / "private_local"
PRIVATE_LOCAL_DATASET_DIR = ROOT / "kaggle" / "dataset" / "private_local"


def private_local_script_path(filename: str) -> Path:
    return PRIVATE_LOCAL_SCRIPTS_DIR / filename


def require_private_local_script(filename: str) -> Path:
    path = private_local_script_path(filename)
    if not path.is_file():
        raise RuntimeError(
            f"Missing local private asset: {path}. "
            "Keep private generation assets under scripts/private_local/."
        )
    return path


def load_private_local_module(filename: str, module_name: str) -> ModuleType:
    path = require_private_local_script(filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load local private module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
