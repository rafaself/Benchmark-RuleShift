from __future__ import annotations

from types import ModuleType

from scripts.private_local_loader import load_private_local_module

_PRIVATE_MODULE: ModuleType | None = None


def _module() -> ModuleType:
    global _PRIVATE_MODULE
    if _PRIVATE_MODULE is None:
        _PRIVATE_MODULE = load_private_local_module(
            "private_cogflex_bundle.py",
            "scripts.private_local.private_cogflex_bundle",
        )
    return _PRIVATE_MODULE


def __getattr__(name: str) -> object:
    return getattr(_module(), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_module())))
