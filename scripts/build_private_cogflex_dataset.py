#!/usr/bin/env python3

from __future__ import annotations

from types import ModuleType

from scripts.private_local_loader import PRIVATE_LOCAL_DATASET_DIR, load_private_local_module

_PRIVATE_MODULE: ModuleType | None = None


def _module() -> ModuleType:
    global _PRIVATE_MODULE
    if _PRIVATE_MODULE is None:
        _PRIVATE_MODULE = load_private_local_module(
            "build_private_cogflex_dataset.py",
            "scripts.private_local.build_private_cogflex_dataset",
        )
    return _PRIVATE_MODULE


PRIVATE_DATASET_DIR = PRIVATE_LOCAL_DATASET_DIR


def build_private_bundle(*args: object, **kwargs: object):
    return _module().build_private_bundle(*args, **kwargs)


def main() -> None:
    _module().main()


if __name__ == "__main__":
    main()
