"""Recompute entry_points sha256 hashes in frozen_artifacts_manifest.json."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MANIFEST = _REPO_ROOT / "packaging" / "kaggle" / "frozen_artifacts_manifest.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))

    entry_points = manifest.get("entry_points", {})
    updated: list[str] = []
    for key, entry in entry_points.items():
        path = _REPO_ROOT / entry["path"]
        new_hash = _sha256(path)
        old_hash = entry.get("sha256", "")
        if new_hash != old_hash:
            entry["sha256"] = new_hash
            updated.append(f"  {key}: {old_hash[:12]}... -> {new_hash[:12]}...")

    _MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if updated:
        print("Updated hashes:")
        print("\n".join(updated))
    else:
        print("All entry_points hashes are up to date.")


if __name__ == "__main__":
    main()
