import json
import hashlib
from pathlib import Path

manifest_path = Path("packaging/kaggle/frozen_artifacts_manifest.json")
target_path = Path("packaging/kaggle/kernel-metadata.json")

manifest = json.loads(manifest_path.read_text())
new_sha = hashlib.sha256(target_path.read_bytes()).hexdigest()

manifest["entry_points"]["kernel_metadata"]["sha256"] = new_sha

manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
print(new_sha)