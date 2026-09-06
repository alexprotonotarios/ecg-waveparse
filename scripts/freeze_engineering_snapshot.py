"""Pin tracked and new nonignored source bytes without resuming any campaign."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def freeze(root: Path, destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=False)
    files = sorted(set(subprocess.check_output(["git", "ls-files", "-c", "-o", "--exclude-standard", "-z"], cwd=root).decode().strip("\0").split("\0")))
    hashes = {}
    for name in files:
        source = root / name
        if source.is_symlink():
            raise ValueError(f"Snapshot refuses a source symlink: {name}")
        if not source.is_file():
            continue  # Deleted files are recorded by gitStatus below.
        target = destination / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    content_hash = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    manifest = {"version": 1, "sourceCommit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root).decode().strip(),
                "gitStatus": subprocess.check_output(["git", "status", "--porcelain"], cwd=root).decode(),
                "sourceTreeSha256": content_hash, "files": hashes, "clinicalValidationUse": False}
    (destination / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (destination / ".keep-artifacts.json").write_text(json.dumps({"version": 1, "reason": "Pinned engineering comparison; retain source and evidence."}) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = freeze(Path(__file__).resolve().parents[1], args.output.resolve())
    print(json.dumps({"sourceTreeSha256": manifest["sourceTreeSha256"], "sourceFiles": len(manifest["files"])}))
