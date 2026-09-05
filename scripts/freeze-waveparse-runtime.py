"""Maintainer command: freeze installed source identities, never executed at install."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
release = json.loads((ROOT / "config/release-runtime.json").read_text())
waveparse = json.loads((ROOT / "config/waveparse-release.json").read_text())
engine = ROOT / ".external/open-ecg-digitizer"
commit = release["engine"]["commit"]
files = subprocess.check_output(["git", "-C", str(engine), "ls-tree", "-r", "--name-only", commit], text=True).splitlines()
source_files = {}
for name in files:
    if name.startswith("src/") or name in ("LICENSE", "setup.py", "pyproject.toml"):
        data = subprocess.check_output(["git", "-C", str(engine), "show", f"{commit}:{name}"])
        source_files[name] = hashlib.sha256(data).hexdigest()
sha = lambda p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
manifest = {
    "schemaVersion": 1, "runtimeId": f"waveparse-{waveparse['version']}-{commit[:12]}",
    "node": waveparse["node"], "python": waveparse["python"], "engineCommit": commit,
    "sourceUrl": f"https://codeload.github.com/Ahus-AIM/Open-ECG-Digitizer/tar.gz/{commit}",
    "sourceFiles": source_files,
    "models": [{**model, "url": f"https://media.githubusercontent.com/media/Ahus-AIM/Open-ECG-Digitizer/{commit}/{model['path']}"} for model in release["engine"]["models"]],
    "ocrModels": release["ocr"]["models"], "bootstrapSha256": sha("requirements-runtime-bootstrap.txt"),
    "platforms": {target: {"requirements": filename, "requirementsSha256": sha(filename)} for target, filename in [("darwin-arm64", "requirements-runtime-macos-arm64.txt"), ("linux-x64", "requirements-runtime-linux-x64.txt")]},
}
(ROOT / "config/waveparse-runtime.json").write_text(json.dumps(manifest, indent=2) + "\n")
