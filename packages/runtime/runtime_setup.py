"""Explicit installation and offline verification for the WaveParse runtime.

Only setup performs network access. It never modifies the calling environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import fcntl


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def host_target() -> str:
    system = {"Darwin": "darwin", "Linux": "linux"}.get(platform.system(), "unsupported")
    machine = {"arm64": "arm64", "aarch64": "arm64", "x86_64": "x64", "AMD64": "x64"}.get(platform.machine(), platform.machine())
    return f"{system}-{machine}"


def default_directory(manifest: dict) -> Path:
    base = Path.home() / "Library/Caches" if sys.platform == "darwin" else Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "ecg-waveparse" / manifest["runtimeId"] / host_target()


def run(command: list[str], **kwargs) -> None:
    subprocess.run(command, check=True, stdout=sys.stderr, stderr=sys.stderr, **kwargs)


def verify_toolchain(manifest: dict) -> None:
    if platform.python_version() != manifest["python"]:
        raise RuntimeError(f"CPython {manifest['python']} required; found {platform.python_version()}.")
    if platform.python_implementation() != "CPython":
        raise RuntimeError("CPython is required.")
    node = os.environ.get("WAVEPARSE_NODE") or shutil.which("node")
    if not node:
        raise RuntimeError(f"Node.js {manifest['node']} is required.")
    version = subprocess.check_output([node, "--version"], text=True).strip()
    if version != "v" + manifest["node"]:
        raise RuntimeError(f"Node.js {manifest['node']} required; found {version}.")
    if host_target() not in manifest["platforms"]:
        raise RuntimeError(f"Unsupported runtime target {host_target()}; supported: {', '.join(manifest['platforms'])}.")


def download(url: str, destination: Path, expected_hash: str | None = None) -> None:
    if destination.is_file() and expected_hash and digest(destination) == expected_hash:
        return
    if not url.startswith("https://"):
        raise RuntimeError("Runtime downloads require HTTPS.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "ecg-waveparse-runtime-setup"})
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
        if not response.geturl().startswith("https://"):
            raise RuntimeError("Insecure download redirect refused.")
        shutil.copyfileobj(response, output)
    if expected_hash and digest(partial) != expected_hash:
        raise RuntimeError(f"Checksum mismatch for {destination.name}; file not activated.")
    partial.replace(destination)


def extract_source(archive: Path, engine: Path, files: dict[str, str]) -> None:
    """Extract only individually hashed, regular source files from the archive."""
    remaining = set(files)
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            parts = Path(member.name).parts
            if len(parts) < 2:
                continue
            name = "/".join(parts[1:])
            if name not in files:
                continue
            if name not in remaining or not member.isfile() or ".." in parts or Path(name).is_absolute():
                raise RuntimeError("Invalid or duplicate source archive entry.")
            stream = tar.extractfile(member)
            if stream is None:
                raise RuntimeError(f"Unreadable source archive entry: {name}")
            data = stream.read()
            if hashlib.sha256(data).hexdigest() != files[name]:
                raise RuntimeError(f"Source checksum mismatch: {name}")
            target = engine / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            remaining.remove(name)
    if remaining:
        raise RuntimeError(f"Source archive is missing {len(remaining)} locked files.")


def verify_source(engine: Path, files: dict[str, str]) -> None:
    for name, expected in files.items():
        file = engine / name
        if file.is_symlink() or not file.is_file() or digest(file) != expected:
            raise RuntimeError(f"Upstream source integrity mismatch: {name}")
    actual = {str(p.relative_to(engine)) for p in (engine / "src").rglob("*") if p.is_file() and "__pycache__" not in p.parts and not p.name.endswith((".pyc", ".pyo"))}
    if actual != {name for name in files if name.startswith("src/")}:
        raise RuntimeError("Untracked or missing engine source files.")


def verify_dependencies(python: Path, requirements: Path, resources: Path, engine: Path, device: str) -> None:
    probe = r'''
import importlib.metadata as m, json, os, platform, re, sys
from pathlib import Path
expected = {}
for line in Path(sys.argv[1]).read_text().splitlines():
    match = re.match(r"([A-Za-z0-9_.-]+)==([^\s;]+)", line)
    if match: expected[re.sub(r"[-_.]+", "-", match[1]).lower()] = match[2]
actual = {re.sub(r"[-_.]+", "-", d.metadata["Name"]).lower(): d.version for d in m.distributions()}
if actual != expected: raise RuntimeError("Installed Python distributions differ from the locked runtime: " + str({k:(expected.get(k),actual.get(k)) for k in expected.keys() | actual.keys() if expected.get(k) != actual.get(k)}))
os.environ["ORT_DISABLE_TELEMETRY"] = "1"
import onnxruntime as ort
ort.disable_telemetry_events()
import torch, cv2, numpy, scipy, pandas, rapidocr
sys.path[:0] = [sys.argv[2], sys.argv[3]]
from ecg_pipeline.fidelity_inference_wrapper import FidelityInferenceWrapper
from ecg_pipeline.native_grid_digitizer import trace_path
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
figure = plt.figure(figsize=(1, 1))
figure.canvas.draw()
plt.close(figure)
if "matplotlib.backends._macosx" in sys.modules: raise RuntimeError("GUI plotting backend loaded in local runtime")
if sys.argv[4] == "mps" and not torch.backends.mps.is_available(): raise RuntimeError("MPS is unavailable")
if platform.python_version() != "3.12.9": raise RuntimeError("Wrong inference Python version")
print(json.dumps({"rapidocrRoot": str(Path(rapidocr.__file__).parent), "plottingBackend": matplotlib.get_backend()}))
'''
    response = subprocess.check_output([str(python), "-c", probe, str(requirements), str(resources), str(engine), device], text=True, timeout=300, env={**os.environ, "PYTHONNOUSERSITE": "1", "MPLBACKEND": "Agg"})
    result = json.loads(response)
    if result["plottingBackend"].lower() != "agg":
        raise RuntimeError("Local runtime requires the non-interactive Agg plotting backend.")
    manifest = json.loads((resources / "config/waveparse-runtime.json").read_text())
    for model in manifest["ocrModels"]:
        file = Path(result["rapidocrRoot"]) / model["path"].removeprefix("rapidocr/")
        if not file.is_file() or digest(file) != model["sha256"]:
            raise RuntimeError(f"OCR model integrity mismatch: {model['path']}")
    subprocess.run([str(python), "-m", "pip", "check"], check=True, capture_output=True, text=True, timeout=60)


def doctor(directory: Path, resources: Path, manifest: dict, device: str, *, require_receipt: bool = True) -> dict:
    verify_toolchain(manifest)
    if device not in ("cpu", "mps") or (device == "mps" and host_target() != "darwin-arm64"):
        raise RuntimeError("The requested device is unsupported.")
    expected_manifest = digest(resources / "config/waveparse-runtime.json")
    if require_receipt:
        receipt = json.loads((directory / "installed.json").read_text())
        if receipt.get("manifestSha256") != expected_manifest or receipt.get("target") != host_target():
            raise RuntimeError("Installed runtime belongs to a different manifest or platform; run setup for this release.")
    engine = directory / "engine"
    verify_source(engine, manifest["sourceFiles"])
    for model in manifest["models"]:
        file = engine / model["path"]
        if not file.is_file() or file.is_symlink() or digest(file) != model["sha256"]:
            raise RuntimeError(f"Model integrity mismatch: {model['path']}")
    requirement = resources / manifest["platforms"][host_target()]["requirements"]
    if digest(requirement) != manifest["platforms"][host_target()]["requirementsSha256"]:
        raise RuntimeError("Packaged requirements lock was modified.")
    verify_dependencies(engine / ".venv/bin/python", requirement, resources, engine, device)
    return {"ok": True, "target": host_target(), "runtimeDir": str(directory), "manifestSha256": expected_manifest, "device": device, "plottingBackend": "Agg"}


def setup(directory: Path, resources: Path, manifest: dict, device: str) -> dict:
    verify_toolchain(manifest)
    directory.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path = directory.with_name(directory.name + ".setup.lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (directory / "installed.json").exists():
            return doctor(directory, resources, manifest, device)
        if directory.exists():
            raise RuntimeError("An incomplete runtime exists at the destination. Choose a new --runtime-dir; existing files were preserved.")
        stage = directory.with_name(directory.name + ".partial")
        stage.mkdir(mode=0o700, exist_ok=True)
        if shutil.disk_usage(stage).free < 6 * 1024**3:
            raise RuntimeError("Runtime setup requires at least 6 GiB of free disk space.")
        engine = stage / "engine"
        engine.mkdir(exist_ok=True)
        archive = stage / "source.tar.gz"
        if not archive.exists():
            download(manifest["sourceUrl"], archive)
        extract_source(archive, engine, manifest["sourceFiles"])
        for model in manifest["models"]:
            download(model["url"], engine / model["path"], model["sha256"])
        python = engine / ".venv/bin/python"
        if not python.exists():
            run([sys.executable, "-m", "venv", str(engine / ".venv")])
        bootstrap = resources / "requirements-runtime-bootstrap.txt"
        if digest(bootstrap) != manifest["bootstrapSha256"]:
            raise RuntimeError("Packaged bootstrap lock was modified.")
        run([str(python), "-m", "pip", "install", "--disable-pip-version-check", "--require-hashes", "--no-deps", "-r", str(bootstrap)])
        requirements = resources / manifest["platforms"][host_target()]["requirements"]
        if digest(requirements) != manifest["platforms"][host_target()]["requirementsSha256"]:
            raise RuntimeError("Packaged runtime lock was modified.")
        run([str(python), "-m", "pip", "install", "--disable-pip-version-check", "--require-hashes", "--no-deps", "--only-binary=:all:", "--no-binary=antlr4-python3-runtime", "--no-build-isolation", "-r", str(requirements)])
        doctor(stage, resources, manifest, device, require_receipt=False)
        archive.unlink()
        # Relocate text entry points before atomic activation. Interpreter
        # symlinks continue to point to the user-provided base Python.
        for script in (engine / ".venv/bin").iterdir():
            if script.is_symlink() or not script.is_file():
                continue
            data = script.read_bytes()
            if b"\0" not in data:
                script.write_bytes(data.replace(str(stage).encode(), str(directory).encode()))
        (stage / "installed.json").write_text(json.dumps({"manifestSha256": digest(resources / "config/waveparse-runtime.json"), "target": host_target()}))
        stage.rename(directory)
        return doctor(directory, resources, manifest, device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Explicit runtime setup and offline verification for ECG WaveParse")
    parser.add_argument("operation", choices=["setup", "doctor"])
    parser.add_argument("--resource-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--runtime-dir", type=Path)
    parser.add_argument("--workspace-dir", type=Path)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    args = parser.parse_args()
    resources = args.resource_root.resolve()
    manifest = json.loads((resources / "config/waveparse-runtime.json").read_text())
    directory = (args.runtime_dir or default_directory(manifest)).resolve()
    operation = setup if args.operation == "setup" else doctor
    result = operation(directory, resources, manifest, args.device)
    if args.workspace_dir:
        args.workspace_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        if shutil.disk_usage(args.workspace_dir).free < 6 * 1024**3:
            raise RuntimeError("Workspace requires at least 6 GiB free for the first run.")
        with tempfile.TemporaryFile(dir=args.workspace_dir) as probe:
            probe.write(b"waveparse"); probe.flush(); os.fsync(probe.fileno())
    print(json.dumps(result))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"ok": False, "error": str(error)}), file=sys.stderr)
        sys.exit(1)
