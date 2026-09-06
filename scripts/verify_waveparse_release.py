"""Verify a reviewed native CI run and copy its exact archives for publication.

Uses only the standard library. Does not install or execute downloaded packages.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile

REPOSITORY = "alexprotonotarios/ecg-waveparse"
PLATFORMS = ("ubuntu-24.04", "macos-15")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_run(run, jobs, run_id, source_commit):
    require(re.fullmatch(r"[1-9][0-9]*", run_id), "Invalid package run ID")
    require(re.fullmatch(r"[0-9a-f]{40}", source_commit), "Expected a full source commit")
    require(str(run.get("id")) == run_id, "Different package run")
    require(run.get("repository", {}).get("full_name") == REPOSITORY, "Different repository")
    require(run.get("head_repository", {}).get("full_name") == REPOSITORY, "Forked run")
    require(run.get("path") == ".github/workflows/waveparse-packages.yml", "Different workflow")
    require(run.get("event") == "workflow_dispatch" and run.get("head_branch") == "main", "Expected a manual main-branch package run")
    require(run.get("head_sha") == source_commit, "Different source commit")
    require(run.get("status") == "completed" and run.get("conclusion") == "success", "Package run has not passed")
    for platform in PLATFORMS:
        matches = [job for job in jobs if job.get("name") == f"package ({platform})"]
        require(len(matches) == 1 and matches[0].get("conclusion") == "success", f"Missing successful native job: {platform}")


def find_one(directory, name):
    matches = list(directory.rglob(name))
    require(len(matches) == 1 and matches[0].is_file() and not matches[0].is_symlink(), f"Expected one regular {name} in {directory.name}")
    return matches[0]


def tar_files(path):
    """Ignore tar timestamps/compression differences when comparing source archives."""
    result = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            require(not member.name.startswith("/") and ".." not in Path(member.name).parts, "Unsafe archive path")
            require(member.isdir() or member.isfile(), "Archive contains a link or special file")
            if member.isfile():
                name = str(Path(member.name))
                require(name not in result, "Duplicate archive member")
                require(member.size < 8 * 1024 * 1024, "Unexpected large source file")
                result[name] = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
    return result


def verify_artifacts(directory, output, release, hashes, run_id, source_commit):
    version = release["version"]
    require(re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version), "Unsupported release version")
    names = {
        f"ecg-waveparse-{version}.tgz": "npm",
        f"ecg_waveparse-{version}-py3-none-any.whl": "python",
        f"ecg_waveparse-{version}.tar.gz": "python",
        f"ecg-waveparse-{version}-source.tar.gz": "source",
    }
    require(set(hashes) == set(names), "Provide exactly the four reviewed archive hashes")
    require(all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes.values()), "Invalid SHA-256")
    require(not output.exists(), "Use a fresh output directory")
    selected = {}
    for platform in PLATFORMS:
        root = directory / f"waveparse-{platform}"
        audit = json.loads(find_one(root, "source-audit.json").read_text())
        required_checks = {"dataFilesAbsent", "originalGitHistoryAbsent", "symlinksAbsent", "knownCredentialPatternsAbsent", "referenceStubVerified", "thirdPartyLicenceIncluded"}
        require(all(audit.get("checks", {}).get(check) is True for check in required_checks), f"Source audit incomplete: {platform}")
        inference = json.loads(find_one(root, "inference.json").read_text())
        require(inference.get("status") == "needs_review" and inference.get("layout") == "standard_6x2", f"Native inference evidence missing: {platform}")
        require(re.fullmatch(r"[0-9a-f]{64}", inference.get("canonicalSha256", "")), "Missing inference identity")
        find_one(root, "runtime-licences.json")
        for name in names:
            file = find_one(root, name)
            if platform == PLATFORMS[0]:
                require(sha256(file) == hashes[name], f"Reviewed hash mismatch: {name}")
                selected[name] = file
            elif name.endswith((".tgz", ".whl")):
                require(sha256(file) == hashes[name], f"Native package mismatch: {name}")
            else:
                require(tar_files(file) == tar_files(selected[name]), f"Native source contents differ: {name}")
    # Copy only after every check passes. Keep source archives out of PyPI's upload directory.
    for name, destination in names.items():
        target = output / destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(selected[name], target)
    receipt = {"schemaVersion": 1, "repository": REPOSITORY, "packageRunId": run_id,
               "sourceCommit": source_commit, "version": version, "sha256": hashes,
               "published": False, "rebuiltDuringPublication": False}
    (output / "release-verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def github_api(endpoint):
    return json.loads(subprocess.check_output(["gh", "api", endpoint], text=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("metadata", "artifacts"))
    parser.add_argument("--directory", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("verified"))
    args = parser.parse_args()
    run_id = os.environ["PACKAGE_RUN_ID"]
    source_commit = os.environ["SOURCE_COMMIT"]
    require(re.fullmatch(r"[1-9][0-9]*", run_id), "Invalid package run ID")
    require(re.fullmatch(r"[0-9a-f]{40}", source_commit), "Expected a full source commit")
    if args.mode == "metadata":
        run = github_api(f"repos/{REPOSITORY}/actions/runs/{run_id}")
        jobs = github_api(f"repos/{REPOSITORY}/actions/runs/{run_id}/jobs?per_page=100")["jobs"]
        validate_run(run, jobs, run_id, source_commit)
        if os.environ.get("PUBLICATION_TARGET", "verify") != "verify":
            require(os.environ.get("PUBLICATION_ENABLED") == "true", "Publication remains disabled; complete the documented release decision first")
            require(github_api(f"repos/{REPOSITORY}")["private"] is False, "Release source must be public before registry publication")
        print(f"Verified successful native package run {run_id} at {source_commit}")
    else:
        release = json.loads(Path("config/waveparse-release.json").read_text())
        hashes = json.loads(os.environ["ARTIFACT_HASHES"])
        receipt = verify_artifacts(args.directory, args.output, release, hashes, run_id, source_commit)
        print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
