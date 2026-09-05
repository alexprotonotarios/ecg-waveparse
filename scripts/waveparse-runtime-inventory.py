"""Record installed dependency licence evidence without importing inference code.

Run with the isolated runtime's interpreter. This inventories declarations and
licence files; it does not turn a package declaration into a model-specific grant.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import platform
import re
from pathlib import Path


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def collect(manifest_path: Path) -> dict:
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    distributions = []
    for distribution in sorted(metadata.distributions(), key=lambda d: d.metadata["Name"].lower()):
        info = distribution.metadata
        declared = info.get("License-Expression") or info.get("License")
        classifiers = [value for value in info.get_all("Classifier", []) if value.startswith("License ::")]
        notices = []
        for relative in distribution.files or []:
            # Avoid unrelated source filenames such as licence-plate classifiers.
            if not re.match(r"^(licen[sc]e|copying|notice|thirdpartynotices)(\b|[._-])", relative.name, re.I):
                continue
            absolute = Path(distribution.locate_file(relative))
            if absolute.is_file():
                data = absolute.read_bytes()
                notices.append({"path": str(relative), "sha256": digest(data), "bytes": len(data)})
        distributions.append({
            "name": info["Name"], "version": distribution.version,
            "declaredLicense": declared if declared and len(declared) < 250 else None,
            "licenseDeclarationSha256": digest(declared.encode()) if declared else None,
            "licenseClassifiers": classifiers,
            "projectUrls": info.get_all("Project-URL", []),
            "homePage": info.get("Home-page"), "licenseFiles": notices,
            "metadataSha256": digest((distribution.read_text("METADATA") or distribution.read_text("PKG-INFO") or "").encode()),
        })
    return {
        "schemaVersion": 1, "python": platform.python_version(),
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "runtimeManifestSha256": digest(manifest_bytes),
        "distributionCount": len(distributions), "distributions": distributions,
        "engine": {"commit": manifest["engineCommit"], "license": "CC-BY-SA-4.0",
                   "licenseSha256": manifest["sourceFiles"]["LICENSE"]},
        "models": manifest["models"], "ocrModels": manifest["ocrModels"],
        "modelTermsStatus": "Repository/package declarations inventoried; no separate model-specific grant has been established. See LICENSING.md.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Inventoried {result['distributionCount']} distributions for {result['platform']}")
