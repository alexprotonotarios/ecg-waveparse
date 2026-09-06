"""Release checks must reject untrusted runs and changed approved archives."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("release", Path(__file__).with_name("verify_waveparse_release.py"))
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


class ReleaseRunTests(unittest.TestCase):
    def setUp(self):
        self.run = {"id": 123, "repository": {"full_name": release.REPOSITORY},
                    "head_repository": {"full_name": release.REPOSITORY},
                    "path": ".github/workflows/waveparse-packages.yml", "event": "workflow_dispatch",
                    "head_branch": "main", "head_sha": "a" * 40, "status": "completed", "conclusion": "success"}
        self.jobs = [{"name": f"package ({platform})", "conclusion": "success"} for platform in release.PLATFORMS]

    def test_requires_both_native_jobs_and_reviewed_revision(self):
        release.validate_run(self.run, self.jobs, "123", "a" * 40)
        with self.assertRaisesRegex(ValueError, "native job"):
            release.validate_run(self.run, self.jobs[:1], "123", "a" * 40)
        with self.assertRaisesRegex(ValueError, "source commit"):
            release.validate_run(self.run, self.jobs, "123", "b" * 40)

    def test_refuses_pr_fork_and_failed_runs(self):
        for updates in ({"event": "pull_request"}, {"conclusion": "failure"},
                        {"head_repository": {"full_name": "someone/fork"}},
                        {"path": ".github/workflows/other.yml"}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                release.validate_run({**self.run, **updates}, self.jobs, "123", "a" * 40)

    def test_does_not_copy_changed_approved_archives(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            platform = root / "waveparse-ubuntu-24.04"
            platform.mkdir()
            checks = dict.fromkeys(("dataFilesAbsent", "originalGitHistoryAbsent", "symlinksAbsent", "knownCredentialPatternsAbsent", "referenceStubVerified", "thirdPartyLicenceIncluded"), True)
            (platform / "source-audit.json").write_text(json.dumps({"checks": checks}))
            (platform / "inference.json").write_text(json.dumps({"status": "needs_review", "layout": "standard_6x2", "canonicalSha256": "a" * 64}))
            (platform / "runtime-licences.json").write_text("{}")
            (platform / "ecg-waveparse-0.1.0.tgz").write_bytes(b"changed archive")
            names = ("ecg-waveparse-0.1.0.tgz", "ecg_waveparse-0.1.0-py3-none-any.whl", "ecg_waveparse-0.1.0.tar.gz", "ecg-waveparse-0.1.0-source.tar.gz")
            with self.assertRaisesRegex(ValueError, "Reviewed hash mismatch"):
                release.verify_artifacts(root, root / "out", {"version": "0.1.0"}, dict.fromkeys(names, "a" * 64), "123", "a" * 40)
            self.assertFalse((root / "out").exists())


if __name__ == "__main__":
    unittest.main()
