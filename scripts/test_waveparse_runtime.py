"""Installer tests use synthetic archives only; no network or model downloads."""
import hashlib
import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("runtime_setup", Path(__file__).resolve().parents[1] / "packages/runtime/runtime_setup.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


class RuntimeArchiveTest(unittest.TestCase):
    def archive(self, directory, entries):
        archive = directory / "source.tar.gz"
        with tarfile.open(archive, "w:gz") as output:
            for name, content in entries:
                entry = tarfile.TarInfo(name)
                entry.size = len(content)
                output.addfile(entry, io.BytesIO(content))
        return archive

    def test_only_hash_allowlisted_source_is_extracted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = {"src/main.py": hashlib.sha256(b"safe").hexdigest()}
            archive = self.archive(root, [("upstream/src/main.py", b"safe"), ("upstream/../../escape", b"bad"), ("upstream/private.png", b"private")])
            runtime.extract_source(archive, root / "engine", files)
            self.assertEqual((root / "engine/src/main.py").read_bytes(), b"safe")
            self.assertEqual([p.name for p in (root / "engine").iterdir()], ["src"])
            runtime.verify_source(root / "engine", files)
            (root / "engine/src/extra.py").write_text("extra")
            with self.assertRaises(RuntimeError):
                runtime.verify_source(root / "engine", files)

    def test_missing_duplicate_and_modified_files_fail_closed(self):
        for entries in [[], [("root/src/main.py", b"wrong")], [("root/src/main.py", b"safe")] * 2]:
            with self.subTest(entries=entries), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                archive = self.archive(root, entries)
                with self.assertRaises(RuntimeError):
                    runtime.extract_source(archive, root / "engine", {"src/main.py": hashlib.sha256(b"safe").hexdigest()})


if __name__ == "__main__":
    unittest.main()
