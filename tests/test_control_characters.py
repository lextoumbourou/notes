import contextlib
import importlib.util
import io
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_control_characters.py"
spec = importlib.util.spec_from_file_location("control_check", SCRIPT)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class ControlCharacterTests(unittest.TestCase):
    def test_all_forbidden_bytes_are_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            forbidden = set(range(32)) - {9, 10, 13}
            for byte in forbidden:
                (root / f"control-{byte}.md").write_bytes(b"First line\nSecond " + bytes([byte]))
            findings = list(checker.scan(root))
            self.assertEqual({byte for _, _, byte in findings}, forbidden)
            self.assertEqual(len(findings), len(forbidden))
            self.assertTrue(all(line == 2 for _, line, _ in findings))

    def test_valid_whitespace_unicode_and_non_utf8_bytes_are_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "valid.md").write_bytes("Café π 🙂\n\tIndented\r\n".encode() + b"\xff\x7f")
            self.assertEqual(list(checker.scan(root)), [])

    def test_scan_includes_hidden_files_and_only_the_existing_text_types(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hidden = root / ".hidden"
            hidden.mkdir()
            names = ["a.md", "space name.rst", "line\nbreak.txt"]
            for name in names:
                (hidden / name).write_bytes(b"bad\x00")
            for name in ("image.png", "notebook.ipynb", "upper.MD"):
                (hidden / name).write_bytes(b"bad\x00")
            self.assertEqual({path.name for path, _, _ in checker.scan(root)}, set(names))

    def test_symlink_files_and_directories_are_not_followed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            notes = root / "notes"
            outside = root / "outside"
            notes.mkdir()
            outside.mkdir()
            bad = outside / "bad.md"
            bad.write_bytes(b"\x00")
            (notes / "linked.md").symlink_to(bad)
            (notes / "linked-directory").symlink_to(outside, target_is_directory=True)
            self.assertEqual(list(checker.scan(notes)), [])

    def test_missing_directory_and_read_errors_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "note.md").write_text("valid")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(checker.main([str(root / "missing")]), 2)
                with patch.object(Path, "read_bytes", side_effect=PermissionError("unreadable")):
                    self.assertEqual(checker.main([str(root)]), 2)
                with patch.object(os, "scandir", side_effect=PermissionError("unreadable directory")):
                    self.assertEqual(checker.main([str(root)]), 2)

    def test_build_stops_on_bad_notes_and_proceeds_on_clean_notes(self):
        for bad in (True, False):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "notes").mkdir()
                (root / "scripts").mkdir()
                (root / "bin").mkdir()
                (root / "notes" / "note.md").write_bytes(b"First\nBad\x1b" if bad else b"Clean\n\tNote\r\n")
                shutil.copy2(SCRIPT, root / "scripts" / SCRIPT.name)
                commands = {
                    "uv": '#!/bin/sh\nshift\nif [ "$1" = python ]; then\nshift\nexec ' + shlex.quote(sys.executable) + ' "$@"\nfi\ntouch pelican-ran\n',
                    "npx": '#!/bin/sh\ntouch pagefind-ran\necho "Pagefind finished"\n',
                }
                for name, source in commands.items():
                    executable = root / "bin" / name
                    executable.write_text(source)
                    executable.chmod(0o755)
                result = subprocess.run(
                    ["bash", str(ROOT / "build.sh")], cwd=root, capture_output=True, text=True,
                    env=dict(os.environ, MERMAID_RENDERER="mmdc", PATH=f"{root / 'bin'}:/usr/bin:/bin"),
                )
                self.assertEqual(result.returncode, 1 if bad else 0, result.stdout + result.stderr)
                self.assertEqual((root / "pelican-ran").exists(), not bad)
                self.assertEqual((root / "pagefind-ran").exists(), not bad)
                if bad:
                    self.assertIn("note.md':2: control character U+001B", result.stderr)
                    self.assertRegex(result.stderr, r"Build failed after \d+s \(exit 1\)\.\n$")
                else:
                    self.assertRegex(result.stdout, r"Pagefind finished\nBuild completed in \d+s\.\n$")


if __name__ == "__main__":
    unittest.main()
