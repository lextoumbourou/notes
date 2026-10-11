import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_execution_failures.py"
spec = importlib.util.spec_from_file_location("execution_check", SCRIPT)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)

FAILED_PAGE = """<html><body>
<pre><code>scores = query @ key</code></pre>
<div class="nb-status-error">Execution failed</div>
</body></html>
"""


def run(directory):
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        status = checker.main([str(directory)])
    return status, stderr.getvalue()


class ExecutionFailureTests(unittest.TestCase):
    def test_failed_block_fails_the_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "posts").mkdir()
            (root / "posts" / "note.html").write_text(FAILED_PAGE)
            (root / "clean.html").write_text("<p>Execution failed is fine in prose.</p>")
            status, stderr = run(root)
            self.assertEqual(status, 1)
            self.assertIn("note.html':3:", stderr)
            self.assertNotIn("clean.html", stderr)

    def test_clean_output_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "index.html").write_text('<div class="nb-output">ok</div>')
            self.assertEqual(run(directory), (0, ""))

    def test_missing_directory_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            status, stderr = run(Path(directory) / "missing")
            self.assertEqual(status, 2)
            self.assertIn("Not a directory", stderr)


if __name__ == "__main__":
    unittest.main()
