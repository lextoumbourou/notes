import base64
import hashlib
import io
import os
from pathlib import Path
import platform
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from markdown import Markdown

import mermaid_renderer as renderer

SVG = '<svg xmlns="http://www.w3.org/2000/svg"><text>A → B</text></svg>'


class RendererTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {"MERMAID_RENDERER": "mmdr"})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_native_success_does_not_start_browser(self):
        with patch.object(renderer, "render_native", return_value=SVG), \
             patch.object(renderer, "render_official") as official:
            self.assertEqual(renderer.render_svg("graph LR; A-->B"), SVG)
            official.assert_not_called()

    def test_invalid_native_svg_falls_back(self):
        with patch.object(renderer, "run", return_value="not SVG"), \
             patch.object(renderer, "render_official", return_value=SVG) as official, \
             self.assertLogs(renderer.logger, "WARNING"):
            self.assertEqual(renderer.render_svg("graph LR; A-->B"), SVG)
            official.assert_called_once_with("graph LR; A-->B")

    def test_both_renderers_failing_is_an_error(self):
        with patch.object(renderer, "render_native", side_effect=renderer.RenderError("native failed")), \
             patch.object(renderer, "render_official", side_effect=renderer.RenderError("CLI failed")), \
             self.assertLogs(renderer.logger, "WARNING"):
            with self.assertRaisesRegex(renderer.RenderError, "Both Mermaid renderers failed"):
                renderer.render_svg("invalid diagram")

    def test_timeout_becomes_render_error(self):
        with patch.object(subprocess, "run", side_effect=subprocess.TimeoutExpired("mmdr", 30)):
            with self.assertRaises(renderer.RenderError):
                renderer.run(["mmdr"], "graph LR; A-->B")

    def test_official_renderer_can_be_selected_explicitly(self):
        with patch.dict(os.environ, {"MERMAID_RENDERER": "mmdc"}), \
             patch.object(renderer, "render_native") as native, \
             patch.object(renderer, "render_official", return_value=SVG):
            self.assertEqual(renderer.render_svg("graph LR; A-->B"), SVG)
            native.assert_not_called()

    def test_non_svg_xml_is_rejected(self):
        with self.assertRaises(renderer.RenderError):
            renderer.validate_svg("<html>error</html>")

    def test_chained_flowchart_nodes_use_official_renderer(self):
        sources = [
            "graph LR\nA((Node A)) --- B((Node B)) --- C((Node C))",
            "flowchart TD\nA[Start] --> B[Middle] ==> C[End]",
        ]
        for source in sources:
            with self.subTest(source=source), \
                 patch.object(renderer, "render_native") as native, \
                 patch.object(renderer, "render_official", return_value=SVG) as official, \
                 self.assertLogs(renderer.logger, "INFO"):
                self.assertEqual(renderer.render_svg(source), SVG)
                native.assert_not_called()
                official.assert_called_once_with(source)

    def test_flowchart_edges_on_separate_lines_use_native_renderer(self):
        sources = [
            "graph LR\nA((Node A)) --- B((Node B))\nB --- C((Node C))",
            "graph LR\nA -- 5 --> B\nB -- 2 --> C",
        ]
        for source in sources:
            with self.subTest(source=source), \
                 patch.object(renderer, "render_native", return_value=SVG) as native, \
                 patch.object(renderer, "render_official") as official:
                self.assertEqual(renderer.render_svg(source), SVG)
                native.assert_called_once_with(source)
                official.assert_not_called()


class MarkdownTests(unittest.TestCase):
    def test_multiple_diagrams_preserve_surrounding_article(self):
        source = "Before\n\n```mermaid\ngraph LR; A-->B\n```\n\nBetween\n\n```mermaid\ngraph TD; C-->D\n```\n\nAfter"
        with patch("markdown_mermaid.render_svg", return_value=SVG) as render:
            html = Markdown(extensions=["markdown_mermaid"]).convert(source)
        self.assertEqual(render.call_count, 2)
        self.assertEqual(html.count("data:image/svg+xml;base64,"), 2)
        encoded = base64.b64encode(SVG.encode()).decode()
        self.assertIn(encoded, html)
        for paragraph in ("Before", "Between", "After"):
            self.assertIn(f"<p>{paragraph}</p>", html)

    def test_normal_code_does_not_invoke_renderer(self):
        with patch("markdown_mermaid.render_svg") as render:
            Markdown(extensions=["markdown_mermaid", "fenced_code"]).convert("```python\nprint('hello')\n```")
            render.assert_not_called()

    def test_render_error_is_not_embedded_as_successful_article(self):
        with patch("markdown_mermaid.render_svg", side_effect=renderer.RenderError("failed")):
            with self.assertRaises(renderer.RenderError):
                Markdown(extensions=["markdown_mermaid"]).convert("```mermaid\nbroken\n```")


class InstallationTests(unittest.TestCase):
    def test_corrupt_download_is_not_installed(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "mmdr"
            with patch.dict(os.environ, {"MMDR_BINARY": ""}), \
                 patch.object(renderer, "native_binary", return_value=binary), \
                 patch.object(renderer, "urlopen", return_value=io.BytesIO(b"corrupt archive")):
                with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                    renderer.install_native()
            self.assertFalse(binary.exists())

    def test_verified_archive_installs_only_executable(self):
        data = io.BytesIO()
        with tarfile.open(fileobj=data, mode="w:gz") as archive:
            for name, content in [("./mmdr", b"native executable"), ("../escape", b"unwanted")]:
                member = tarfile.TarInfo(name)
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))
        payload = data.getvalue()
        assets = {(platform.system(), platform.machine()): ("test", hashlib.sha256(payload).hexdigest())}
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "bin/mmdr"
            with patch.dict(os.environ, {"MMDR_BINARY": ""}), \
                 patch.object(renderer, "ASSETS", assets), \
                 patch.object(renderer, "native_binary", return_value=binary), \
                 patch.object(renderer, "urlopen", return_value=io.BytesIO(payload)) as download:
                self.assertEqual(renderer.install_native(), binary)
                self.assertEqual(binary.read_bytes(), b"native executable")
                self.assertTrue(os.access(binary, os.X_OK))
                renderer.install_native()
                self.assertEqual(download.call_count, 1)
            self.assertFalse((Path(directory) / "escape").exists())


class BuildScriptTests(unittest.TestCase):
    def test_failed_pelican_does_not_run_pagefind(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "notes").mkdir()
            bin_dir = directory / "bin"
            bin_dir.mkdir()
            commands = {
                "uv": '#!/bin/sh\nif [ "$2" = pelican ]; then exit 7; fi\n',
                "npx": '#!/bin/sh\ntouch pagefind-ran\n',
            }
            for name, script in commands.items():
                path = bin_dir / name
                path.write_text(script)
                path.chmod(0o755)
            env = dict(os.environ, PATH=f"{bin_dir}:/usr/bin:/bin", MERMAID_RENDERER="mmdr")
            result = subprocess.run(["bash", str(renderer.ROOT / "build.sh")], cwd=directory,
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 7, result.stdout + result.stderr)
            self.assertFalse((directory / "pagefind-ran").exists())
            self.assertRegex(result.stderr, r"Build failed after \d+s \(exit 7\)\.\n$")


if __name__ == "__main__":
    unittest.main()
