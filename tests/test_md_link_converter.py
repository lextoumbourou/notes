import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest


spec = importlib.util.spec_from_file_location(
    "site_md_links", Path(__file__).resolve().parents[1] / "my-plugins" / "md_link_converter.py"
)
md_links = importlib.util.module_from_spec(spec)
spec.loader.exec_module(md_links)


class MdLinkConverterTests(unittest.TestCase):
    def convert(self, notes_dir: Path, html: str) -> str:
        note = SimpleNamespace(
            _content=html, metadata={}, source_path=str(notes_dir / "linking-note.md")
        )
        md_links.convert_md_links(note)
        return note._content

    def test_uses_the_linked_notes_slug(self):
        with TemporaryDirectory() as tmp:
            notes = Path(tmp)
            (notes / "a-long-filename-for-an-essay.md").write_text(
                '---\ntitle: An essay\nslug: short-essay\n---\n\nBody.\n'
            )
            html = self.convert(notes, '<a href="a-long-filename-for-an-essay.md">An essay</a>')
            self.assertEqual(html, '<a href="short-essay.html">An essay</a>')

    def test_falls_back_to_the_filename(self):
        with TemporaryDirectory() as tmp:
            notes = Path(tmp)
            (notes / "plain-note.md").write_text("---\ntitle: Plain\n---\n\nBody.\n")
            html = self.convert(notes, '<a href="plain-note.md">Plain</a> <a href="missing-note.md">Gone</a>')
            self.assertEqual(html, '<a href="plain-note.html">Plain</a> <a href="missing-note.html">Gone</a>')


if __name__ == "__main__":
    unittest.main()
