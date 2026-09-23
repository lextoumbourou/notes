"""Small rendering helpers. API calls stay in the notebook, not in this library."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import re
from typing import Any

__all__ = [
    "HTMLPreview", "read_article", "render_html", "render_anthropic_html",
    "render_openai_html", "load_html",
]


def read_article(path: str | Path, *, before: str = "Hands-on example") -> str:
    """Read up to the input-end marker, falling back to a named section."""
    text = Path(path).read_text(encoding="utf-8")
    if text.startswith("---\n"):
        _, separator, text = text[4:].partition("\n---\n")
        if not separator:
            raise ValueError("Unclosed article frontmatter")
    marker = re.search(r"^<!-- notebook-input-end -->[ \t]*$", text, flags=re.M)
    if marker:
        return text[:marker.start()].strip()
    body, separator, _ = text.partition(f"\n## {before}\n")
    if not separator:
        raise ValueError(
            "Add <!-- notebook-input-end --> before the example, "
            f"or pass before= with its section heading (currently {before!r})."
        )
    return body.strip()


def _document(text: str) -> str:
    text = text.strip()
    fence = re.fullmatch(r"```(?:html)?\s*\n(.*?)\n```", text, flags=re.I | re.S)
    if fence:
        text = fence[1].strip()
    if not re.match(r"<!doctype\s+html\s*>\s*<html\b", text, flags=re.I):
        raise ValueError("Expected a complete HTML document starting with <!doctype html>")
    if not text.lower().endswith("</html>"):
        raise ValueError("HTML document is incomplete")
    return text


@dataclass(frozen=True)
class HTMLPreview:
    """A rich notebook result whose scripts run in a separate, sandboxed frame."""

    html: str
    height: int = 900
    title: str = "Generated HTML"

    def _repr_html_(self) -> str:
        if not isinstance(self.height, int) or self.height <= 0:
            raise ValueError("Preview height must be a positive integer")
        return (
            f'<iframe title="{escape(self.title, quote=True)}" '
            f'width="100%" height="{self.height}" style="border:0;display:block" '
            'sandbox="allow-scripts" '
            f'srcdoc="{escape(self.html, quote=True)}"></iframe>'
        )


def render_html(
    text: str, path: str | Path, *, height: int = 900, title: str = "Generated HTML"
) -> HTMLPreview:
    """Validate and save a complete HTML page, returning its notebook preview."""
    page = _document(text)
    preview = HTMLPreview(page, height, title)
    preview._repr_html_()  # Validate display settings before overwriting a saved page.
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(page, encoding="utf-8")
    return preview


def _field(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def render_anthropic_html(
    message: Any, path: str | Path, *, height: int = 900, title: str = "Generated HTML"
) -> HTMLPreview:
    """Save a completed Anthropic Message's text and usage, excluding thinking."""
    reason = _field(message, "stop_reason")
    if reason != "end_turn":
        raise ValueError(f"Anthropic response is incomplete (stop_reason={reason!r})")
    text = "\n".join(
        _field(block, "text", "")
        for block in _field(message, "content", [])
        if _field(block, "type") == "text"
    )
    usage = _field(message, "usage", {})
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump(mode="json")
    record = {
        "provider": "anthropic",
        "model": _field(message, "model"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stop_reason": reason,
        "usage": usage,
    }
    receipt = json.dumps(record, indent=2) + "\n"
    preview = render_html(text, path, height=height, title=title)
    Path(path).with_suffix(".json").write_text(receipt, encoding="utf-8")
    return preview


def render_openai_html(
    response: Any, path: str | Path, *, height: int = 900, title: str = "Generated HTML"
) -> HTMLPreview:
    """Save a completed Responses API result's text and usage, excluding reasoning."""
    status = _field(response, "status")
    if status != "completed":
        raise ValueError(f"OpenAI response is incomplete (status={status!r})")
    text = "\n".join(
        _field(block, "text", "")
        for item in _field(response, "output", [])
        if _field(item, "type") == "message"
        for block in _field(item, "content", [])
        if _field(block, "type") == "output_text"
    )
    usage = _field(response, "usage", {})
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump(mode="json")
    reasoning = _field(response, "reasoning", {})
    record = {
        "provider": "openai",
        "model": _field(response, "model"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "reasoning_effort": _field(reasoning, "effort"),
        "service_tier": _field(response, "service_tier"),
        "usage": usage,
    }
    receipt = json.dumps(record, indent=2) + "\n"
    preview = render_html(text, path, height=height, title=title)
    Path(path).with_suffix(".json").write_text(receipt, encoding="utf-8")
    return preview


def load_html(path: str | Path, *, height: int = 900, title: str = "Generated HTML") -> HTMLPreview:
    """Preview a saved page without calling a model or rewriting its receipt."""
    return HTMLPreview(_document(Path(path).read_text(encoding="utf-8")), height, title)
