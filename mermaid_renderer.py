"""Pinned native Mermaid renderer, with the official CLI as a fallback."""

import hashlib
import io
import json
import logging
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tarfile
import tempfile
from urllib.request import urlopen
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parent
VERSION = "0.3.1"
# Official GitHub release asset SHA-256 digests for v0.3.1.
ASSETS = {
    ("Darwin", "arm64"): ("aarch64-apple-darwin", "562d0250cb8588adefe398a23e4bbdf67f242849ea8d888c38268bcc3edf3223"),
    ("Darwin", "x86_64"): ("x86_64-apple-darwin", "ad035258822b60ee6bb3c086c3c7325dafe314bfab16a3bb99d692c18fad59cd"),
    ("Linux", "aarch64"): ("aarch64-unknown-linux-gnu", "a74a121a2dd3bc8d30c17c954b3a036e7467a53b21532e889601a26cc6dc39d9"),
    ("Linux", "x86_64"): ("x86_64-unknown-linux-gnu", "e1da47b758769bff21b82a480fad640a3b893a90fc0dc321551486f8b50c7200"),
}
logger = logging.getLogger(__name__)
# Count complete connectors, not the opening `--` of `A -- label --> B`.
FLOWCHART_EDGE = re.compile(r"(?:<?--+(?:[>ox]|-)|<?==+(?:>|=)|-\.+-(?:[>ox])?|~{3,})")


class RenderError(RuntimeError):
    pass


def native_binary():
    override = os.environ.get("MMDR_BINARY")
    if override:
        return Path(shutil.which(override) or override).expanduser().resolve()
    target, _ = ASSETS[(platform.system(), platform.machine())]
    return ROOT / ".tools" / "mmdr" / VERSION / target / "mmdr"


def install_native():
    """Download the pinned release into this checkout, never into global PATH."""
    try:
        binary = native_binary()
    except KeyError as exc:
        raise RuntimeError("No bundled mmdr for this platform; set MMDR_BINARY or MERMAID_RENDERER=mmdc") from exc
    if binary.is_file() and os.access(binary, os.X_OK):
        return binary
    if os.environ.get("MMDR_BINARY"):
        raise RuntimeError(f"MMDR_BINARY is not an executable file: {binary}")

    target, checksum = ASSETS[(platform.system(), platform.machine())]
    url = f"https://github.com/1jehuang/mermaid-rs-renderer/releases/download/v{VERSION}/mmdr-{target}.tar.gz"
    print(f"Installing Mermaid renderer mmdr {VERSION} ({target})...")
    with urlopen(url, timeout=60) as response:
        archive = response.read()
    if hashlib.sha256(archive).hexdigest() != checksum:
        raise RuntimeError("mmdr download checksum mismatch; executable was not installed")

    # Read just the executable instead of extracting arbitrary archive paths.
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
        member = next(m for m in bundle.getmembers() if m.name in {"mmdr", "./mmdr"})
        if not member.isfile():
            raise RuntimeError("mmdr release does not contain a regular executable file")
        executable = bundle.extractfile(member).read()

    binary.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=binary.parent, delete=False) as temp:
        temporary = Path(temp.name)
        temp.write(executable)
    try:
        temporary.chmod(0o755)
        temporary.replace(binary)
    finally:
        temporary.unlink(missing_ok=True)
    return binary


def validate_svg(svg):
    try:
        root = ElementTree.fromstring(svg)
    except ElementTree.ParseError as exc:
        raise RenderError("Renderer did not produce valid SVG") from exc
    if root.tag not in {"svg", "{http://www.w3.org/2000/svg}svg"}:
        raise RenderError("Renderer output is not an SVG document")
    return svg


def run(command, source):
    try:
        result = subprocess.run(command, input=source, text=True,
                                capture_output=True, timeout=30, check=True)
    except subprocess.CalledProcessError as exc:
        raise RenderError(f"{Path(command[0]).name}: {exc.stderr.strip()[:1000]}") from exc
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RenderError(f"{Path(command[0]).name}: {exc}") from exc
    return result.stdout


def render_native(source):
    try:
        binary = native_binary()
    except KeyError as exc:
        raise RenderError("Native renderer is not configured for this platform") from exc
    return validate_svg(run([str(binary), "-i", "-", "-e", "svg", "-t", "default"], source))


def render_official(source):
    # Preserve the previous global CLI when installed; use npm's local binary otherwise.
    binary = os.environ.get("MMDC_BINARY") or shutil.which("mmdc") or str(ROOT / "node_modules/.bin/mmdc")
    with tempfile.TemporaryDirectory(prefix="mermaid-") as directory:
        directory = Path(directory)
        output = directory / "diagram.svg"
        config = directory / "puppeteer.json"
        config.write_text(json.dumps({"args": ["--no-sandbox", "--disable-setuid-sandbox", "--disable-gpu"]}))
        run([binary, "-p", str(config), "-o", str(output)], source)
        if not output.is_file():
            raise RenderError("Mermaid CLI completed without creating an SVG")
        return validate_svg(output.read_text())


def needs_official_renderer(source):
    # v0.3.1 can silently merge node definitions in chained flowchart edges.
    # Be conservative: an arrow-like label may also select the official CLI.
    return bool(re.search(r"(?m)^\s*(?:graph|flowchart)\b", source)) and any(
        len(FLOWCHART_EDGE.findall(line)) > 1 for line in source.splitlines()
    )


def render_svg(source):
    renderer = os.environ.get("MERMAID_RENDERER", "mmdr")
    if renderer == "mmdc":
        return render_official(source)
    if renderer != "mmdr":
        raise RenderError("MERMAID_RENDERER must be mmdr or mmdc")
    if needs_official_renderer(source):
        logger.info("Mermaid CLI required for chained flowchart edges (mmdr v%s compatibility)", VERSION)
        return render_official(source)
    try:
        return render_native(source)
    except RenderError as native_error:
        logger.warning("Native Mermaid rendering failed; trying Mermaid CLI: %s", native_error)
        try:
            return render_official(source)
        except RenderError as official_error:
            raise RenderError(f"Both Mermaid renderers failed. Native: {native_error}. CLI: {official_error}") from official_error
