"""Turn files and web pages into plain text sections ready for chunking."""

import io
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import PurePath

from . import config

TEXT_EXTENSIONS = {".md", ".markdown", ".txt", ".text", ".rst", ".org", ".csv", ".json", ".log"}
HTML_EXTENSIONS = {".html", ".htm"}
IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


class ExtractError(Exception):
    """The content could not be turned into text. The message is safe to show the user."""


@dataclass
class Extracted:
    title: str
    source_type: str
    # (text, page number or None)
    sections: list[tuple[str, int | None]] = field(default_factory=list)


def _title_from_text(text: str, fallback: str) -> str:
    match = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else fallback


class _HTMLText(HTMLParser):
    _SKIP = {"script", "style", "noscript", "nav", "footer", "aside", "svg", "form", "template"}
    _BLOCK = {"p", "div", "section", "article", "br", "li", "tr", "ul", "ol", "table",
              "blockquote", "pre", "main", "header"}
    _HEADINGS = {"h1": "#", "h2": "##", "h3": "###", "h4": "####", "h5": "#####", "h6": "######"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag in self._HEADINGS:
            self.parts.append(f"\n\n{self._HEADINGS[tag]} ")
        elif tag in self._BLOCK:
            self.parts.append("\n\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag == "title":
            self._in_title = False
        elif tag in self._HEADINGS or tag in self._BLOCK:
            self.parts.append("\n\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip_depth:
            self.parts.append(data)


def html_to_text(html: str) -> tuple[str, str]:
    """Return (title, readable text) for an HTML page."""
    parser = _HTMLText()
    parser.feed(html)
    text = "".join(parser.parts)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return " ".join(parser.title.split()), text


def _extract_pdf(data: bytes, fallback_title: str) -> Extracted:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ExtractError("This PDF is password-protected.")
        sections = []
        for number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                sections.append((text, number))
        meta_title = (reader.metadata.title if reader.metadata else None) or ""
    except PdfReadError as exc:
        raise ExtractError(f"Could not read this PDF: {exc}") from exc
    if not sections:
        raise ExtractError(
            "This PDF has no selectable text (it is probably scanned). "
            "Upload the pages as images instead."
        )
    return Extracted(meta_title.strip() or fallback_title, "pdf", sections)


def extract_file(filename: str, data: bytes) -> Extracted:
    path = PurePath(filename)
    stem, ext = path.stem or "Untitled", path.suffix.lower()

    if ext == ".pdf" or data[:5] == b"%PDF-":
        return _extract_pdf(data, stem)

    if ext in IMAGE_TYPES:
        from . import llm

        text = llm.describe_image(data, IMAGE_TYPES[ext])
        return Extracted(stem, "image", [(text, None)])

    if ext in HTML_EXTENSIONS:
        title, text = html_to_text(data.decode("utf-8", errors="replace"))
        if not text:
            raise ExtractError("No readable text found in this HTML file.")
        return Extracted(title or stem, "html", [(text, None)])

    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        if ext in TEXT_EXTENSIONS:
            text = data.decode("utf-8", errors="replace")
        else:
            raise ExtractError(
                f"Unsupported file type '{ext or filename}'. "
                "Cortex accepts text, Markdown, PDF, HTML and images."
            ) from exc
    if not text.strip():
        raise ExtractError("This file is empty.")
    source_type = "markdown" if ext in {".md", ".markdown"} else "text"
    return Extracted(_title_from_text(text, stem), source_type, [(text, None)])


def extract_url(url: str) -> Extracted:
    if not re.match(r"^https?://", url, flags=re.IGNORECASE):
        raise ExtractError("Only http:// and https:// links can be saved.")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; Cortex/0.1)"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            content_type = response.headers.get_content_type()
            charset = response.headers.get_content_charset() or "utf-8"
            data = response.read(config.MAX_UPLOAD_BYTES + 1)
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        raise ExtractError(f"Could not fetch that page: {exc}") from exc
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise ExtractError("That page is too large to save.")

    if content_type == "application/pdf" or data[:5] == b"%PDF-":
        extracted = _extract_pdf(data, url)
        extracted.source_type = "web"
        return extracted
    try:
        html = data.decode(charset, errors="replace")
    except LookupError:
        html = data.decode("utf-8", errors="replace")
    if content_type == "text/plain":
        title, text = url, html.strip()
    else:
        title, text = html_to_text(html)
    if not text:
        raise ExtractError("No readable text found on that page.")
    return Extracted(title or url, "web", [(text, None)])
