"""Visible text of a Bronze evidence artifact (HTML via stdlib parser, PDF via pypdf).

`normalise()` is the normalisation used by the verbatim check of U1.2: Unicode NFC,
HTML entities unescaped, all whitespace runs collapsed to one space, typographic
quotes/dashes kept as they are (no rewriting of the text).
"""
from __future__ import annotations

import re
import unicodedata
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

BLOCK_TAGS = {"p", "h1", "h2", "h3", "h4", "h5", "li", "td", "th", "div", "br", "tr", "section",
              "article", "figcaption", "blockquote", "span", "strong", "em", "b", "i", "a", "time"}
SKIP_TAGS = {"script", "style", "noscript", "svg", "template", "iframe"}
BREAK_TAGS = {"p", "h1", "h2", "h3", "h4", "h5", "li", "td", "th", "div", "br", "tr", "section",
              "article", "figcaption", "blockquote"}


def normalise(text: str) -> str:
    t = unicodedata.normalize("NFC", unescape(text or ""))
    t = t.replace(" ", " ").replace("​", "")
    return re.sub(r"\s+", " ", t).strip()


class _Blocks(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[str] = []
        self.cur: list[str] = []
        self.skip = 0

    def _flush(self) -> None:
        t = normalise("".join(self.cur))
        if t:
            self.blocks.append(t)
        self.cur = []

    def handle_starttag(self, tag, attrs):
        if tag in SKIP_TAGS:
            self.skip += 1
        elif tag in BREAK_TAGS:
            self._flush()

    def handle_endtag(self, tag):
        if tag in SKIP_TAGS:
            self.skip = max(0, self.skip - 1)
        elif tag in BREAK_TAGS:
            self._flush()

    def handle_data(self, data):
        if not self.skip:
            self.cur.append(data)

    def close(self):
        super().close()
        self._flush()


def decode_bytes(raw: bytes) -> str:
    m = re.search(rb'charset=["\']?([A-Za-z0-9_\-]+)', raw[:4096])
    for enc in ([m.group(1).decode("ascii", "ignore")] if m else []) + ["utf-8", "cp1258", "latin-1"]:
        try:
            return raw.decode(enc)
        except (LookupError, UnicodeDecodeError):
            continue
    return raw.decode("utf-8", "replace")


def html_blocks(path: Path) -> list[str]:
    p = _Blocks()
    p.feed(decode_bytes(Path(path).read_bytes()))
    p.close()
    out, seen = [], set()
    for b in p.blocks:
        if b not in seen:
            seen.add(b)
            out.append(b)
    return out


def html_meta(path: Path) -> dict[str, str]:
    doc = decode_bytes(Path(path).read_bytes())
    meta = {}
    for key in ("og:title", "article:published_time", "og:description", "description"):
        m = re.search(r'<meta[^>]+(?:property|name)=["\']' + re.escape(key)
                      + r'["\'][^>]*content=["\']([^"\']*)["\']', doc, re.I)
        if m:
            meta[key] = normalise(m.group(1))
    return meta


def pdf_pages(path: Path) -> list[str]:
    from pypdf import PdfReader
    return [normalise(pg.extract_text() or "") for pg in PdfReader(str(path)).pages]


def artifact_text(path: Path) -> str:
    """Single normalised string of the artifact's visible text (blocks joined by ' \n ')."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return "\n".join(pdf_pages(path))
    if path.suffix.lower() in {".html", ".htm"}:
        blocks = html_blocks(path)
        meta = html_meta(path)
        return "\n".join(blocks + [v for v in meta.values()])
    return ""


def is_verbatim(passage: str, text: str) -> bool:
    """Normalised substring check (whitespace-insensitive at run level, case-sensitive)."""
    p, t = normalise(passage), normalise(text.replace("\n", " "))
    return bool(p) and p in t
