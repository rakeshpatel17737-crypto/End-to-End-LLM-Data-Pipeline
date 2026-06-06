"""PDF document loader using PyMuPDF."""
from __future__ import annotations

import io
from dataclasses import dataclass


@dataclass
class RawDocument:
    source_uri: str
    source_type: str
    title: str | None
    pages: list[str]      # text per page
    total_chars: int


def load_pdf(path_or_bytes: str | bytes) -> RawDocument:
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise ImportError("Install PyMuPDF: pip install pymupdf") from e

    if isinstance(path_or_bytes, bytes):
        doc = fitz.open(stream=path_or_bytes, filetype="pdf")
        uri = "bytes://pdf"
    else:
        doc = fitz.open(path_or_bytes)
        uri = path_or_bytes

    pages = []
    for page in doc:
        text = page.get_text("text").strip()
        if text:
            pages.append(text)

    title = doc.metadata.get("title") or None
    total_chars = sum(len(p) for p in pages)
    doc.close()

    return RawDocument(
        source_uri=uri,
        source_type="pdf",
        title=title,
        pages=pages,
        total_chars=total_chars,
    )
