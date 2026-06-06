"""Web page loader using httpx + BeautifulSoup."""
from __future__ import annotations

import re
from dataclasses import dataclass

import httpx

from .pdf_loader import RawDocument


def load_web(url: str, timeout: int = 15) -> RawDocument:
    try:
        from bs4 import BeautifulSoup
    except ImportError as e:
        raise ImportError("Install beautifulsoup4: pip install beautifulsoup4") from e

    headers = {"User-Agent": "LLMDataPlatform/1.0 (research scraper)"}
    resp = httpx.get(url, timeout=timeout, headers=headers, follow_redirects=True)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else None

    text = soup.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return RawDocument(
        source_uri=url,
        source_type="web",
        title=title,
        pages=[text],
        total_chars=len(text),
    )
