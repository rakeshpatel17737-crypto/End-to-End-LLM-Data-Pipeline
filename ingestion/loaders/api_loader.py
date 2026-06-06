"""Generic REST API loader — fetches JSON and flattens text fields."""
from __future__ import annotations

import json

import httpx

from .pdf_loader import RawDocument


def load_api(
    url: str,
    text_fields: list[str] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 15,
) -> RawDocument:
    """Fetch a JSON endpoint and extract text from specified fields.

    text_fields: list of dot-separated field paths to extract (e.g. ["body", "data.text"]).
    If None, the full JSON is serialized as text.
    """
    resp = httpx.get(url, timeout=timeout, headers=headers or {}, follow_redirects=True)
    resp.raise_for_status()

    data = resp.json()

    if text_fields:
        parts = []
        for field_path in text_fields:
            value = data
            for key in field_path.split("."):
                if isinstance(value, dict):
                    value = value.get(key, "")
                else:
                    value = ""
                    break
            if value:
                parts.append(str(value))
        text = "\n\n".join(parts)
    else:
        text = json.dumps(data, indent=2, ensure_ascii=False)

    title = data.get("title") or data.get("name") if isinstance(data, dict) else None

    return RawDocument(
        source_uri=url,
        source_type="api",
        title=title,
        pages=[text],
        total_chars=len(text),
    )
