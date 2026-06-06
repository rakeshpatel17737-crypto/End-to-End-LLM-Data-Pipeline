"""Chunk-level quality scoring: PII detection, language detection, deduplication."""
from __future__ import annotations

import re
from dataclasses import dataclass

_PII_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),                       # SSN
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),  # email
    re.compile(r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),  # US phone
    re.compile(r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b"),  # CC
]


@dataclass
class QualityReport:
    quality_score: float
    language: str | None
    has_pii: bool
    is_duplicate: bool
    rejection_reason: str | None


def score_chunk(
    content: str,
    content_hash: str,
    existing_hashes: set[str],
) -> QualityReport:
    if content_hash in existing_hashes:
        return QualityReport(
            quality_score=0.0,
            language=None,
            has_pii=False,
            is_duplicate=True,
            rejection_reason="exact_duplicate",
        )

    chars = len(content.strip())
    if chars < 50:
        length_score = 0.0
    elif chars >= 200:
        length_score = 1.0
    else:
        length_score = (chars - 50) / 150.0

    language, lang_confidence = _detect_language(content)
    language_score = 1.0 if lang_confidence >= 0.85 else 0.6

    has_pii = any(p.search(content) for p in _PII_PATTERNS)
    pii_penalty = -0.3 if has_pii else 0.0

    quality = max(0.0, min(1.0, length_score * language_score + pii_penalty))

    return QualityReport(
        quality_score=round(quality, 4),
        language=language,
        has_pii=has_pii,
        is_duplicate=False,
        rejection_reason=None,
    )


def _detect_language(text: str) -> tuple[str | None, float]:
    try:
        from langdetect import detect_langs
        results = detect_langs(text[:500])
        if results:
            top = results[0]
            return top.lang, top.prob
    except Exception:
        pass
    return None, 0.0
