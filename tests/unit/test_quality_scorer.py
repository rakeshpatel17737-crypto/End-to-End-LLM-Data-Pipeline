"""Unit tests for ingestion/quality_scorer.py"""
from ingestion.quality_scorer import score_chunk


def test_short_content_low_score():
    report = score_chunk("Hi", "abc", set())
    assert report.quality_score < 0.5
    assert not report.is_duplicate


def test_long_content_high_score():
    text = "This is a meaningful paragraph with plenty of information. " * 5
    import hashlib
    h = hashlib.sha256(text.encode()).hexdigest()
    report = score_chunk(text, h, set())
    assert report.quality_score > 0.5


def test_exact_duplicate_rejected():
    import hashlib
    text = "some content"
    h = hashlib.sha256(text.encode()).hexdigest()
    report = score_chunk(text, h, {h})
    assert report.is_duplicate
    assert report.quality_score == 0.0
    assert report.rejection_reason == "exact_duplicate"


def test_pii_detection_email():
    text = "Please contact user@example.com for more info. " * 10
    report = score_chunk(text, "hash1", set())
    assert report.has_pii
    assert report.quality_score < 0.8


def test_pii_detection_ssn():
    text = ("The SSN is 123-45-6789. " + "Some filler content. ") * 10
    report = score_chunk(text, "hash2", set())
    assert report.has_pii
