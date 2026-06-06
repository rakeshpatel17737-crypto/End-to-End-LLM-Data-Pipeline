"""Unit tests for drift/embedding_drift_detector.py"""
import numpy as np
import pytest


def _random_embeddings(n: int, dim: int = 1536, seed: int = 0) -> list[list[float]]:
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return (vecs / norms).tolist()


def test_no_drift_identical():
    from drift.embedding_drift_detector import detect_embedding_drift
    embs = _random_embeddings(100)
    result = detect_embedding_drift(embs, embs)
    assert result.drift_severity == "ok"
    assert result.centroid_shift < 0.01
    assert result.health_score >= 80


def test_alert_on_shifted():
    from drift.embedding_drift_detector import detect_embedding_drift
    current = _random_embeddings(100, seed=42)
    baseline = _random_embeddings(100, seed=99)
    result = detect_embedding_drift(current, baseline)
    assert result.health_score <= 100
    assert result.drift_severity in ("ok", "warn", "alert")


def test_empty_inputs():
    from drift.embedding_drift_detector import detect_embedding_drift
    result = detect_embedding_drift([], [])
    assert result.drift_severity == "ok"
    assert result.health_score == 100


def test_health_score_range():
    from drift.embedding_drift_detector import detect_embedding_drift
    current = _random_embeddings(50, seed=1)
    baseline = _random_embeddings(50, seed=2)
    result = detect_embedding_drift(current, baseline)
    assert 0 <= result.health_score <= 100


def test_is_drifting_property():
    from drift.embedding_drift_detector import EmbeddingDriftResult, detect_embedding_drift
    current = _random_embeddings(50, seed=1)
    baseline = _random_embeddings(50, seed=2)
    result = detect_embedding_drift(current, baseline)
    if result.drift_severity == "ok":
        assert not result.is_drifting
    else:
        assert result.is_drifting
