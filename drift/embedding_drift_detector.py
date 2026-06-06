"""Embedding drift detection: centroid shift, norm IQR, cosine similarity degradation.

Adapts the three-signal detection pattern from feature-store/drift_detection/drift_detector.py
for the embedding domain. Reuses ks_test() and z_score_test() from statistical_tests.py verbatim.
Health score adapts the SEVERITY_PENALTY pattern from AI-Reliability/engine/predictor.py.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .statistical_tests import ks_test, z_score_test


@dataclass
class EmbeddingDriftResult:
    centroid_shift: float
    norm_iqr: float
    norm_mean: float
    norm_std: float
    cosine_sim_mean: float
    cosine_sim_std: float
    ks_statistic: float
    ks_p_value: float
    z_score_norm_mean: float
    drift_severity: str        # 'ok' | 'warn' | 'alert'
    health_score: int          # 0–100
    chunk_sample_size: int
    baseline_size: int
    computed_at: float = field(default_factory=time.time)

    @property
    def is_drifting(self) -> bool:
        return self.drift_severity in ("warn", "alert")


def detect_embedding_drift(
    current_embeddings: list[list[float]],
    baseline_embeddings: list[list[float]],
) -> EmbeddingDriftResult:
    """Three drift signals specific to embedding spaces.

    Signal 1 — Centroid shift: cosine distance between mean(current) and mean(baseline).
      Thresholds: warn > 0.05, alert > 0.15.

    Signal 2 — Norm distribution: KS test + Z-score on L2 norms.
      Reuses ks_test() and z_score_test() from statistical_tests.py unchanged.

    Signal 3 — Cosine similarity to baseline centroid.
      Thresholds: warn if mean < 0.6, alert if mean < 0.4.

    Health score adapted from AI-Reliability SEVERITY_PENALTY pattern:
      100 - 30 (alert) - 15 (warn) - 10 (ks_p_value < 0.01) - 10 (z_score > 4.0)
    """
    if not current_embeddings or not baseline_embeddings:
        return EmbeddingDriftResult(
            centroid_shift=0.0, norm_iqr=0.0, norm_mean=0.0, norm_std=0.0,
            cosine_sim_mean=1.0, cosine_sim_std=0.0,
            ks_statistic=0.0, ks_p_value=1.0, z_score_norm_mean=0.0,
            drift_severity="ok", health_score=100,
            chunk_sample_size=0, baseline_size=0,
        )

    current = np.array(current_embeddings, dtype=np.float32)
    baseline = np.array(baseline_embeddings, dtype=np.float32)

    current_centroid = current.mean(axis=0)
    baseline_centroid = baseline.mean(axis=0)
    centroid_shift = float(
        1.0 - np.dot(current_centroid, baseline_centroid) /
        (np.linalg.norm(current_centroid) * np.linalg.norm(baseline_centroid) + 1e-9)
    )

    current_norms = np.linalg.norm(current, axis=1).tolist()
    baseline_norms = np.linalg.norm(baseline, axis=1).tolist()
    norm_iqr = float(np.percentile(current_norms, 75) - np.percentile(current_norms, 25))
    norm_mean = float(np.mean(current_norms))
    norm_std = float(np.std(current_norms))

    ks = ks_test(baseline_norms, current_norms, alpha=0.05)
    zs = z_score_test(baseline_norms, current_norms, threshold=3.0)

    bl_norm = np.linalg.norm(baseline_centroid)
    if bl_norm < 0.01:
        # Near-zero centroid (many diverse unit vectors cancel out) — skip this signal
        cosine_sim_mean = 1.0
        cosine_sim_std = 0.0
    else:
        curr_row_norms = np.linalg.norm(current, axis=1) + 1e-9
        cosine_sims = (current @ baseline_centroid) / (curr_row_norms * bl_norm)
        cosine_sim_mean = float(cosine_sims.mean())
        cosine_sim_std = float(cosine_sims.std())

    # Severity uses centroid_shift and KS test only.
    # cosine_sim_mean is a reported metric but NOT a threshold trigger —
    # in high-dimensional spaces random unit vectors have near-zero cosine sim
    # to their centroid by construction, making absolute thresholds unreliable.
    severity = "ok"
    if centroid_shift > 0.15:
        severity = "alert"
    elif centroid_shift > 0.05 or ks.significant:
        severity = "warn"

    health = 100
    if severity == "alert":
        health -= 30
    elif severity == "warn":
        health -= 15
    if ks.p_value < 0.01:
        health -= 10
    if abs(zs.z_score) > 4.0:
        health -= 10
    health = max(0, health)

    return EmbeddingDriftResult(
        centroid_shift=round(centroid_shift, 6),
        norm_iqr=round(norm_iqr, 6),
        norm_mean=round(norm_mean, 6),
        norm_std=round(norm_std, 6),
        cosine_sim_mean=round(cosine_sim_mean, 6),
        cosine_sim_std=round(cosine_sim_std, 6),
        ks_statistic=round(ks.statistic, 6),
        ks_p_value=round(ks.p_value, 6),
        z_score_norm_mean=round(zs.z_score, 6),
        drift_severity=severity,
        health_score=health,
        chunk_sample_size=len(current_embeddings),
        baseline_size=len(baseline_embeddings),
    )
