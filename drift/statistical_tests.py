"""Statistical tests for feature drift detection."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class KSTestResult:
    statistic: float
    p_value: float
    significant: bool  # True = distributions differ significantly


@dataclass
class ZScoreResult:
    current_mean: float
    baseline_mean: float
    baseline_std: float
    z_score: float
    is_anomalous: bool


def ks_test(baseline: list[float], current: list[float], alpha: float = 0.05) -> KSTestResult:
    """Two-sample KS test. Significant p-value (<alpha) = distributions differ."""
    if not baseline or not current:
        return KSTestResult(statistic=0.0, p_value=1.0, significant=False)

    result = stats.ks_2samp(baseline, current)
    return KSTestResult(
        statistic=round(float(result.statistic), 6),
        p_value=round(float(result.pvalue), 6),
        significant=bool(result.pvalue < alpha),
    )


def z_score_test(
    baseline: list[float],
    current: list[float],
    threshold: float = 3.0,
) -> ZScoreResult:
    """Check if current mean deviates from baseline by more than threshold std devs."""
    if not baseline or not current:
        return ZScoreResult(0.0, 0.0, 1.0, 0.0, False)

    baseline_arr = np.array(baseline, dtype=float)
    baseline_mean = float(np.mean(baseline_arr))
    baseline_std = float(np.std(baseline_arr)) + 1e-6
    current_mean = float(np.mean(current))

    z = abs(current_mean - baseline_mean) / baseline_std
    return ZScoreResult(
        current_mean=round(current_mean, 4),
        baseline_mean=round(baseline_mean, 4),
        baseline_std=round(baseline_std, 4),
        z_score=round(z, 4),
        is_anomalous=z > threshold,
    )


def kl_divergence(p: list[float], q: list[float], n_bins: int = 10) -> float:
    """KL divergence D(P||Q) via binned probability distributions."""
    if not p or not q:
        return 0.0

    p_arr = np.array(p, dtype=float)
    q_arr = np.array(q, dtype=float)

    combined_min = min(p_arr.min(), q_arr.min())
    combined_max = max(p_arr.max(), q_arr.max()) + 1e-9

    bins = np.linspace(combined_min, combined_max, n_bins + 1)
    p_hist, _ = np.histogram(p_arr, bins=bins, density=True)
    q_hist, _ = np.histogram(q_arr, bins=bins, density=True)

    # Smooth to avoid log(0)
    p_hist = p_hist + 1e-10
    q_hist = q_hist + 1e-10

    p_hist /= p_hist.sum()
    q_hist /= q_hist.sum()

    return float(np.sum(p_hist * np.log(p_hist / q_hist)))
