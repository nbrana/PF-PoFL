from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance


def label_histogram(labels: np.ndarray, num_classes: int) -> np.ndarray:
    h = np.bincount(labels.astype(int), minlength=num_classes).astype(np.float64)
    if h.sum() > 0:
        h = h / h.sum()
    return h


def emd_histograms(p: np.ndarray, q: np.ndarray) -> float:
    """Earth mover's distance between discrete distributions p, q (same support)."""
    support = np.arange(len(p), dtype=np.float64)
    return float(wasserstein_distance(support, support, u_weights=p, v_weights=q))


def pairwise_mean_emd(histograms: list[np.ndarray]) -> float:
    """Average pairwise EMD between node label histograms."""
    m = len(histograms)
    if m < 2:
        return 0.0
    acc = 0.0
    cnt = 0
    for i in range(m):
        for j in range(i + 1, m):
            acc += emd_histograms(histograms[i], histograms[j])
            cnt += 1
    return acc / max(cnt, 1)


def pool_internal_mean_emd(histograms: list[np.ndarray]) -> float:
    return pairwise_mean_emd(histograms)
