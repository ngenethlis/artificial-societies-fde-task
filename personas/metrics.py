"""Distance metrics between a simulated answer distribution and the real one.

Each function takes two probability vectors of equal length and returns a
scalar. total_variation is the headline; variance_ratio flags mode collapse.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def _norm(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    s = p.sum()
    return p / s if s > 0 else p


def total_variation(p, q) -> float:
    """TVD = 0.5 * sum|p-q|. In [0,1]; "share of mass that would move"."""
    p, q = _norm(p), _norm(q)
    return float(0.5 * np.abs(p - q).sum())


def jensen_shannon(p, q) -> float:
    """Symmetric, bounded [0,1] divergence (log base 2)."""
    p, q = _norm(p), _norm(q)
    m = 0.5 * (p + q)

    def _kl(a, b):
        a = np.clip(a, _EPS, 1)
        b = np.clip(b, _EPS, 1)
        return float(np.sum(a * np.log2(a / b)))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def wasserstein_ordinal(p, q) -> float:
    """1-D earth-mover distance over ordered categories 0..k-1.

    Only meaningful for ordinal scales; for k categories it lies in [0, k-1].
    Captures that "strongly support" vs "tend to support" is a smaller miss
    than "strongly support" vs "strongly oppose".
    """
    p, q = _norm(p), _norm(q)
    return float(np.abs(np.cumsum(p) - np.cumsum(q)).sum())


def modal_accuracy(p, q) -> float:
    """1.0 if the two distributions share a modal answer, else 0.0.

    Proxy for "would the headline decision change?".
    """
    return float(np.argmax(_norm(p)) == np.argmax(_norm(q)))


def entropy(p) -> float:
    """Shannon entropy in bits."""
    p = np.clip(_norm(p), _EPS, 1)
    return float(-np.sum(p * np.log2(p)))


def variance_ratio(sim, real) -> float:
    """entropy(sim) / entropy(real). Below 1 means the personas flattened toward
    one option; around 1 means the spread was preserved; above 1 is over-dispersed."""
    er = entropy(real)
    return float(entropy(sim) / er) if er > 0 else float("nan")


def all_metrics(sim, real, ordinal: bool) -> dict[str, float]:
    out = {
        "tvd": total_variation(sim, real),
        "jsd": jensen_shannon(sim, real),
        "modal_match": modal_accuracy(sim, real),
        "variance_ratio": variance_ratio(sim, real),
    }
    if ordinal:
        out["wasserstein"] = wasserstein_ordinal(sim, real)
    return out
