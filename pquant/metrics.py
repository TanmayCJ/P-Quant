"""Probabilistic and point forecast metrics + significance tests.

All ensemble functions accept optional per-sample weights so the same code
scores Kronos-prior and news-reweighted posteriors.
"""
from __future__ import annotations

import math

import numpy as np


def _norm_weights(x: np.ndarray, w: np.ndarray | None) -> np.ndarray:
    if w is None:
        return np.full(x.shape, 1.0 / x.shape[1])
    w = np.atleast_2d(np.asarray(w, dtype=float))
    if w.shape != x.shape or (w < 0).any():
        raise ValueError("weights must be non-negative and match samples shape")
    return w / w.sum(axis=1, keepdims=True)


def crps_ensemble(samples, y, weights=None) -> np.ndarray:
    """CRPS of (weighted) ensemble: E|X-y| - 0.5 E|X-X'|. samples [B,n] or [n]; y [B] or scalar."""
    x = np.atleast_2d(np.asarray(samples, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    w = _norm_weights(x, weights)
    t1 = (w * np.abs(x - y[:, None])).sum(axis=1)
    t2 = 0.5 * np.einsum("bi,bj,bij->b", w, w, np.abs(x[:, :, None] - x[:, None, :]))
    return t1 - t2


def weighted_quantile(samples, qs, weights=None) -> np.ndarray:
    """Quantiles [B, len(qs)] using midpoint cumulative weights (Hazen for uniform w)."""
    x = np.atleast_2d(np.asarray(samples, dtype=float))
    w = _norm_weights(x, weights)
    order = np.argsort(x, axis=1)
    xs = np.take_along_axis(x, order, axis=1)
    ws = np.take_along_axis(w, order, axis=1)
    cw = np.cumsum(ws, axis=1) - 0.5 * ws
    qs = np.atleast_1d(qs)
    return np.stack([np.interp(qs, cw[b], xs[b]) for b in range(x.shape[0])])


def interval_metrics(samples, y, level: float, weights=None) -> dict:
    lo_hi = weighted_quantile(samples, [(1 - level) / 2, (1 + level) / 2], weights)
    y = np.atleast_1d(np.asarray(y, dtype=float))
    covered = (y >= lo_hi[:, 0]) & (y <= lo_hi[:, 1])
    return {"coverage": covered.mean(), "width": (lo_hi[:, 1] - lo_hi[:, 0]).mean(),
            "covered": covered}


def interval_score(lo, hi, y, level: float) -> np.ndarray:
    """Winkler / interval score (proper for central intervals; lower is better)."""
    lo, hi, y = (np.asarray(a, float) for a in (lo, hi, y))
    a = 1 - level
    return (hi - lo) + (2 / a) * (lo - y) * (y < lo) + (2 / a) * (y - hi) * (y > hi)


def pit(samples, y, weights=None) -> np.ndarray:
    """Probability integral transform values (uniform if calibrated)."""
    x = np.atleast_2d(np.asarray(samples, dtype=float))
    w = _norm_weights(x, weights)
    y = np.atleast_1d(np.asarray(y, dtype=float))
    return (w * (x < y[:, None])).sum(1) + 0.5 * (w * (x == y[:, None])).sum(1)


def pit_ks(p) -> float:
    """Kolmogorov-Smirnov distance between PIT values and Uniform(0,1)."""
    p = np.sort(np.asarray(p, float))
    n = len(p)
    i = np.arange(1, n + 1)
    return float(max((i / n - p).max(), (p - (i - 1) / n).max()))


def _rank(a: np.ndarray) -> np.ndarray:
    order = a.argsort(kind="mergesort")
    r = np.empty(len(a))
    r[order] = np.arange(len(a))
    # average ties
    vals, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r)
    return (sums / counts)[inv]


def ic(pred, true) -> float:
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    if pred.std() == 0 or true.std() == 0:
        return float("nan")
    return float(np.corrcoef(pred, true)[0, 1])


def rank_ic(pred, true) -> float:
    return ic(_rank(np.asarray(pred, float)), _rank(np.asarray(true, float)))


def directional_accuracy(pred, true) -> float:
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    m = (true != 0) & (pred != 0)
    return float((np.sign(pred[m]) == np.sign(true[m])).mean()) if m.any() else float("nan")


def _t_sf_two_sided(t: float, df: int) -> float:
    try:
        from scipy import stats

        return float(2 * stats.t.sf(abs(t), df))
    except ImportError:  # normal approximation; fine for df > ~100
        return float(math.erfc(abs(t) / math.sqrt(2)))


def dm_test(loss_a, loss_b, h: int = 1) -> dict:
    """Diebold-Mariano with Newey-West variance (lag h-1) and HLN small-sample correction.
    stat < 0 => model A has lower loss."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    T = len(d)
    if T < 10:
        return {"stat": float("nan"), "p": float("nan"), "mean_diff": float(d.mean()) if T else float("nan")}
    dc = d - d.mean()
    var = dc @ dc / T
    for k in range(1, h):
        var += 2 * (1 - k / h) * (dc[k:] @ dc[:-k]) / T
    var = max(var, 1e-18)
    stat = d.mean() / math.sqrt(var / T)
    stat *= math.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    return {"stat": float(stat), "p": _t_sf_two_sided(stat, T - 1), "mean_diff": float(d.mean())}


def hac_se_sum(u, lag: int) -> float:
    """Newey-West (Bartlett) standard error of sum(u) for a mean-zero, serially dependent series
    (u = per-date influence contributions, as in Driscoll-Kraay)."""
    u = np.asarray(u, float)
    T = len(u)
    v = u @ u
    for k in range(1, min(lag, T - 1) + 1):
        v += 2 * (1 - k / (lag + 1)) * (u[k:] @ u[:-k])
    return math.sqrt(max(v, 0.0) * T / (T - 1))


def nw_auto_lag(T: int) -> int:
    """Newey-West rule-of-thumb bandwidth floor(4 (T/100)^(2/9))."""
    return int(4 * (T / 100) ** (2 / 9))


def block_bootstrap_ci(x, block: int = 10, n_boot: int = 2000, alpha: float = 0.05,
                       seed: int = 0) -> tuple[float, float]:
    """Moving-block bootstrap CI for the mean of a serially dependent series."""
    x = np.asarray(x, float)
    T = len(x)
    block = max(1, min(block, T))
    rng = np.random.default_rng(seed)
    n_blocks = -(-T // block)
    starts = rng.integers(0, T - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :T]
    means = x[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
