"""News-likelihood reweighting of ensemble forecasts (log-linear pooling over return bins).

For origin t with samples x_i (weights v_i) and an LLM distribution q over K return bins:
    p_t(b)  = sum of v_i over samples in bin b        (prior bin mass, smoothed)
    w_i    ∝ v_i * (q(b_i) / p_t(b_i)) ** lam
The posterior bin mass is then ∝ p^(1-lam) q^lam. lam=0 recovers the prior; lam is
chosen online by minimizing past CRPS on realized origins only (delay-aware).
"""
from __future__ import annotations

import numpy as np

from . import metrics as M

Z_EDGES = (-1.5, -0.75, -0.25, 0.25, 0.75, 1.5)  # in units of horizon sigma -> 7 bins
LETTERS = "ABCDEFG"


def bin_edges(sigma_h: np.ndarray, z_edges=Z_EDGES) -> np.ndarray:
    """[T, K-1] return-space edges per origin."""
    return np.asarray(sigma_h, float)[:, None] * np.asarray(z_edges, float)[None, :]


def assign_bins(X: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin index [T, n] for each sample (0..K-1)."""
    return np.stack([np.searchsorted(edges[t], X[t], side="right") for t in range(len(X))])


def prior_bin_probs(B: np.ndarray, K: int, W=None, eps: float = 1e-3) -> np.ndarray:
    T, n = B.shape
    w = np.full((T, n), 1.0 / n) if W is None else W / W.sum(1, keepdims=True)
    P = np.zeros((T, K))
    for k in range(K):
        P[:, k] = (w * (B == k)).sum(1)
    P = P + eps
    return P / P.sum(1, keepdims=True)


def reweight(X, Q, edges, lam, W=None, eps: float = 1e-3) -> np.ndarray:
    """Posterior sample weights [T, n] for a scalar or per-origin lam."""
    K = Q.shape[1]
    B = assign_bins(X, edges)
    P = prior_bin_probs(B, K, W, eps)
    Qs = (Q + eps) / (Q + eps).sum(1, keepdims=True)
    ratio = np.take_along_axis(Qs, B, 1) / np.take_along_axis(P, B, 1)
    lam = np.broadcast_to(np.asarray(lam, float).reshape(-1, 1), ratio.shape)
    v = np.full(X.shape, 1.0 / X.shape[1]) if W is None else W / W.sum(1, keepdims=True)
    w = v * ratio ** lam
    return w / w.sum(1, keepdims=True)


def debias_q(Q, window: int = 250, warmup: int = 20) -> np.ndarray:
    """Divide each LLM distribution by the LLM's own average answer over past origins and
    renormalize: q~_t(b) ∝ q_t(b) / mean_{s<t} q_s(b). Removes standing biases (news
    negativity, letter-position preference) so only changes in the LLM's view count.
    Uses past prompts only (no outcomes), so no feedback delay is needed."""
    Q = np.asarray(Q, float)
    out = Q.copy()
    for t in range(warmup, len(Q)):
        base = Q[max(0, t - window):t].mean(0)
        r = Q[t] / np.maximum(base, 1e-6)
        out[t] = r / r.sum()
    return out


def online_lambda(X, y, Q, edges, delay: int, grid=(0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0),
                  window: int = 250, warmup: int = 50, W=None):
    """Pick lam_t from grid minimizing mean CRPS over past realized origins.
    Returns (weights [T,n], lam [T], crps_grid [T, G]). y is used only through past rows."""
    grid = np.asarray(grid, float)
    crps = np.stack([M.crps_ensemble(X, y, reweight(X, Q, edges, g, W)) for g in grid], axis=1)
    T = len(y)
    lam = np.zeros(T)
    for t in range(T):
        end = t - delay + 1
        if end < warmup:
            continue
        past = crps[max(0, end - window):end]
        lam[t] = grid[np.argmin(past.mean(0))]
    return reweight(X, Q, edges, lam, W), lam, crps


def llm_only_samples(Q, edges, sigma_h, n: int, tail_z: float = 3.0) -> np.ndarray:
    """Deterministic quantile samples from the LLM bin distribution, uniform within bins;
    outer bins are bounded at +/- tail_z * sigma_h."""
    T, K = Q.shape
    sig = np.asarray(sigma_h, float)
    full = np.column_stack([-tail_z * sig, edges, tail_z * sig])  # [T, K+1]
    full = np.maximum.accumulate(full, axis=1)
    taus = (np.arange(n) + 0.5) / n
    out = np.empty((T, n))
    for t in range(T):
        cdf = np.concatenate([[0.0], np.cumsum(Q[t] / Q[t].sum())])
        out[t] = np.interp(taus, cdf, full[t])
    return out
