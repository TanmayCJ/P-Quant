"""Online, delay-aware calibration of ensemble forecasts.

Inputs are time-ordered arrays: X [T, n] samples, y [T] realized targets,
optional W [T, n] sample weights. The outcome of origin s only becomes known
at origin s + delay (delay = ceil(pred_len / stride)), so every method at step
t uses only y[: t - delay + 1]. y is passed whole for convenience; each
function enforces causality via _past().

Interval methods:   raw, aci (Gibbs & Candes 2021), cqr (rolling split conformal on ensemble quantiles)
Distribution methods (also give CRPS): pit_recalibrate (Kuleshov et al. 2018, online),
                                        spread_rescale (spread-skill variance scaling),
                                        emos_affine (online location + scale correction)
"""
from __future__ import annotations

import numpy as np

from . import metrics as M


def _past(t: int, delay: int, window: int | None) -> slice:
    end = t - delay + 1
    if end <= 0:
        return slice(0, 0)
    return slice(max(0, end - window) if window else 0, end)


def _row_w(W, t):
    return None if W is None else W[t:t + 1]


def raw_interval(X, level: float, W=None):
    q = M.weighted_quantile(X, [(1 - level) / 2, (1 + level) / 2], W)
    return q[:, 0], q[:, 1]


def aci_interval(X, y, level: float, delay: int = 1, gamma: float = 0.005, W=None, start: int = 0):
    """Adaptive conformal inference on the ensemble quantile level. Returns lo, hi, alpha_t.
    Outcomes of origins < start are ignored (e.g. while an upstream corrector warms up)."""
    T = len(y)
    target = 1.0 - level
    a = target
    lo, hi, alphas = np.empty(T), np.empty(T), np.empty(T)
    for t in range(T):
        s = t - delay
        if s >= start:  # outcome of origin s just became available
            a += gamma * (target - float(not lo[s] <= y[s] <= hi[s]))
        alphas[t] = a
        ac = min(max(a, 0.0), 1.0)
        lo[t], hi[t] = M.weighted_quantile(X[t:t + 1], [ac / 2, 1 - ac / 2], _row_w(W, t))[0]
    return lo, hi, alphas


def cqr_interval(X, y, level: float, delay: int = 1, window: int = 250, warmup: int = 50, W=None,
                 start: int = 0):
    """Conformalized ensemble-quantile interval: widen/shrink by a quantile of past scores.
    Scores of origins < start are ignored (e.g. while an upstream corrector warms up)."""
    blo, bhi = raw_interval(X, level, W)
    scores = np.maximum(blo - y, y - bhi)  # only past entries are read below
    lo, hi = blo.copy(), bhi.copy()
    for t in range(len(y)):
        sl = _past(t, delay, window)
        past = scores[max(sl.start, start):sl.stop]
        n = len(past)
        if n < warmup:
            continue
        k = int(np.ceil((n + 1) * level))
        q = np.inf if k > n else np.sort(past)[k - 1]
        lo[t] -= q
        hi[t] += q
    return lo, hi


def pit_recalibrate(X, y, delay: int = 1, window: int = 250, warmup: int = 50, W=None):
    """Map nominal levels through the empirical CDF of past PITs. Returns (samples [T,n], None)."""
    T, n = X.shape
    pits = M.pit(X, y, W)
    taus = (np.arange(n) + 0.5) / n
    out = np.empty((T, n))
    for t in range(T):
        past = pits[_past(t, delay, window)]
        levels = taus if len(past) < warmup else np.quantile(past, taus)
        out[t] = M.weighted_quantile(X[t:t + 1], levels, _row_w(W, t))[0]
    return out, None


def emos_affine(X, y, delay: int = 1, window: int = 250, warmup: int = 50, W=None, Z=None,
                ridge: float = 0.1, b_bounds=(-2.0, 2.0), k_bounds=(0.2, 10.0), k_mode: str = "mean_ratio"):
    """Online EMOS-style location/scale correction (Gneiting et al. 2005, ensemble form).

    Fit y ~ a + b*mu + c.Z on past realized origins (ridge on standardized slopes, penalty
    ridge * n), then X' = loc + k*(X - mu) where k matches ensemble spread to past residuals.
    Z [T, p] holds covariates known at forecast time; passing the context gap lets the fit
    remove Kronos' reversion toward the context-window mean explicitly instead of through b.
    Returns (samples, W)."""
    w = np.full(X.shape, 1.0 / X.shape[1]) if W is None else W / W.sum(1, keepdims=True)
    mu = (w * X).sum(1)
    sd = np.sqrt((w * (X - mu[:, None]) ** 2).sum(1)) + 1e-12
    F = mu[:, None] if Z is None else np.column_stack([mu, np.asarray(Z, float).reshape(len(mu), -1)])
    out = np.empty_like(X, dtype=float)
    for t in range(len(y)):
        sl = _past(t, delay, window)
        Fp, yy = F[sl], y[sl]
        n = len(yy)
        if n < warmup:
            out[t] = X[t]
            continue
        fm, fs = Fp.mean(0), Fp.std(0)
        fs[fs == 0] = 1.0
        Fs = (Fp - fm) / fs
        beta = np.linalg.solve(Fs.T @ Fs + ridge * n * np.eye(F.shape[1]), Fs.T @ (yy - yy.mean())) / fs
        beta[0] = np.clip(beta[0], *b_bounds)
        a = yy.mean() - fm @ beta
        resid = yy - (a + Fp @ beta)
        k = float(np.clip(_spread_ratio(resid, sd[sl], k_mode), *k_bounds))
        out[t] = a + F[t] @ beta + k * (X[t] - mu[t])
    return out, W


def _spread_ratio(resid, sd, mode: str) -> float:
    """How much to widen the ensemble. 'mean_ratio' = sqrt(mean((resid/sd)^2)), sensitive to
    near-degenerate ensembles (one tiny sd explodes it); 'pooled' = sqrt(mean(resid^2) / mean(sd^2))."""
    if mode == "pooled":
        return float(np.sqrt(np.mean(resid ** 2) / np.mean(sd ** 2)))
    if mode == "mean_ratio":
        return float(np.sqrt(np.mean((resid / sd) ** 2)))
    raise ValueError(f"unknown k_mode {mode!r}")


def spread_rescale(X, y, delay: int = 1, window: int = 250, warmup: int = 50, W=None,
                   k_bounds=(0.2, 10.0)):
    """Scale ensemble spread around its mean by sqrt(mean past z^2). Returns (samples, W)."""
    w = np.full(X.shape, 1.0 / X.shape[1]) if W is None else W / W.sum(1, keepdims=True)
    mu = (w * X).sum(1)
    sd = np.sqrt((w * (X - mu[:, None]) ** 2).sum(1)) + 1e-12
    z2 = ((y - mu) / sd) ** 2
    out = np.empty_like(X, dtype=float)
    for t in range(len(y)):
        past = z2[_past(t, delay, window)]
        k = 1.0 if len(past) < warmup else float(np.clip(np.sqrt(past.mean()), *k_bounds))
        out[t] = mu[t] + k * (X[t] - mu[t])
    return out, W
