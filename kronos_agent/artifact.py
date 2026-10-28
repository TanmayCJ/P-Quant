"""Volatility-scaling artifact in LLM return elicitation (Phase A).

Claim. If an LLM's answer distribution q over K return ranges is (close to) constant, and the
ranges are scaled by recent volatility sigma_t, then the elicited expected return is
    E_Q[r]_t = sigma_t * m(q),  m(q) = sum_k q_k c_k   (c_k = bin centres in sigma units),
so corr(E_Q, r_{t+h}) = sign(m(q)) * corr(sigma_t, r_{t+h}) exactly: the "signal" is the
volatility-return relation, carried by the prompt's ranges, with no news content at all.
A shuffled-headline placebo keeps sigma_t and therefore reproduces the effect; fixed-scale
ranges and direction labels do not carry it.

This module provides the empirical decomposition (IC, IC per sigma, IC partialling out sigma)
and a simulation that reproduces the mechanism under controlled conditions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics as M
from .confirm import _one_sided
from .fusion import bin_edges, llm_only_samples


def partial_ic(signal, r, sigma) -> float:
    """corr(residual of signal ~ 1 + sigma, r): the part of the signal not explained by volatility."""
    s, r, v = (np.asarray(a, float) for a in (signal, r, sigma))
    X = np.column_stack([np.ones_like(v), v])
    beta, *_ = np.linalg.lstsq(X, s, rcond=None)
    return M.ic(s - X @ beta, r)


def _zprod(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return np.zeros(len(x))
    return (x - x.mean()) / x.std() * (y - y.mean()) / y.std()


def _resid(s, v):
    X = np.column_stack([np.ones_like(v), v])
    beta, *_ = np.linalg.lstsq(X, s, rcond=None)
    return s - X @ beta


def decompose(frames: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """frames: asset, mode, variant, date, eq, sigma, r. Returns (per-asset, pooled) tables."""
    rows, pooled = [], []
    for (mode, variant), f in frames.groupby(["mode", "variant"]):
        per_date_raw, per_date_part = [], []
        for asset, g in f.groupby("asset"):
            e, v, r = g["eq"].to_numpy(float), g["sigma"].to_numpy(float), g["r"].to_numpy(float)
            rows.append({"mode": mode, "variant": variant, "asset": asset, "n": len(g),
                         "ic": M.ic(e, r), "ic_partial_sigma": partial_ic(e, r, v),
                         "ic_neg_sigma": M.ic(-v, r), "corr_signal_sigma": M.ic(e, v)})
            per_date_raw.append(pd.Series(_zprod(e, r), index=g["date"].to_numpy()))
            per_date_part.append(pd.Series(_zprod(_resid(e, v), r), index=g["date"].to_numpy()))
        for label, parts in (("raw", per_date_raw), ("partial_sigma", per_date_part)):
            s = pd.concat(parts, axis=1).mean(axis=1).sort_index().to_numpy()
            t = M.dm_test(s, np.zeros_like(s), h=1)
            pooled.append({"mode": mode, "variant": variant, "signal": label, "n_dates": len(s),
                           "mean_d": float(s.mean()), "t": t["stat"], "p_two_sided": t["p"],
                           "p_one_sided_neg": _one_sided(t)})
    return pd.DataFrame(rows), pd.DataFrame(pooled)


def simulate(n_assets: int = 9, T: int = 400, kappa: float = 0.25, info: float = 0.0,
             bias=(0.06, 0.32, 0.30, 0.22, 0.05, 0.03, 0.02), noise: float = 0.6, seed: int = 0) -> pd.DataFrame:
    """Synthetic panel reproducing the mechanism.

    sigma_t: log-AR(1) volatility. r_{t+h} = kappa * (sigma_t - mean) + sigma_t * eps (vol-return
    relation, no news). The 'LLM' answers q_t = softmax(log bias + noise_t + info * z_t), where
    z_t is the standardized future return (info=0 => no information). Labels use the same logit
    noise mapped to up/down/no-effect. Returns frames for modes volbins/fixbins/labels and
    variants real/shuffled (shuffled permutes the LLM answers across dates, keeping sigma_t).
    """
    rng = np.random.default_rng(seed)
    K = len(bias)
    lb = np.log(np.asarray(bias, float))
    out = []
    dates = pd.date_range("2024-07-01", periods=T, freq="W")
    for a in range(n_assets):
        ls = np.empty(T)
        ls[0] = 0.0
        for t in range(1, T):
            ls[t] = 0.9 * ls[t - 1] + rng.normal(0, 0.3)
        sigma = 0.02 * np.exp(ls)
        eps = rng.normal(size=T)
        r = kappa * (sigma - sigma.mean()) + sigma * eps
        z = (r - r.mean()) / r.std()
        logits = lb + noise * rng.normal(size=(T, K))
        logits += info * z[:, None] * np.linspace(-1, 1, K)[None, :]
        Q = np.exp(logits - logits.max(1, keepdims=True))
        Q /= Q.sum(1, keepdims=True)
        up, down = Q[:, 4:].sum(1), Q[:, :3].sum(1)  # label view of the same beliefs
        perm = rng.permutation(T)
        for variant, idx in (("real", np.arange(T)), ("shuffled", perm)):
            Qv = Q[idx]
            vol = llm_only_samples(Qv, bin_edges(sigma), sigma, 64).mean(1)
            c = np.full(T, np.median(sigma))
            fix = llm_only_samples(Qv, bin_edges(c), c, 64).mean(1)
            lab = up[idx] - down[idx]
            for mode, e in (("volbins", vol), ("fixbins", fix), ("labels", lab)):
                out.append(pd.DataFrame({"asset": f"S{a}", "mode": mode, "variant": variant,
                                         "date": dates, "eq": e, "sigma": sigma, "r": r}))
    return pd.concat(out, ignore_index=True)
