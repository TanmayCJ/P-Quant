"""The volatility-scaling artifact: how answer ranges scaled by volatility fake a news signal.

In short: the LLM picks from return ranges whose width grows with recent volatility. If its
answers barely change (a steady lean), the expected return it implies just tracks volatility,
and volatility itself predicts returns, so the result looks like a news signal.

In symbols: if the answer distribution q over K ranges is (close to) constant and the ranges
are scaled by sigma_t, the elicited expected return is
    E_Q[r]_t = sigma_t * m(q),  m(q) = sum_k q_k c_k   (c_k = bin centres in sigma units),
so corr(E_Q, r_{t+h}) = sign(m(q)) * corr(sigma_t, r_{t+h}) exactly: the "signal" is the
volatility-return relation, carried by the prompt's ranges, with no news content at all.
A shuffled-headline placebo keeps sigma_t and therefore reproduces the effect; fixed-scale
ranges and direction labels do not carry it.

This module provides the empirical decomposition (IC, IC per sigma, IC partialling out sigma),
an answer-shuffle placebo for real data, and a simulation that reproduces the mechanism under
controlled conditions.

Answer-shuffle placebo. Permute the LLM's answers q across dates while keeping each date's bin
scale and sigma. The news placebo cannot separate the two channels because it keeps sigma; this
one breaks the answers' link to their dates and keeps only the scale. A signal that survives it
is carried by the scale, not by what the model said.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics as M
from .prereg_tests import _one_sided
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
    """Residual of s ~ 1 + v (v: one column or a 2-D array of controls)."""
    v = np.asarray(v, float)
    X = np.column_stack([np.ones(len(s)), v.reshape(len(s), -1)])
    beta, *_ = np.linalg.lstsq(X, s, rcond=None)
    return s - X @ beta


def past_return(df: pd.DataFrame, origin_ts, h: int) -> np.ndarray:
    """Log return over the h bars before each origin (ending at the last close before the forecast
    window, the same close the realized return starts from). NaN where history is too short."""
    close = df["close"].to_numpy(float)
    idx = df.index.get_indexer(pd.to_datetime(origin_ts))
    out = np.full(len(idx), np.nan)
    ok = idx - 1 - h >= 0
    out[ok] = np.log(close[idx[ok] - 1] / close[idx[ok] - 1 - h])
    return out


def decompose(frames: pd.DataFrame, controls: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """frames: asset, mode, variant, date, eq, sigma, r (+ any control columns).
    controls: {signal label: [columns to partial out]}; default {"partial_sigma": ["sigma"]}.
    Returns (per-asset, pooled) tables."""
    controls = controls or {"partial_sigma": ["sigma"]}
    extra = sorted({c for cols in controls.values() for c in cols} - {"sigma"})
    rows, pooled = [], []
    for (mode, variant), f in frames.groupby(["mode", "variant"]):
        per_date = {label: [] for label in ["raw", *controls]}
        for asset, g in f.groupby("asset"):
            e, v, r = g["eq"].to_numpy(float), g["sigma"].to_numpy(float), g["r"].to_numpy(float)
            row = {"mode": mode, "variant": variant, "asset": asset, "n": len(g),
                   "ic": M.ic(e, r), "ic_partial_sigma": partial_ic(e, r, v),
                   "ic_neg_sigma": M.ic(-v, r), "corr_signal_sigma": M.ic(e, v)}
            for c in extra:
                row[f"corr_signal_{c}"] = M.ic(e, g[c].to_numpy(float))
            dates = g["date"].to_numpy()
            per_date["raw"].append(pd.Series(_zprod(e, r), index=dates))
            for label, cols in controls.items():
                res = _resid(e, g[cols].to_numpy(float))
                if label != "partial_sigma":
                    row[f"ic_{label}"] = M.ic(res, r)
                per_date[label].append(pd.Series(_zprod(res, r), index=dates))
            rows.append(row)
        for label, parts in per_date.items():
            s = pd.concat(parts, axis=1).mean(axis=1).sort_index().to_numpy()
            t = M.dm_test(s, np.zeros_like(s), h=1)
            pooled.append({"mode": mode, "variant": variant, "signal": label, "n_dates": len(s),
                           "mean_d": float(s.mean()), "t": t["stat"], "p_two_sided": t["p"],
                           "p_one_sided_neg": _one_sided(t)})
    return pd.DataFrame(rows), pd.DataFrame(pooled)


def shuffle_answers(llm: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Answer-shuffle placebo for one asset's scores: permute q across rows, keep everything
    tied to the date (origin, sigma_h, edges, scale)."""
    rng = np.random.default_rng(seed)
    out = llm.copy()
    out["q"] = llm["q"].to_numpy()[rng.permutation(len(llm))]
    return out


def tilt(bias, alpha: float) -> tuple:
    """Scale the strength of a response bias: alpha=0 is a uniform (unbiased) answer
    distribution, alpha=1 the bias itself, alpha>1 a stronger version of it."""
    b = np.asarray(bias, float) ** alpha
    return tuple(b / b.sum())


BIAS = (0.06, 0.32, 0.30, 0.22, 0.05, 0.03, 0.02)  # bearish answer habit, close to Llama's


def simulate(n_assets: int = 9, T: int = 400, kappa: float = 0.25, info: float = 0.0,
             bias=BIAS, noise: float = 0.6, seed: int = 0) -> pd.DataFrame:
    """Synthetic panel reproducing the mechanism.

    sigma_t: log-AR(1) volatility. r_{t+h} = kappa * (sigma_t - mean) + sigma_t * eps (vol-return
    relation, no news). The 'LLM' answers q_t = softmax(log bias + noise_t + info * z_t), where
    z_t is the standardized future return (info=0 => no information). Labels use the same logit
    noise mapped to up/down/no-effect. Returns frames for modes volbins/fixbins/labels and
    variants real/shuffled (shuffled permutes the LLM answers across dates, keeping sigma_t:
    the answer-shuffle placebo).
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


# --- volatility-shock weeks ------------------------------------------------------------------

EVENTS = {"yen_carry_2024-08": "2024-08-05", "tariffs_2025-04": "2025-04-03"}


def pooled_series(frames: pd.DataFrame, controls: dict | None = None) -> pd.DataFrame:
    """Per-date contributions to the pooled test (the series whose mean decompose() tests):
    one column per (mode, variant, signal), indexed by date."""
    controls = controls or {"partial_sigma": ["sigma"]}
    cols = {}
    for (mode, variant), f in frames.groupby(["mode", "variant"]):
        parts = {label: [] for label in ["raw", *controls]}
        for _, g in f.groupby("asset"):
            e, r, dates = g["eq"].to_numpy(float), g["r"].to_numpy(float), g["date"].to_numpy()
            parts["raw"].append(pd.Series(_zprod(e, r), index=dates))
            for label, c in controls.items():
                parts[label].append(pd.Series(_zprod(_resid(e, g[c].to_numpy(float)), r), index=dates))
        for label, p in parts.items():
            cols[(mode, variant, label)] = pd.concat(p, axis=1).mean(axis=1)
    return pd.DataFrame(cols).sort_index()


def vol_rank(frames: pd.DataFrame) -> pd.Series:
    """Per date: mean across assets of sigma's percentile rank within its own asset (0-1)."""
    f = frames.drop_duplicates(["asset", "date"])
    rk = f.groupby("asset")["sigma"].rank(pct=True)
    return rk.groupby(f["date"]).mean().sort_index()


def shock_dates(frames: pd.DataFrame, q: float = 0.9) -> pd.DatetimeIndex:
    """Dates whose mean volatility rank is in the top (1 - q) share."""
    vr = vol_rank(frames)
    return pd.DatetimeIndex(vr.index[vr >= vr.quantile(q)])


def event_dates(dates, events: dict = EVENTS, days: int = 7) -> dict:
    """Forecast dates within +-days of each named event."""
    d = pd.DatetimeIndex(dates)
    out = {}
    for name, day in events.items():
        e = pd.Timestamp(day)
        if d.tz is not None:
            e = e.tz_localize(d.tz)
        out[name] = d[(d > e - pd.Timedelta(days=days)) & (d <= e + pd.Timedelta(days=days))]
    return out


def drop_test(s: pd.Series, drop, B: int = 1000, seed: int = 0) -> dict:
    """Pooled t with and without the given dates, and how unusual the change is compared with
    dropping the same number of random dates (p_random: share of random drops whose t ends at
    least as high, i.e. at least as far from a negative signal)."""
    s = s.dropna()
    mask = s.index.isin(pd.DatetimeIndex(drop))
    k = int(mask.sum())
    x = s.to_numpy(float)

    def tstat(v):
        return M.dm_test(v, np.zeros_like(v), h=1)["stat"]

    t_all, t_kept = tstat(x), tstat(x[~mask])
    rng = np.random.default_rng(seed)
    rand = np.array([tstat(np.delete(x, rng.choice(len(x), k, replace=False))) for _ in range(B)])
    return {"n_dates": len(x), "n_dropped": k, "t_all": t_all, "t_kept": t_kept,
            "share_of_sum": float(x[mask].sum() / x.sum()) if x.sum() else float("nan"),
            "t_random_mean": float(rand.mean()), "p_random": float((rand >= t_kept).mean())}
