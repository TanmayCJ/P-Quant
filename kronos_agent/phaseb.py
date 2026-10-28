"""Phase B: Kronos (and comparison models) across many assets + a synthetic-trend mechanism test.

Cross-asset report: per asset and run (model x lookback) -> reversion slope of the mean forecast
on the context gap, CRPS raw / EMOS(gap)-corrected / Gaussian baseline; pooled Diebold-Mariano
across assets on volatility-normalised CRPS differences (Fama-MacBeth style, one row per date).

Synthetic test: random-walk prices with a known drift (no mean reversion by construction). A
forecaster that reverts to the window mean shows a negative slope on the gap where the truth
has slope zero, and forecasts against the trend.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics as M
from .conformal import emos_affine
from .data import COLS


def asset_metrics(ev: pd.DataFrame, cfg: dict, delay: int, k_mode: str = "mean_ratio") -> tuple[dict, pd.DataFrame]:
    """ev: one asset/run eval frame (evaluate.build_eval_frame). Returns (summary row, per-date losses).
    k_mode: EMOS spread estimator ('mean_ratio' = pre-specified, 'pooled' = robustness check)."""
    ev = ev.sort_values("origin_ts").reset_index(drop=True)
    y = ev["r_true"].to_numpy(float)
    gap = ev["gap"].to_numpy(float)
    X = np.stack(ev["kronos"].to_numpy()).astype(float)   # column holds whichever model was sampled
    G = np.stack(ev["gauss"].to_numpy()).astype(float)
    cal = cfg["calib"]
    E, _ = emos_affine(X, y, delay=delay, window=cal["window"], warmup=cal["warmup"], Z=gap, k_mode=k_mode)
    m = np.arange(len(y)) >= cal["warmup"] + delay - 1
    mu = X.mean(1)
    c_raw, c_emos, c_g = (M.crps_ensemble(A[m], y[m]) for A in (X, E, G))
    scale = c_g.mean()
    b, a = np.polyfit(gap[m], mu[m], 1)
    # influence of each date on the slope (sums to 0); summed over assets -> Driscoll-Kraay SE
    xd = gap[m] - gap[m].mean()
    psi = xd * (mu[m] - a - b * gap[m]) / (xd @ xd)
    row = {"n": int(m.sum()), "bias": float((mu - y)[m].mean()),
           "slope_gap": float(b), "corr_gap": M.ic(mu[m], gap[m]),
           "crps_raw": c_raw.mean(), "crps_emos": c_emos.mean(), "crps_gauss": scale,
           "pit_ks_emos": M.pit_ks(M.pit(E[m], y[m])),
           "emos_beats_gauss": bool(c_emos.mean() < scale)}
    losses = pd.DataFrame({"date": pd.to_datetime(ev["origin_ts"][m]).dt.normalize().to_numpy(),
                           "d_raw": (c_raw - c_g) / scale, "d_emos": (c_emos - c_g) / scale,
                           "psi_slope": psi})
    return row, losses


def slope_influence(psi: pd.DataFrame) -> pd.Series:
    """psi: rows (asset, date, psi_slope) of one run -> per-date influence on the mean-across-assets
    slope (sum over assets / n_assets), sorted by date."""
    w = psi.pivot_table(index="date", columns="asset", values="psi_slope", aggfunc="sum").fillna(0.0)
    return w.sort_index().mean(axis=1)


def slope_dk(u: pd.Series, lags=(None, 10, 20)) -> dict:
    """Driscoll-Kraay SEs for a mean slope from its per-date influence u (lag None = NW rule of thumb)."""
    T = len(u)
    out = {"n_dates": T}
    for L in lags:
        lag = M.nw_auto_lag(T) if L is None else L
        out["se_dk" if L is None else f"se_dk_L{L}"] = M.hac_se_sum(u.to_numpy(), lag)
    out["lag_auto"] = M.nw_auto_lag(T)
    return out


def pooled_dm(losses: list[pd.DataFrame], col: str) -> dict:
    """Average the normalised loss difference across assets per date, then a DM/HAC t-test (lag 0)."""
    s = pd.concat([l.set_index("date")[col] for l in losses], axis=1).mean(axis=1).sort_index().to_numpy()
    t = M.dm_test(s, np.zeros_like(s), h=1)
    return {"n_dates": len(s), "mean_rel_diff": float(s.mean()), "t": t["stat"], "p": t["p"]}


def synthetic_series(n_bars: int, drift: float, vol: float = 0.012, seed: int = 0,
                     start: str = "2020-01-01") -> pd.DataFrame:
    """Daily OHLCV random walk with drift (per bar, log units). No mean reversion by construction."""
    rng = np.random.default_rng(seed)
    lr = rng.normal(drift, vol, n_bars)
    close = 100 * np.exp(np.cumsum(lr))
    open_ = np.concatenate([[100.0], close[:-1]]) * np.exp(rng.normal(0, vol / 4, n_bars))
    hi = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, vol / 3, n_bars)))
    lo = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, vol / 3, n_bars)))
    volume = rng.lognormal(16, 0.3, n_bars)
    df = pd.DataFrame({"open": open_, "high": hi, "low": lo, "close": close, "volume": volume},
                      index=pd.bdate_range(start, periods=n_bars))
    df["amount"] = df["volume"] * df[["open", "high", "low", "close"]].mean(axis=1)
    df.index.name = "timestamp"
    return df[COLS]


def run_synthetic(sampler, lookback: int, horizon: int = 5, drifts=(-0.002, 0.0, 0.002),
                  n_series: int = 40, n_paths: int = 32, seed: int = 0, log=print) -> pd.DataFrame:
    """One origin per synthetic series at bar `lookback`; returns per-series forecast vs truth."""
    rows = []
    for d in drifts:
        for k in range(n_series):
            s = seed * 100000 + int(round(d * 1e6)) * 100 + k
            df = synthetic_series(lookback + horizon, d, seed=abs(s))
            ctx, y_ts = df.iloc[:lookback], df.index[lookback:]
            paths = sampler.sample(ctx, y_ts, n_paths, seed=abs(s))["close"].astype(float)
            c0, c = float(ctx["close"].iloc[-1]), ctx["close"].to_numpy(float)
            rows.append({"drift": d, "series": k, "fc": float(np.log(paths[:, -1] / c0).mean()),
                         "true_mean": d * horizon, "realized": float(np.log(df["close"].iloc[-1] / c0)),
                         "gap": float(np.log(c0) - np.log(c).mean())})
        log(f"  drift {d:+.4f}: {n_series} series done")
    return pd.DataFrame(rows)


def summarize_synthetic(res: pd.DataFrame) -> pd.DataFrame:
    """Per drift and pooled: mean forecast vs true drift, and slope of forecast on gap (truth: 0)."""
    out = []
    for d, g in list(res.groupby("drift")) + [("all", res)]:
        out.append({"drift": d, "n": len(g), "mean_fc": g["fc"].mean(), "true_mean": g["true_mean"].mean(),
                    "slope_fc_gap": float(np.polyfit(g["gap"], g["fc"], 1)[0]),
                    "corr_fc_gap": M.ic(g["fc"], g["gap"]),
                    "share_against_trend": float((np.sign(g["fc"]) == -np.sign(g["true_mean"])).mean())
                    if d != 0.0 and d != "all" else float("nan")})
    return pd.DataFrame(out)
