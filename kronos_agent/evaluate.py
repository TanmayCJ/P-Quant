"""Phase-1 baseline evaluation: Kronos ensemble vs simple probabilistic baselines.

Target: horizon log return r = log(close[i+h-1] / close[i-1]).
Methods (all ensembles of the same size n, so CRPS is comparable):
  kronos  - Kronos sampled paths
  gauss   - N(0, sigma_ctx * sqrt(h)), sigma from context log returns
  boot    - iid bootstrap of context 1-step log returns, summed over h
  rw      - point forecast 0 (CRPS == absolute error)
Volatility target: std of realized 1-step log returns over the horizon.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics as M
from .pipeline import decode_paths

LEVELS = (0.5, 0.8, 0.9)


def build_eval_frame(store_df: pd.DataFrame, data: pd.DataFrame, lookback: int,
                     seed: int = 0) -> pd.DataFrame:
    close = data["close"].to_numpy(float)
    rng = np.random.default_rng(seed)
    out = []
    for rec in store_df.to_dict("records"):
        i, h, n = rec["row"], rec["pred_len"], rec["n_paths"]
        if data.index[i] != pd.Timestamp(rec["origin_ts"]):
            raise ValueError(f"row/timestamp mismatch for {rec['key']}: data changed since sampling")
        c0 = rec["last_close"]
        if not np.isclose(close[i - 1], c0, rtol=1e-6):
            raise ValueError(f"last_close mismatch for {rec['key']}")
        realized = close[i:i + h]
        paths = decode_paths(rec, "close").astype(float)
        lp = np.log(np.concatenate([np.full((n, 1), c0), paths], axis=1))
        lr = np.log(np.concatenate([[c0], realized]))
        ctx_ret = np.diff(np.log(close[i - lookback:i]))
        sigma = ctx_ret.std(ddof=1)
        out.append({
            "key": rec["key"], "asset": rec["asset"], "origin_ts": rec["origin_ts"],
            "period": rec["period"], "h": h,
            "r_true": lr[-1] - lr[0],
            "gap": np.log(c0) - np.log(close[i - lookback:i]).mean(),  # last price vs context mean
            "kronos": lp[:, -1] - lp[:, 0],
            "gauss": rng.normal(0.0, sigma * np.sqrt(h), n),
            "boot": rng.choice(ctx_ret, size=(n, h)).sum(axis=1),
            "vol_true": np.diff(lr).std(ddof=1) if h > 1 else abs(lr[-1] - lr[0]),
            "vol_kronos": np.diff(lp, axis=1).std(axis=1, ddof=1).mean() if h > 1
                          else np.abs(lp[:, -1] - lp[:, 0]).mean(),
            "vol_hist": sigma,
        })
    return pd.DataFrame(out)


def summarize(ev: pd.DataFrame, dm_h: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (metrics table per period x method, significance table per period)."""
    rows, tests = [], []
    for period, g in ev.groupby("period", sort=False):
        y = g["r_true"].to_numpy()
        losses = {}
        for m in ("kronos", "gauss", "boot"):
            X = np.stack(g[m].to_numpy())
            mean = X.mean(axis=1)
            crps = M.crps_ensemble(X, y)
            losses[m] = {"crps": crps, "ae": np.abs(mean - y)}
            row = {"period": period, "method": m, "n": len(y), "crps": crps.mean(),
                   "mae": np.abs(mean - y).mean(), "ic": M.ic(mean, y),
                   "rank_ic": M.rank_ic(mean, y), "dir_acc": M.directional_accuracy(mean, y)}
            for lv in LEVELS:
                im = M.interval_metrics(X, y, lv)
                row[f"cov{int(lv * 100)}"] = im["coverage"]
                row[f"wid{int(lv * 100)}"] = im["width"]
            rows.append(row)
        losses["rw"] = {"crps": np.abs(y), "ae": np.abs(y)}
        rows.append({"period": period, "method": "rw", "n": len(y), "crps": np.abs(y).mean(),
                     "mae": np.abs(y).mean()})

        for base in ("gauss", "boot", "rw"):
            for loss in ("crps", "ae"):
                t = M.dm_test(losses["kronos"][loss], losses[base][loss], h=dm_h)
                lo, hi = M.block_bootstrap_ci(losses["kronos"][loss] - losses[base][loss],
                                              block=max(5, dm_h))
                tests.append({"period": period, "kronos_vs": base, "loss": loss, **t,
                              "ci_lo": lo, "ci_hi": hi})
        vt = g["vol_true"].to_numpy()
        for vm in ("vol_kronos", "vol_hist"):
            rows.append({"period": period, "method": vm, "n": len(vt),
                         "mae": np.abs(g[vm].to_numpy() - vt).mean(),
                         "ic": M.ic(g[vm].to_numpy(), vt)})
        t = M.dm_test(np.abs(g["vol_kronos"].to_numpy() - vt), np.abs(g["vol_hist"].to_numpy() - vt), h=dm_h)
        tests.append({"period": period, "kronos_vs": "vol_hist", "loss": "vol_ae", **t})
    return pd.DataFrame(rows), pd.DataFrame(tests)
