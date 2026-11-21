"""Confirmation study (see PREREGISTRATION.md): is the LLM's news-implied expected return
negatively correlated with realized returns on held-out assets / periods?
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics as M
from .fusion import llm_only_samples


def asset_frame(cfg: dict, df: pd.DataFrame, llm: pd.DataFrame, n: int = 64) -> pd.DataFrame:
    """One row per (variant, origin): E_Q[r] and realized horizon log return."""
    if llm.empty:
        return pd.DataFrame()
    close = df["close"].to_numpy(float)
    h = int(cfg["pred_len"])
    idx = df.index.get_indexer(pd.to_datetime(llm["origin_ts"]))
    if (idx < 1).any():
        raise ValueError(f"{cfg['asset']}: LLM origins not found in price data")
    ok = idx + h - 1 < len(close)
    llm, idx = llm[ok], idx[ok]
    Q = np.stack(llm["q"].to_numpy()).astype(float)
    sigma = llm["sigma_h"].to_numpy(float)
    if Q.shape[1] == 3:  # labels: signal = P(up) - P(down)
        eq = Q[:, 0] - Q[:, 1]
    else:  # ranges: expected return; tails bounded by the bin scale actually shown to the LLM
        E = np.stack(llm["edges"].to_numpy()).astype(float)
        scale = llm["scale"].to_numpy(float) if "scale" in llm else sigma
        eq = llm_only_samples(Q, E, scale, n).mean(1)
    return pd.DataFrame({"asset": cfg["asset"], "variant": llm["variant"].to_numpy(),
                         "origin_ts": pd.to_datetime(llm["origin_ts"]).to_numpy(),
                         "date": pd.to_datetime(llm["origin_ts"]).dt.normalize().to_numpy(),
                         "eq": eq, "sigma": sigma, "r": np.log(close[idx + h - 1] / close[idx - 1])})


def _one_sided(t: dict) -> float:
    """p-value for H1: mean < 0, from a two-sided result."""
    if np.isnan(t["stat"]):
        return float("nan")
    return t["p"] / 2 if t["stat"] < 0 else 1 - t["p"] / 2


def per_asset(frames: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (asset, variant), g in frames.groupby(["asset", "variant"]):
        d = _zprod(g)
        t = M.dm_test(d, np.zeros_like(d), h=1)
        rows.append({"asset": asset, "variant": variant, "n": len(g), "ic": M.ic(g["eq"], g["r"]),
                     "rank_ic": M.rank_ic(g["eq"], g["r"]), "t": t["stat"], "p_one_sided": _one_sided(t)})
    return pd.DataFrame(rows)


def _zprod(g: pd.DataFrame) -> np.ndarray:
    x, y = g["eq"].to_numpy(float), g["r"].to_numpy(float)
    if x.std() == 0 or y.std() == 0:
        return np.zeros(len(g))
    return (x - x.mean()) / x.std() * (y - y.mean()) / y.std()


def family_test(frames: pd.DataFrame, variant: str) -> dict:
    """Fama-MacBeth: per-asset z-scored cross products, averaged across assets per date,
    one-sided Newey-West test of mean < 0 (lag 0; non-overlapping targets)."""
    f = frames[frames["variant"] == variant]
    if f.empty:
        return {"variant": variant, "n_dates": 0, "n_assets": 0, "mean_d": np.nan, "t": np.nan, "p_one_sided": np.nan}
    parts = []
    for _, g in f.groupby("asset"):
        parts.append(pd.Series(_zprod(g), index=g["date"].to_numpy()))
    panel = pd.concat(parts, axis=1)
    series = panel.mean(axis=1, skipna=True).sort_index().to_numpy()
    t = M.dm_test(series, np.zeros_like(series), h=1)
    return {"variant": variant, "n_dates": len(series), "n_assets": f["asset"].nunique(),
            "mean_d": float(series.mean()), "t": t["stat"], "p_one_sided": _one_sided(t)}


def perm_test(frames: pd.DataFrame, variant: str, B: int = 2000, seed: int = 0, col: str = "eq") -> dict:
    """Permutation version of family_test: within each asset, permute the signal across dates
    (breaking any signal-return link, keeping each asset's return series and the date panel
    intact), recompute the pooled mean of z(signal)*z(r). No normality or HAC approximation."""
    f = frames[frames["variant"] == variant]
    if f.empty:
        return {"variant": variant, "obs": np.nan, "p_one_sided_neg": np.nan, "p_two_sided": np.nan, "B": B}
    codes, uniq = pd.factorize(pd.to_datetime(f["date"]), sort=True)
    f = f.assign(_code=codes)
    blocks = []
    for _, g in f.groupby("asset"):
        x, y = g[col].to_numpy(float), g["r"].to_numpy(float)
        if x.std() == 0 or y.std() == 0:
            continue
        blocks.append(((x - x.mean()) / x.std(), (y - y.mean()) / y.std(), g["_code"].to_numpy()))
    n_dates = len(uniq)
    counts = np.zeros(n_dates)
    for _, _, c in blocks:
        counts += np.bincount(c, minlength=n_dates)
    have = counts > 0

    def pooled(perms):
        sums = np.zeros(n_dates)
        for (zx, zy, c), pm in zip(blocks, perms):
            sums += np.bincount(c, weights=(zx if pm is None else zx[pm]) * zy, minlength=n_dates)
        return float((sums[have] / counts[have]).mean())

    obs = pooled([None] * len(blocks))
    rng = np.random.default_rng(seed)
    null = np.array([pooled([rng.permutation(len(b[0])) for b in blocks]) for _ in range(B)])
    return {"variant": variant, "obs": obs, "B": B,
            "p_one_sided_neg": float((1 + (null <= obs).sum()) / (B + 1)),
            "p_two_sided": float((1 + (np.abs(null) >= abs(obs)).sum()) / (B + 1))}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for k, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - k) * p))
        out[name] = running
    return out
