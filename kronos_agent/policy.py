"""Trade / abstain policy, cost-aware backtest and selective-prediction curves.

Returns are horizon log returns; costs are per side in log-return units.
"""
from __future__ import annotations

import numpy as np

RULES = ("always", "edge", "edge_width", "interval", "buy_hold")


def positions(rule: str, mu, lo, hi, cost_side: float, width_q: float = 0.5,
              width_warmup: int = 20):
    """Return (pos in {-1,0,1}, reason[str]). Widths are known ex ante, so the width gate
    compares width_t with widths of origins < t (no outcome delay needed)."""
    mu, lo, hi = (np.asarray(a, float) for a in (mu, lo, hi))
    T = len(mu)
    rt = 2 * cost_side
    width = hi - lo
    pos = np.sign(mu)
    reason = np.full(T, "trade", dtype=object)
    if rule == "buy_hold":
        return np.ones(T), np.full(T, "trade", dtype=object)
    if rule == "always":
        return pos, reason
    if rule in ("edge", "edge_width"):
        no_edge = np.abs(mu) <= rt
        pos[no_edge] = 0
        reason[no_edge] = "no_edge"
        if rule == "edge_width":
            for t in range(T):
                if t >= width_warmup and pos[t] != 0 and width[t] > np.quantile(width[:t], width_q):
                    pos[t] = 0
                    reason[t] = "too_uncertain"
        return pos, reason
    if rule == "interval":
        long_, short = lo > rt, hi < -rt
        pos = np.where(long_, 1.0, np.where(short, -1.0, 0.0))
        reason[pos == 0] = "interval_spans_zero"
        return pos, reason
    raise ValueError(f"unknown rule {rule}")


def backtest(pos, r, cost_side: float, step: int, periods_per_year: float) -> dict:
    """Non-overlapping backtest averaged over the `step` possible start offsets."""
    pos, r = np.asarray(pos, float), np.asarray(r, float)
    res = []
    for off in range(step):
        p, ret = pos[off::step], r[off::step]
        if len(p) < 2:
            continue
        turnover = np.abs(np.diff(np.concatenate([[0.0], p])))
        pnl = p * ret - cost_side * turnover
        cum = np.cumsum(pnl)
        traded = p != 0
        sd = pnl.std(ddof=1)
        res.append({
            "sharpe": pnl.mean() / sd * np.sqrt(periods_per_year) if sd > 0 else np.nan,
            "ann_ret": pnl.mean() * periods_per_year,
            "total_logret": cum[-1],
            "max_dd": np.max(np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:] - cum),
            "hit_rate": (np.sign(p[traded]) == np.sign(ret[traded])).mean() if traded.any() else np.nan,
            "trade_frac": traded.mean(),
            "turnover": turnover.mean(),
        })
    if not res:
        return {}
    out = {}
    for k in res[0]:
        vals = np.array([d[k] for d in res], float)
        out[k] = float(vals[~np.isnan(vals)].mean()) if (~np.isnan(vals)).any() else float("nan")
    return out


def risk_coverage(confidence, correct, grid=None) -> tuple[np.ndarray, np.ndarray, float]:
    """Selective accuracy as we keep the most-confident fraction. Returns (coverage, acc, AURC)."""
    confidence, correct = np.asarray(confidence, float), np.asarray(correct, float)
    order = np.argsort(-confidence, kind="mergesort")
    c = correct[order]
    k = np.arange(1, len(c) + 1)
    acc = np.cumsum(c) / k
    aurc = float(np.mean(1 - acc))
    cov = k / len(c)
    if grid is not None:
        idx = np.clip(np.ceil(np.asarray(grid) * len(c)).astype(int) - 1, 0, len(c) - 1)
        return np.asarray(grid), acc[idx], aurc
    return cov, acc, aurc
