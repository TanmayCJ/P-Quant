"""Calibration report: correct Kronos's bias and check whether its uncertainty is honest.

It covers:
- bias diagnostics (how far the mean forecast is off);
- calibration methods: raw, recal, spread, emos, and cqr / aci intervals on raw and emos;
- coverage (do 80% intervals contain 80% of outcomes?), also by market condition;
- a trade-or-abstain backtest, selective-prediction curves and a per-forecast decision log.

Every correction restarts at the start of each window (before / after the training cutoff),
and all methods are scored on the same forecasts (t >= warmup + delay - 1), so the
comparison is fair.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import calibration as C
from . import metrics as M
from . import policy as P

RC_GRID = np.round(np.arange(0.1, 1.01, 0.1), 2)
INTERVAL_BASES = ("raw", "emos", "emos_gap")  # dists that also get cqr / aci intervals


def _mean(X, W):
    return X.mean(axis=1) if W is None else (X * W).sum(1) / W.sum(1)


def interval_key(dist: str, method: str) -> str:
    """'quantile' = plain quantiles of the dist; cqr/aci are conformal wrappers around it."""
    if method == "quantile":
        return dist
    return method if dist == "raw" else f"{dist}+{method}"


def calibrate(X, y, cfg: dict, delay: int, gap=None):
    """gap: per-origin log(last price / context mean), known ex ante; enables 'emos_gap'."""
    cal = cfg["calib"]
    kw = dict(delay=delay, window=cal["window"], warmup=cal["warmup"])
    dists = {"raw": (X, None)}
    if cfg["features"]["conformal"]:
        dists["recal"] = C.pit_recalibrate(X, y, **kw)
        dists["spread"] = C.spread_rescale(X, y, **kw)
        dists["emos"] = C.emos_affine(X, y, **kw)
        if gap is not None:
            dists["emos_gap"] = C.emos_affine(X, y, Z=gap, **kw)
    # conformal wrappers around a corrected dist only learn from origins where it is active
    ready = cal["warmup"] + delay - 1
    intervals = {}
    for lv in cal["levels"]:
        for name, (Xd, Wd) in dists.items():
            intervals[(name, lv)] = C.raw_interval(Xd, lv, Wd)
        if cfg["features"]["conformal"]:
            for b in INTERVAL_BASES:
                if b not in dists:
                    continue
                Xd, Wd = dists[b]
                st = 0 if b == "raw" else ready
                intervals[(interval_key(b, "cqr"), lv)] = C.cqr_interval(Xd, y, lv, W=Wd, start=st, **kw)
                intervals[(interval_key(b, "aci"), lv)] = C.aci_interval(
                    Xd, y, lv, delay=delay, gamma=cal["gamma"], W=Wd, start=st)[:2]
    return dists, intervals


def _score_distributions(period, base, dists, y, scored, gap):
    """CRPS / PIT row per distribution, a bias row per distribution, and the per-forecast CRPS."""
    calib_rows, bias_rows, crps = [], [], {}
    n = int(scored.sum())
    for name, (samples, weights) in dists.items():
        w = None if weights is None else weights[scored]
        crps[name] = M.crps_ensemble(samples[scored], y[scored], w)
        calib_rows.append({"period": period, "base": base, "method": name, "level": np.nan, "n": n,
                           "crps": crps[name].mean(),
                           "pit_ks": M.pit_ks(M.pit(samples[scored], y[scored], w))})
        mean = _mean(samples, weights)
        error = mean - y
        row = {"period": period, "base": base, "dist": name, "n": n,
               "bias": error[scored].mean(), "mae": np.abs(error)[scored].mean(),
               "ic": M.ic(mean[scored], y[scored]),
               "dir_acc": M.directional_accuracy(mean[scored], y[scored])}
        if gap is not None:
            row["corr_mu_gap"] = M.ic(mean[scored], gap[scored])
            row["slope_mu_gap"] = np.polyfit(gap[scored], mean[scored], 1)[0]
        bias_rows.append(row)
    return calib_rows, bias_rows, crps


def _score_intervals(period, base, intervals, samples, y, scored, vol_group):
    """Coverage, width and interval score per interval, overall and by volatility tercile."""
    calib_rows, cond_rows = [], []
    median = M.weighted_quantile(samples, [0.5])[:, 0]
    for (name, level), (lo, hi) in intervals.items():
        covered = (y >= lo) & (y <= hi)
        width = hi - lo
        calib_rows.append({"period": period, "base": base, "method": name, "level": level,
                           "n": int(scored.sum()), "coverage": covered[scored].mean(),
                           "width": np.median(width[scored]),
                           "winkler": M.interval_score(lo[scored], hi[scored], y[scored], level).mean(),
                           "spread_skill": M.ic(width[scored], np.abs(y - median)[scored])})
        for group in ("low", "mid", "high"):
            rows = scored & (vol_group == group)
            cond_rows.append({"period": period, "base": base, "method": name, "level": level,
                              "vol_tercile": group, "n": int(rows.sum()), "coverage": covered[rows].mean(),
                              "width": np.median(width[rows])})
    return calib_rows, cond_rows


def _chosen_distribution(cfg, dists):
    """The corrected Kronos distribution and interval the trading rule uses."""
    use_corrections = cfg["features"]["conformal"]
    dist = cfg["policy"]["dist"] if use_corrections else "raw"
    if dist not in dists:  # e.g. emos_gap without a gap column
        dist = "emos"
    method = cfg["policy"]["interval_method"] if use_corrections else "quantile"
    return dist, interval_key(dist, method)


def _trading_rules(period, g, mean, lo, hi, y, scored, cfg, delay, dist, interval_name):
    """Backtest each trade-or-abstain rule; also return the decision log of the 'edge_width' rule."""
    cost = cfg["cost_bps"] / 1e4
    periods_per_year = cfg["bars_per_year"] / (cfg["stride"] * delay)
    rules = P.RULES if cfg["features"]["abstain"] else ("always", "buy_hold")
    policy_rows, decisions = [], None
    for rule in rules:
        pos, reason = P.positions(rule, mean, lo, hi, cost, width_q=cfg["policy"]["width_q"])
        result = P.backtest(pos[scored], y[scored], cost, step=delay, periods_per_year=periods_per_year)
        shares = pd.Series(reason[scored]).value_counts(normalize=True).to_dict()
        policy_rows.append({"period": period, "rule": rule, "dist": dist, "interval": interval_name, **result,
                            **{f"frac_{k}": v for k, v in shares.items()}})
        if rule == "edge_width":
            decisions = pd.DataFrame({
                "period": period, "origin_ts": g["origin_ts"], "mu": mean, "lo": lo, "hi": hi,
                "width": hi - lo, "pos": pos, "reason": reason, "r_true": y, "scored": scored})
    return policy_rows, decisions


def _selective_prediction(period, dist, mean, lo, hi, y, scored, seed):
    """Risk-coverage curves: does acting only on the most confident forecasts raise accuracy?"""
    ok = scored & (y != 0) & (mean != 0)
    correct = (np.sign(mean) == np.sign(y))[ok].astype(float)
    width = (hi - lo)[ok]
    confidence = {"abs_mu": np.abs(mean[ok]), "neg_width": -width, "mu_over_width": np.abs(mean[ok]) / width,
                  "random": np.random.default_rng(seed).random(ok.sum())}
    rows = []
    for name, conf in confidence.items():
        coverage, accuracy, aurc = P.risk_coverage(conf, correct, RC_GRID)
        for cv, acc in zip(coverage, accuracy):
            rows.append({"period": period, "dist": dist, "confidence": name, "coverage": cv,
                         "sel_acc": acc, "aurc": aurc})
    return rows


def _crps_tests(period, crps, delay):
    """Each Kronos distribution against the raw and gap-corrected Gaussian: DM test and bootstrap CI."""
    rows = []
    for (base1, dist1), loss1 in crps.items():
        if base1 != "kronos":
            continue
        for (base2, dist2), loss2 in crps.items():
            if base2 == "gauss" and dist2 in ("raw", "emos_gap"):
                test = M.dm_test(loss1, loss2, h=delay)
                ci_lo, ci_hi = M.block_bootstrap_ci(loss1 - loss2, block=max(5, delay))
                rows.append({"period": period, "kronos": dist1, "vs_gauss": dist2, **test,
                             "ci_lo": ci_lo, "ci_hi": ci_hi})
    return rows


def calibration_report(ev: pd.DataFrame, cfg: dict, delay: int) -> dict[str, pd.DataFrame]:
    if not cfg.get("bars_per_year"):
        raise ValueError("config needs bars_per_year (e.g. 252 for daily stocks, 8760 for hourly crypto)")
    first_scored = cfg["calib"]["warmup"] + delay - 1
    calib, cond, policy, risk_cov, bias, decisions, tests = [], [], [], [], [], [], []

    for period, g in ev.groupby("period", sort=False):
        g = g.sort_values("origin_ts").reset_index(drop=True)
        y = g["r_true"].to_numpy(float)
        scored = np.arange(len(y)) >= first_scored
        if scored.sum() < 20:
            continue
        vol_group = pd.qcut(g["vol_hist"].rank(method="first"), 3, labels=["low", "mid", "high"]).to_numpy()
        gap = g["gap"].to_numpy(float) if "gap" in g else None
        crps = {}

        for base in ("kronos", "gauss"):
            samples = np.stack(g[base].to_numpy()).astype(float)
            dists, intervals = calibrate(samples, y, cfg, delay, gap)
            rows, bias_rows, base_crps = _score_distributions(period, base, dists, y, scored, gap)
            calib += rows
            bias += bias_rows
            crps.update({(base, name): loss for name, loss in base_crps.items()})
            rows, cond_rows = _score_intervals(period, base, intervals, samples, y, scored, vol_group)
            calib += rows
            cond += cond_rows

            if base != "kronos":
                continue
            dist, interval_name = _chosen_distribution(cfg, dists)
            mean = _mean(*dists[dist])
            lo, hi = intervals[(interval_name, cfg["policy"]["level"])]
            rows, log = _trading_rules(period, g, mean, lo, hi, y, scored, cfg, delay, dist, interval_name)
            policy += rows
            if log is not None:
                decisions.append(log)
            q_lo, q_hi = intervals[(dist, cfg["policy"]["level"])]
            risk_cov += _selective_prediction(period, dist, mean, q_lo, q_hi, y, scored, cfg["seed"])

        tests += _crps_tests(period, crps, delay)

    return {"bias": pd.DataFrame(bias), "crps_tests": pd.DataFrame(tests), "calibration": pd.DataFrame(calib),
            "conditional_coverage": pd.DataFrame(cond), "policy": pd.DataFrame(policy),
            "risk_coverage": pd.DataFrame(risk_cov),
            "decisions": pd.concat(decisions, ignore_index=True) if decisions else pd.DataFrame()}
