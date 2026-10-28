"""Phase 2 report: bias diagnostics, calibration (raw / recal / spread / emos, plus cqr / aci
intervals on raw and on emos), conditional coverage, trade-or-abstain backtests,
selective-prediction curves and a per-origin decision log.

Calibration state is reset at the start of every window (pre/post cutoff), and all
methods are scored on the same index set (t >= warmup + delay - 1) for fairness.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import conformal as C
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


def phase2_report(ev: pd.DataFrame, cfg: dict, delay: int) -> dict[str, pd.DataFrame]:
    if not cfg.get("bars_per_year"):
        raise ValueError("config needs bars_per_year for phase 2")
    start = cfg["calib"]["warmup"] + delay - 1
    cost = cfg["cost_bps"] / 1e4
    ppy = cfg["bars_per_year"] / (cfg["stride"] * delay)
    pol = cfg["policy"]
    calib, cond, pols, rc, bias, decisions, tests = [], [], [], [], [], [], []

    for period, g in ev.groupby("period", sort=False):
        g = g.sort_values("origin_ts").reset_index(drop=True)
        y = g["r_true"].to_numpy(float)
        m = np.arange(len(y)) >= start
        if m.sum() < 20:
            continue
        terc = pd.qcut(g["vol_hist"].rank(method="first"), 3, labels=["low", "mid", "high"]).to_numpy()
        gap = g["gap"].to_numpy(float) if "gap" in g else None
        crps_by = {}

        for base in ("kronos", "gauss"):
            X = np.stack(g[base].to_numpy()).astype(float)
            dists, intervals = calibrate(X, y, cfg, delay, gap)
            for name, (Xd, Wd) in dists.items():
                Wm = None if Wd is None else Wd[m]
                crps_by[(base, name)] = M.crps_ensemble(Xd[m], y[m], Wm)
                calib.append({"period": period, "base": base, "method": name, "level": np.nan, "n": int(m.sum()),
                              "crps": crps_by[(base, name)].mean(),
                              "pit_ks": M.pit_ks(M.pit(Xd[m], y[m], Wm))})
                mu_d = _mean(Xd, Wd)
                row = {"period": period, "base": base, "dist": name, "n": int(m.sum()),
                       "bias": (mu_d - y)[m].mean(), "mae": np.abs(mu_d - y)[m].mean(),
                       "ic": M.ic(mu_d[m], y[m]), "dir_acc": M.directional_accuracy(mu_d[m], y[m])}
                if gap is not None:
                    row.update(corr_mu_gap=M.ic(mu_d[m], gap[m]), slope_mu_gap=np.polyfit(gap[m], mu_d[m], 1)[0])
                bias.append(row)
            mid = M.weighted_quantile(X, [0.5])[:, 0]
            for (name, lv), (lo, hi) in intervals.items():
                cov = (y >= lo) & (y <= hi)
                width = hi - lo
                calib.append({"period": period, "base": base, "method": name, "level": lv, "n": int(m.sum()),
                              "coverage": cov[m].mean(), "width": np.median(width[m]),
                              "winkler": M.interval_score(lo[m], hi[m], y[m], lv).mean(),
                              "spread_skill": M.ic(width[m], np.abs(y - mid)[m])})
                for t in ("low", "mid", "high"):
                    k = m & (terc == t)
                    cond.append({"period": period, "base": base, "method": name, "level": lv,
                                 "vol_tercile": t, "n": int(k.sum()), "coverage": cov[k].mean(),
                                 "width": np.median(width[k])})

            if base != "kronos":
                continue
            # --- policy on the chosen (corrected) Kronos distribution ---
            dist = pol["dist"] if cfg["features"]["conformal"] else "raw"
            if dist not in dists:  # e.g. emos_gap without a gap column
                dist = "emos"
            imeth = pol["interval_method"] if cfg["features"]["conformal"] else "quantile"
            ikey = interval_key(dist, imeth)
            Xp, Wp = dists[dist]
            mu = _mean(Xp, Wp)
            lo, hi = intervals[(ikey, pol["level"])]
            rules = P.RULES if cfg["features"]["abstain"] else ("always", "buy_hold")
            for rule in rules:
                pos, reason = P.positions(rule, mu, lo, hi, cost, width_q=pol["width_q"])
                bt = P.backtest(pos[m], y[m], cost, step=delay, periods_per_year=ppy)
                counts = pd.Series(reason[m]).value_counts(normalize=True).to_dict()
                pols.append({"period": period, "rule": rule, "dist": dist, "interval": ikey, **bt,
                             **{f"frac_{k}": v for k, v in counts.items()}})
                if rule == "edge_width":
                    decisions.append(pd.DataFrame({
                        "period": period, "origin_ts": g["origin_ts"], "mu": mu, "lo": lo, "hi": hi,
                        "width": hi - lo, "pos": pos, "reason": reason, "r_true": y, "scored": m}))

            # --- selective prediction: is (corrected) Kronos uncertainty informative? ---
            qlo, qhi = intervals[(dist, pol["level"])]
            ok = m & (y != 0) & (mu != 0)
            correct = (np.sign(mu) == np.sign(y))[ok].astype(float)
            w = (qhi - qlo)[ok]
            confs = {"abs_mu": np.abs(mu[ok]), "neg_width": -w, "mu_over_width": np.abs(mu[ok]) / w,
                     "random": np.random.default_rng(cfg["seed"]).random(ok.sum())}
            for cname, conf in confs.items():
                cov_g, acc_g, aurc = P.risk_coverage(conf, correct, RC_GRID)
                for cv, ac in zip(cov_g, acc_g):
                    rc.append({"period": period, "dist": dist, "confidence": cname, "coverage": cv,
                               "sel_acc": ac, "aurc": aurc})

        # --- significance: each Kronos dist vs each Gaussian-baseline dist (CRPS, DM/HAC) ---
        for (b1, d1), l1 in crps_by.items():
            if b1 != "kronos":
                continue
            for (b2, d2), l2 in crps_by.items():
                if b2 == "gauss" and d2 in ("raw", "emos_gap"):
                    t = M.dm_test(l1, l2, h=delay)
                    lo_ci, hi_ci = M.block_bootstrap_ci(l1 - l2, block=max(5, delay))
                    tests.append({"period": period, "kronos": d1, "vs_gauss": d2, **t,
                                  "ci_lo": lo_ci, "ci_hi": hi_ci})

    return {"bias": pd.DataFrame(bias), "crps_tests": pd.DataFrame(tests), "calibration": pd.DataFrame(calib),
            "conditional_coverage": pd.DataFrame(cond), "policy": pd.DataFrame(pols),
            "risk_coverage": pd.DataFrame(rc),
            "decisions": pd.concat(decisions, ignore_index=True) if decisions else pd.DataFrame()}
