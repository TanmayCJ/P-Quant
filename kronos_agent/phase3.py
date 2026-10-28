"""Phase 3 report: does news-likelihood reweighting improve calibrated forecasts?

For each period x prior (kronos_raw, kronos_emos_gap, gauss_raw) x LLM variant
(real / anon / shuffled), fuse with online lambda and compare against the prior on
the same scored origins: CRPS (+ DM test), PIT-KS, 80% coverage, IC, direction.
Also reports the LLM-only forecast and the LLM signal itself (IC of E_Q[r]).
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from . import metrics as M
from .conformal import emos_affine
from .fusion import debias_q, llm_only_samples, online_lambda
from .phase2 import calibrate


def _wmean(X, W):
    return X.mean(1) if W is None else (X * W).sum(1) / W.sum(1)


def _summ(X, W, y):
    lo_hi = M.weighted_quantile(X, [0.1, 0.9], W)
    mu = _wmean(X, W)
    return {"crps": M.crps_ensemble(X, y, W), "pit_ks": M.pit_ks(M.pit(X, y, W)),
            "cov80": float(((y >= lo_hi[:, 0]) & (y <= lo_hi[:, 1])).mean()),
            "ic": M.ic(mu, y), "dir_acc": M.directional_accuracy(mu, y)}


def phase3_report(ev: pd.DataFrame, llm: pd.DataFrame, cfg: dict, delay: int) -> dict[str, pd.DataFrame]:
    cal, fus = cfg["calib"], cfg["fusion"]
    start = cal["warmup"] + delay - 1
    fused_rows, llm_rows = [], []
    for variant, lv in llm.groupby("variant"):
        joined = ev.merge(lv[["key", "q", "edges", "sigma_h", "letter_mass", "n_heads"]], on="key", how="inner")
        for period, g in joined.groupby("period", sort=False):
            g = g.sort_values("origin_ts").reset_index(drop=True)
            y = g["r_true"].to_numpy(float)
            m = np.arange(len(y)) >= start
            if m.sum() < 20:
                continue
            Q = np.stack(g["q"].to_numpy()).astype(float)
            E = np.stack(g["edges"].to_numpy()).astype(float)
            sig = g["sigma_h"].to_numpy(float)
            gap = g["gap"].to_numpy(float) if "gap" in g else None

            priors = {}
            Xk = np.stack(g["kronos"].to_numpy()).astype(float)
            Xg = np.stack(g["gauss"].to_numpy()).astype(float)
            if "kronos_raw" in fus["priors"]:
                priors["kronos_raw"] = (Xk, None)
            if "kronos_emos_gap" in fus["priors"]:
                d, _ = calibrate(Xk, y, cfg, delay, gap)
                priors["kronos_emos_gap"] = d.get("emos_gap", d.get("emos"))
            if "gauss_raw" in fus["priors"]:
                priors["gauss_raw"] = (Xg, None)

            n = Xk.shape[1]
            Qs = {"raw": Q, "debiased": debias_q(Q, window=cal["window"])}
            emus = {}
            for qm, Qm in Qs.items():
                Xl = llm_only_samples(Qm, E, sig, n)
                s = _summ(Xl[m], None, y[m])
                emus[qm] = emu = Xl.mean(1)
                it = ic_test(emu[m], y[m], delay)
                llm_rows.append({"variant": variant, "q_mode": qm, "period": period, "n": int(m.sum()),
                                 "mean_heads": g["n_heads"].mean(), "letter_mass": g["letter_mass"].mean(),
                                 "entropy": float((-(Qm * np.log(Qm + 1e-12)).sum(1)).mean()),
                                 "ic_EQ": M.ic(emu[m], y[m]), "ic_t": it["stat"], "ic_p": it["p"],
                                 "rank_ic_EQ": M.rank_ic(emu[m], y[m]),
                                 "llm_only_crps": s["crps"].mean(), "llm_only_pit_ks": s["pit_ks"],
                                 "llm_only_cov80": s["cov80"]})

            def add_row(method, pname, qm, Xa, Wa, Xb, Wb, extra):
                a = _summ(Xa[m], None if Wa is None else Wa[m], y[m])
                b = _summ(Xb[m], None if Wb is None else Wb[m], y[m])
                t = M.dm_test(b["crps"], a["crps"], h=delay)
                lo, hi = M.block_bootstrap_ci(b["crps"] - a["crps"], block=max(5, delay))
                fused_rows.append({
                    "variant": variant, "q_mode": qm, "period": period, "method": method, "prior": pname,
                    "n": int(m.sum()), "crps_prior": a["crps"].mean(), "crps_fused": b["crps"].mean(),
                    "dm_stat": t["stat"], "dm_p": t["p"], "ci_lo": lo, "ci_hi": hi, **extra,
                    "pit_ks_prior": a["pit_ks"], "pit_ks_fused": b["pit_ks"],
                    "cov80_prior": a["cov80"], "cov80_fused": b["cov80"],
                    "ic_prior": a["ic"], "ic_fused": b["ic"],
                    "dir_prior": a["dir_acc"], "dir_fused": b["dir_acc"]})

            # (1) log-linear pooling with a signed, online-chosen lambda
            for (pname, (Xp, Wp)), (qm, Qm) in itertools.product(priors.items(), Qs.items()):
                Wf, lam, _ = online_lambda(Xp, y, Qm, E, delay, grid=fus["grid"], window=cal["window"],
                                           warmup=cal["warmup"], W=Wp)
                add_row("pool", pname, qm, Xp, Wp, Xp, Wf,
                        {"lam_mean": lam[m].mean(), "lam_pos_frac": float((lam[m] > 0).mean()),
                         "lam_neg_frac": float((lam[m] < 0).mean())})

            # (2) EMOS with the LLM's expected return as an extra covariate vs identical EMOS without it
            kw = dict(delay=delay, window=cal["window"], warmup=cal["warmup"])
            Z0 = np.zeros((len(y), 0)) if gap is None else gap[:, None]
            for (bname, Xb), (qm, emu) in itertools.product((("kronos", Xk), ("gauss", Xg)), emus.items()):
                A, _ = emos_affine(Xb, y, Z=Z0 if Z0.shape[1] else None, **kw)
                B, _ = emos_affine(Xb, y, Z=np.column_stack([Z0, emu]), **kw)
                add_row("emos_cov", f"{bname}_emos", qm, A, None, B, None, {})
    return {"fusion": pd.DataFrame(fused_rows), "llm": pd.DataFrame(llm_rows)}


def ic_test(x, y, h: int = 1) -> dict:
    """HAC t-test of corr(x, y) = 0 via the mean of the standardized cross-product."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return {"stat": float("nan"), "p": float("nan")}
    d = (x - x.mean()) / x.std() * (y - y.mean()) / y.std()
    t = M.dm_test(d, np.zeros_like(d), h=h)
    return {"stat": t["stat"], "p": t["p"]}
