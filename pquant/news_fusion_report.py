"""News fusion report: does blending the LLM's reading of the news improve the forecast?

For every combination of
- period (before / after the cutoff),
- starting forecast (kronos_raw, kronos_emos_gap, gauss_raw), and
- news variant (real / anon / shuffled headlines),
the LLM's view is blended in with a weight lambda chosen online, and the result is compared
with the starting forecast on the same forecasts: CRPS (with a DM test), PIT-KS, 80% coverage,
IC and direction. It also scores the LLM on its own, and its signal E_Q[r].
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from . import metrics as M
from .calibration import emos_affine
from .fusion import debias_q, llm_only_samples, online_lambda
from .calibration_report import calibrate


def _wmean(X, W):
    return X.mean(1) if W is None else (X * W).sum(1) / W.sum(1)


def _summary(X, W, y):
    lo_hi = M.weighted_quantile(X, [0.1, 0.9], W)
    mu = _wmean(X, W)
    return {"crps": M.crps_ensemble(X, y, W), "pit_ks": M.pit_ks(M.pit(X, y, W)),
            "cov80": float(((y >= lo_hi[:, 0]) & (y <= lo_hi[:, 1])).mean()),
            "ic": M.ic(mu, y), "dir_acc": M.directional_accuracy(mu, y)}


def _starting_forecasts(cfg, kronos, gauss, y, delay, gap):
    """The forecasts the news is blended into, as {name: (samples, weights)}."""
    wanted = cfg["fusion"]["priors"]
    priors = {}
    if "kronos_raw" in wanted:
        priors["kronos_raw"] = (kronos, None)
    if "kronos_emos_gap" in wanted:
        dists, _ = calibrate(kronos, y, cfg, delay, gap)
        priors["kronos_emos_gap"] = dists.get("emos_gap", dists.get("emos"))
    if "gauss_raw" in wanted:
        priors["gauss_raw"] = (gauss, None)
    return priors


def _llm_only_row(variant, period, q_mode, answers, llm_mean, llm_samples, g, y, scored, delay):
    """How good is the LLM on its own: its signal E_Q[r] and its forecast distribution."""
    summary = _summary(llm_samples[scored], None, y[scored])
    test = ic_test(llm_mean[scored], y[scored], delay)
    return {"variant": variant, "q_mode": q_mode, "period": period, "n": int(scored.sum()),
            "mean_heads": g["n_heads"].mean(), "letter_mass": g["letter_mass"].mean(),
            "entropy": float((-(answers * np.log(answers + 1e-12)).sum(1)).mean()),
            "ic_EQ": M.ic(llm_mean[scored], y[scored]), "ic_t": test["stat"], "ic_p": test["p"],
            "rank_ic_EQ": M.rank_ic(llm_mean[scored], y[scored]),
            "llm_only_crps": summary["crps"].mean(), "llm_only_pit_ks": summary["pit_ks"],
            "llm_only_cov80": summary["cov80"]}


def _compare(label, before, after, y, scored, delay, extra):
    """One fusion row: the forecast before and after adding news, with a DM test on the CRPS change."""
    def summary(samples, weights):
        return _summary(samples[scored], None if weights is None else weights[scored], y[scored])

    a, b = summary(*before), summary(*after)
    test = M.dm_test(b["crps"], a["crps"], h=delay)
    ci_lo, ci_hi = M.block_bootstrap_ci(b["crps"] - a["crps"], block=max(5, delay))
    return {**label, "n": int(scored.sum()), "crps_prior": a["crps"].mean(), "crps_fused": b["crps"].mean(),
            "dm_stat": test["stat"], "dm_p": test["p"], "ci_lo": ci_lo, "ci_hi": ci_hi, **extra,
            "pit_ks_prior": a["pit_ks"], "pit_ks_fused": b["pit_ks"],
            "cov80_prior": a["cov80"], "cov80_fused": b["cov80"],
            "ic_prior": a["ic"], "ic_fused": b["ic"],
            "dir_prior": a["dir_acc"], "dir_fused": b["dir_acc"]}


def news_fusion_report(ev: pd.DataFrame, llm: pd.DataFrame, cfg: dict, delay: int) -> dict[str, pd.DataFrame]:
    calib_cfg, fusion_cfg = cfg["calib"], cfg["fusion"]
    first_scored = calib_cfg["warmup"] + delay - 1
    emos_kw = dict(delay=delay, window=calib_cfg["window"], warmup=calib_cfg["warmup"])
    fused_rows, llm_rows = [], []
    for variant, scores in llm.groupby("variant"):
        joined = ev.merge(scores[["key", "q", "edges", "sigma_h", "letter_mass", "n_heads"]], on="key", how="inner")
        for period, g in joined.groupby("period", sort=False):
            g = g.sort_values("origin_ts").reset_index(drop=True)
            y = g["r_true"].to_numpy(float)
            scored = np.arange(len(y)) >= first_scored
            if scored.sum() < 20:
                continue
            answers = np.stack(g["q"].to_numpy()).astype(float)
            edges = np.stack(g["edges"].to_numpy()).astype(float)
            sigma = g["sigma_h"].to_numpy(float)
            gap = g["gap"].to_numpy(float) if "gap" in g else None
            kronos = np.stack(g["kronos"].to_numpy()).astype(float)
            gauss = np.stack(g["gauss"].to_numpy()).astype(float)
            priors = _starting_forecasts(cfg, kronos, gauss, y, delay, gap)

            # the LLM's answers as given, and with its average lean removed
            answer_modes = {"raw": answers, "debiased": debias_q(answers, window=calib_cfg["window"])}
            llm_mean = {}
            for q_mode, q in answer_modes.items():
                llm_samples = llm_only_samples(q, edges, sigma, kronos.shape[1])
                llm_mean[q_mode] = llm_samples.mean(1)
                llm_rows.append(_llm_only_row(variant, period, q_mode, q, llm_mean[q_mode], llm_samples,
                                              g, y, scored, delay))

            # (1) log-linear pooling with a signed, online-chosen lambda
            for (prior, (samples, weights)), (q_mode, q) in itertools.product(priors.items(), answer_modes.items()):
                fused_weights, lam, _ = online_lambda(samples, y, q, edges, delay, grid=fusion_cfg["grid"],
                                                      window=calib_cfg["window"], warmup=calib_cfg["warmup"],
                                                      W=weights)
                label = {"variant": variant, "q_mode": q_mode, "period": period, "method": "pool", "prior": prior}
                extra = {"lam_mean": lam[scored].mean(), "lam_pos_frac": float((lam[scored] > 0).mean()),
                         "lam_neg_frac": float((lam[scored] < 0).mean())}
                fused_rows.append(_compare(label, (samples, weights), (samples, fused_weights),
                                           y, scored, delay, extra))

            # (2) EMOS with the LLM's expected return as an extra input vs the same EMOS without it
            base_inputs = np.zeros((len(y), 0)) if gap is None else gap[:, None]
            for (base, samples), (q_mode, mean) in itertools.product((("kronos", kronos), ("gauss", gauss)),
                                                                     llm_mean.items()):
                without, _ = emos_affine(samples, y, Z=base_inputs if base_inputs.shape[1] else None, **emos_kw)
                with_news, _ = emos_affine(samples, y, Z=np.column_stack([base_inputs, mean]), **emos_kw)
                label = {"variant": variant, "q_mode": q_mode, "period": period, "method": "emos_cov",
                         "prior": f"{base}_emos"}
                fused_rows.append(_compare(label, (without, None), (with_news, None), y, scored, delay, {}))
    return {"fusion": pd.DataFrame(fused_rows), "llm": pd.DataFrame(llm_rows)}


def ic_test(x, y, h: int = 1) -> dict:
    """HAC t-test of corr(x, y) = 0 via the mean of the standardized cross-product."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return {"stat": float("nan"), "p": float("nan")}
    d = (x - x.mean()) / x.std() * (y - y.mean()) / y.std()
    t = M.dm_test(d, np.zeros_like(d), h=h)
    return {"stat": t["stat"], "p": t["p"]}
