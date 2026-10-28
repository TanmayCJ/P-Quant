import copy

import numpy as np
import pandas as pd

from kronos_agent.confirm import asset_frame, family_test, holm, per_asset
from kronos_agent.config import DEFAULTS
from kronos_agent.fusion import bin_edges
from kronos_agent.newsrun import build_inputs, origins_frame


def _frames(rho, n_assets=6, T=120, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=T, freq="W")
    rows = []
    for a in range(n_assets):
        eq = rng.normal(size=T)
        r = rho * eq + np.sqrt(1 - rho ** 2) * rng.normal(size=T)
        rows.append(pd.DataFrame({"asset": f"A{a}", "variant": "real", "origin_ts": dates,
                                  "date": dates, "eq": eq, "r": r}))
    return pd.concat(rows, ignore_index=True)


def test_family_test_detects_negative_and_not_positive_or_null():
    neg = family_test(_frames(-0.2), "real")
    assert neg["p_one_sided"] < 0.01 and neg["n_assets"] == 6
    assert family_test(_frames(0.2, seed=1), "real")["p_one_sided"] > 0.9
    nulls = [family_test(_frames(0.0, seed=s), "real")["p_one_sided"] for s in range(40)]
    assert 0.0 < np.mean(np.array(nulls) < 0.05) < 0.2  # roughly nominal size


def test_perm_test_detects_signal_and_holds_size():
    from kronos_agent.confirm import perm_test

    assert perm_test(_frames(-0.2), "real", B=300)["p_one_sided_neg"] < 0.01
    ps = [perm_test(_frames(0.0, seed=s), "real", B=200, seed=s)["p_one_sided_neg"] for s in range(30)]
    assert 0 <= np.mean(np.array(ps) < 0.05) <= 0.2 and 0.3 < np.mean(ps) < 0.7


def test_pooled_spread_estimator_survives_degenerate_ensembles():
    from kronos_agent import metrics as M
    from kronos_agent.conformal import emos_affine

    rng = np.random.default_rng(9)
    T, n = 400, 32
    X = rng.normal(0, 0.02, (T, n))
    X[::25] = X[::25].mean(1, keepdims=True) + rng.normal(0, 1e-7, (len(X[::25]), n))  # near-zero spread days
    y = rng.normal(0, 0.02, T)
    burn = slice(100, None)
    bad, _ = emos_affine(X, y, delay=1)
    good, _ = emos_affine(X, y, delay=1, k_mode="pooled")
    assert M.crps_ensemble(good[burn], y[burn]).mean() < 0.8 * M.crps_ensemble(bad[burn], y[burn]).mean()


def test_per_asset_and_holm():
    t = per_asset(_frames(-0.3))
    assert (t["ic"] < 0).all() and (t["p_one_sided"] < 0.05).mean() > 0.5
    adj = holm({"a": 0.01, "b": 0.04})
    assert np.isclose(adj["a"], 0.02) and np.isclose(adj["b"], 0.04)
    adj = holm({"a": 0.03, "b": 0.02})
    assert np.isclose(adj["b"], 0.04) and np.isclose(adj["a"], 0.04)


def test_origins_only_inputs_and_asset_frame():
    from tests.test_pipeline import synth_data

    df = synth_data(600)
    cfg = copy.deepcopy(DEFAULTS)
    cfg.update(asset="SYN", timeframe="1h", lookback=64, pred_len=6, stride=6, seed=1,
               windows={"confirm": ["2024-01-04", "2024-01-20"]})
    cfg["news"] = {**cfg["news"], "symbols": ["SYN"], "asset_name": "Synth", "horizon_text": "6 hours"}
    o = origins_frame(cfg, df)
    assert len(o) > 20 and o["key"].is_unique and (o["row"] >= 64).all()
    ts = pd.date_range("2024-01-01", "2024-01-25", freq="2h")
    news = pd.DataFrame({"id": range(len(ts)), "created_at": ts, "updated_at": ts,
                         "headline": [f"h{i}" for i in range(len(ts))]})
    inp = build_inputs(cfg, df, o, news, "real")
    assert len(inp) == len(o)
    llm = pd.DataFrame({"origin_ts": [x["origin_ts"] for x in inp], "variant": "real",
                        "q": [list(np.full(7, 1 / 7))] * len(inp),
                        "edges": [x["edges"] for x in inp], "sigma_h": [x["sigma_h"] for x in inp]})
    f = asset_frame(cfg, df, llm)
    i = df.index.get_indexer([inp[0]["origin_ts"]])[0]
    c = df["close"].to_numpy()
    assert np.isclose(f["r"].iloc[0], np.log(c[i + 5] / c[i - 1]))
    assert np.allclose(f["eq"], f["eq"].iloc[0], atol=1e-3)  # uniform q -> ~same E_Q
    f1 = asset_frame({**cfg, "pred_len": 1}, df, llm)       # same scores, next-bar return
    assert np.isclose(f1["r"].iloc[0], np.log(c[i] / c[i - 1])) and len(f1) == len(f)
    lab = llm.assign(q=[[0.6, 0.3, 0.1]] * len(llm))         # labels: s = P(up) - P(down)
    assert np.allclose(asset_frame(cfg, df, lab)["eq"], 0.3)
