import copy

import numpy as np
import pandas as pd

from pquant import metrics as M
from pquant.vol_artifact import (BIAS, decompose, drop_test, event_dates, partial_ic, past_return,
                                 pooled_series, shock_dates, shuffle_answers, simulate, tilt)
from pquant.config import DEFAULTS
from pquant.fusion import bin_edges, llm_only_samples
from pquant.news_scoring import build_inputs, llm_dir, origins_frame


def test_constant_answers_give_exactly_the_volatility_ic():
    rng = np.random.default_rng(0)
    T = 300
    sigma = 0.02 * np.exp(rng.normal(0, 0.3, T))
    r = 0.3 * (sigma - sigma.mean()) + sigma * rng.normal(size=T)
    q = np.array([0.06, 0.32, 0.30, 0.22, 0.05, 0.03, 0.02])  # bearish habit, m(q) < 0
    eq = llm_only_samples(np.tile(q, (T, 1)), bin_edges(sigma), sigma, 64).mean(1)
    assert np.isclose(M.ic(eq, r), -M.ic(sigma, r), atol=1e-9)
    assert abs(partial_ic(eq + 1e-12 * rng.normal(size=T), r, sigma)) < 0.2


def test_simulation_reproduces_artifact_and_controls_remove_it():
    _, pooled = decompose(simulate(info=0.0, seed=3))
    p = pooled.set_index(["mode", "variant", "signal"])["p_one_sided_neg"]
    assert p[("volbins", "real", "raw")] < 0.05 and p[("volbins", "shuffled", "raw")] < 0.05
    assert p[("volbins", "real", "partial_sigma")] > 0.05
    assert p[("labels", "real", "raw")] > 0.05 and p[("fixbins", "real", "raw")] > 0.05
    _, pooled = decompose(simulate(info=0.4, seed=4))
    t = pooled.set_index(["mode", "variant", "signal"])["t"]
    assert t[("labels", "real", "raw")] > 3 and t[("fixbins", "real", "raw")] > 3


def test_answer_shuffle_keeps_the_scale_and_permutes_only_the_answers():
    T = 50
    rng = np.random.default_rng(1)
    sigma = 0.02 * np.exp(rng.normal(0, 0.3, T))
    llm = pd.DataFrame({"origin_ts": pd.date_range("2025-01-01", periods=T, freq="D"),
                        "q": list(rng.dirichlet(np.ones(7), T)), "sigma_h": sigma,
                        "edges": list(bin_edges(sigma))})
    sh = shuffle_answers(llm, seed=0)
    assert (sh["sigma_h"] == llm["sigma_h"]).all() and (sh["origin_ts"] == llm["origin_ts"]).all()
    assert all(np.array_equal(a, b) for a, b in zip(sh["edges"], llm["edges"]))
    q0, q1 = np.stack(llm["q"]), np.stack(sh["q"])
    assert not np.array_equal(q0, q1)
    assert np.allclose(np.sort(q0, axis=0), np.sort(q1, axis=0))  # same answers, new dates


def test_bias_tilt_runs_from_uniform_to_stronger():
    assert np.allclose(tilt(BIAS, 0.0), np.full(7, 1 / 7))
    assert np.allclose(tilt(BIAS, 1.0), BIAS)
    assert tilt(BIAS, 1.5)[1] > BIAS[1]


def test_other_llms_get_their_own_score_folder():
    cfg = copy.deepcopy(DEFAULTS)
    assert llm_dir("T", cfg, "real") == "T_real"
    cfg["llm"] = {**cfg["llm"], "model_id": "Qwen/Qwen2.5-7B-Instruct", "mode": "labels"}
    assert llm_dir("T", cfg, "real") == "T_labels_real_Qwen2.5-7B-Instruct"


def test_modes_build_the_right_prompts():
    from tests.test_pipeline import synth_data

    df = synth_data(600)
    cfg = copy.deepcopy(DEFAULTS)
    cfg.update(asset="SYN", timeframe="1h", lookback=64, pred_len=6, stride=6, seed=1,
               windows={"confirm": ["2024-01-04", "2024-01-20"]})
    cfg["news"] = {**cfg["news"], "symbols": ["SYN"], "asset_name": "Synth", "horizon_text": "6 hours"}
    ts = pd.date_range("2024-01-01", "2024-01-25", freq="2h")
    news = pd.DataFrame({"id": range(len(ts)), "created_at": ts, "updated_at": ts,
                         "headline": [f"h{i}" for i in range(len(ts))]})
    o = origins_frame(cfg, df)
    vol = build_inputs(cfg, df, o, news)
    assert len({tuple(x["edges"]) for x in vol}) > 1 and vol[0]["letters"] == "ABCDEFG"
    cfg["llm"] = {**cfg["llm"], "mode": "fixbins"}
    fix = build_inputs(cfg, df, o, news)
    assert len({tuple(x["edges"]) for x in fix}) == 1                     # no time-varying scale
    assert [x["sigma_h"] for x in fix] == [x["sigma_h"] for x in vol]    # true vol still recorded
    assert llm_dir("T", cfg, "real") == "T_fixbins_real"
    cfg["llm"] = {**cfg["llm"], "mode": "labels"}
    lab = build_inputs(cfg, df, o, news)
    assert lab[0]["letters"] == "ABC" and "A: up" in lab[0]["messages"][1]["content"]
    cfg["llm"] = {**cfg["llm"], "mode": "volbins"}
    assert llm_dir("T", cfg, "real") == "T_real"                          # old runs stay valid


def test_past_return_ends_before_forecast_window():
    idx = pd.date_range("2024-01-01", periods=30, freq="D")
    df = pd.DataFrame({"close": np.exp(0.01 * np.arange(30))}, index=idx)
    pr = past_return(df, idx[[3, 10, 20]], h=5)
    assert np.isnan(pr[0])                                   # not enough history
    assert np.allclose(pr[1:], 0.05)                         # close[i-1] / close[i-6]


def test_decompose_partials_out_extra_control():
    rng = np.random.default_rng(7)
    rows = []
    for a in range(6):
        T = 300
        x = rng.normal(size=T)                               # e.g. last week's return
        r = -0.4 * x + rng.normal(size=T)                    # reversal
        eq = x + 0.3 * rng.normal(size=T)                    # signal that only echoes x
        rows.append(pd.DataFrame({"asset": f"S{a}", "mode": "labels", "variant": "real",
                                  "date": pd.date_range("2020-01-05", periods=T, freq="W"),
                                  "eq": eq, "sigma": np.exp(rng.normal(size=T)), "r": r, "x": x}))
    f = pd.concat(rows, ignore_index=True)
    _, default = decompose(f)
    assert set(default["signal"]) == {"raw", "partial_sigma"}
    per, pooled = decompose(f, {"partial_sigma": ["sigma"], "partial_x": ["x"]})
    t = pooled.set_index("signal")["t"]
    assert t["raw"] < -5 and abs(t["partial_x"]) < 3
    assert (per["corr_signal_x"] > 0.8).all() and "ic_partial_x" in per


def test_pooled_series_matches_decompose():
    f = simulate(info=0.0, seed=5, T=120)
    _, pooled = decompose(f)
    ser = pooled_series(f)
    for _, row in pooled.iterrows():
        assert np.isclose(ser[(row["mode"], row["variant"], row["signal"])].mean(), row["mean_d"])


def test_drop_test_finds_concentrated_dates():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2024-07-07", periods=115, freq="W")
    s = pd.Series(rng.normal(0, 0.3, 115), index=idx)
    shock = idx[::12]
    s[shock] -= 2.0                                           # all of the "signal" sits in 10 weeks
    res = drop_test(s, shock, B=300)
    assert res["n_dropped"] == len(shock) and res["t_all"] < -2
    assert abs(res["t_kept"]) < 1.5 and res["p_random"] < 0.01 and res["share_of_sum"] > 0.8


def test_shock_and_event_dates():
    f = simulate(info=0.0, seed=2, T=120)
    d = shock_dates(f, 0.9)
    assert 10 <= len(d) <= 14
    ev = event_dates(pd.date_range("2024-07-07", periods=60, freq="W"))
    assert len(ev["yen_carry_2024-08"]) in (1, 2) and len(ev["tariffs_2025-04"]) in (1, 2)
