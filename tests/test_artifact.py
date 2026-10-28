import copy

import numpy as np
import pandas as pd

from kronos_agent import metrics as M
from kronos_agent.artifact import decompose, partial_ic, simulate
from kronos_agent.config import DEFAULTS
from kronos_agent.fusion import bin_edges, llm_only_samples
from kronos_agent.newsrun import build_inputs, llm_dir, origins_frame


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
