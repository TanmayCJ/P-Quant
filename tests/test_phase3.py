import copy
import tempfile

import numpy as np
import pandas as pd

from kronos_agent import fusion as F
from kronos_agent import metrics as M
from kronos_agent.config import DEFAULTS
from kronos_agent.llm import build_messages
from kronos_agent.news import info_time, select_headlines


def news_frame():
    t = pd.Timestamp("2025-03-10 20:00")
    rows = [
        (1, t - pd.Timedelta(hours=30), t - pd.Timedelta(hours=30), "Old but in window"),
        (2, t - pd.Timedelta(hours=60), t - pd.Timedelta(hours=60), "Too old"),
        (3, t - pd.Timedelta(hours=1), t - pd.Timedelta(hours=1), "Fresh news"),
        (4, t - pd.Timedelta(hours=2), t + pd.Timedelta(hours=3), "Edited after cutoff"),
        (5, t, t, "Exactly at cutoff"),
        (6, t + pd.Timedelta(minutes=5), t + pd.Timedelta(minutes=5), "Future"),
        (7, t - pd.Timedelta(hours=5), t - pd.Timedelta(hours=5), "Fresh news"),  # duplicate headline
    ]
    df = pd.DataFrame(rows, columns=["id", "created_at", "updated_at", "headline"])
    return df.sort_values("created_at").reset_index(drop=True), t


def test_select_headlines_is_leakage_safe():
    news, cutoff = news_frame()
    h = select_headlines(news, cutoff, lookback_hours=48, max_items=10)
    assert list(h["headline"]) == ["Fresh news", "Old but in window"]  # newest first, deduped
    assert (h["created_at"] < cutoff).all() and (h["updated_at"] <= cutoff).all()


def test_info_time():
    idx = pd.DatetimeIndex(pd.date_range("2025-03-03", periods=10, freq="B"))
    assert info_time(idx, 5, "1d") == idx[4] + pd.Timedelta(hours=20)
    hidx = pd.date_range("2025-03-03", periods=10, freq="h")
    assert info_time(hidx, 5, "1h") == hidx[5]


def test_anonymized_prompt_hides_asset_and_date():
    news, cutoff = news_frame()
    h = select_headlines(news, cutoff, 48, 10).assign(headline=["Bitcoin surges as BTC ETF flows jump", "x"])
    e = F.bin_edges(np.array([0.02]))[0]
    msgs = build_messages("Bitcoin (BTC/USD)", "24 hours", cutoff, h, e, anonymize=True,
                          aliases=["Bitcoin", "BTC"])
    txt = msgs[1]["content"]
    assert "bitcoin" not in txt.lower() and "btc" not in txt.lower() and "2025" not in txt
    assert "A: below" in txt and "G: above" in txt
    real = build_messages("Bitcoin (BTC/USD)", "24 hours", cutoff, h, e)[1]["content"]
    assert "Bitcoin" in real and "2025-03-10" in real


def _informative(T=1200, n=32, seed=0, informative=True):
    rng = np.random.default_rng(seed)
    sig = np.full(T, 0.02)
    y = rng.normal(0, 0.02, T)
    X = rng.normal(0, 0.02, (T, n))          # calibrated but uninformative prior
    E = F.bin_edges(sig)
    K = E.shape[1] + 1
    if informative:                          # LLM puts extra mass on the true bin
        true_bin = np.array([np.searchsorted(E[t], y[t], side="right") for t in range(T)])
        Q = np.full((T, K), 0.5 / (K - 1))
        Q[np.arange(T), true_bin] = 0.5
    else:
        Q = rng.dirichlet(np.ones(K) * 5, T)
    return X, y, Q, E


def test_lambda_zero_is_prior_and_weights_normalized():
    X, y, Q, E = _informative(T=50)
    w = F.reweight(X, Q, E, 0.0)
    assert np.allclose(w, 1 / X.shape[1])
    w = F.reweight(X, Q, E, 1.0)
    assert np.allclose(w.sum(1), 1) and (w >= 0).all()


def test_online_lambda_uses_informative_news_and_ignores_noise():
    X, y, Q, E = _informative(informative=True)
    W, lam, _ = F.online_lambda(X, y, Q, E, delay=1)
    burn = slice(300, None)
    assert lam[burn].mean() > 0.5
    assert M.crps_ensemble(X[burn], y[burn], W[burn]).mean() < 0.9 * M.crps_ensemble(X[burn], y[burn]).mean()
    X, y, Q, E = _informative(informative=False, seed=1)
    W, lam, _ = F.online_lambda(X, y, Q, E, delay=1)
    assert lam[burn].mean() < 0.5


def test_online_lambda_causal():
    X, y, Q, E = _informative(T=300)
    y2 = y.copy()
    y2[200:] *= -3
    d = 3
    a = F.online_lambda(X, y, Q, E, delay=d, warmup=50)[1]
    b = F.online_lambda(X, y2, Q, E, delay=d, warmup=50)[1]
    assert np.array_equal(a[:200 + d], b[:200 + d])


def test_llm_only_samples_match_bins():
    E = F.bin_edges(np.array([0.02]))
    Q = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]) + 1e-9
    s = F.llm_only_samples(Q / Q.sum(), E, np.array([0.02]), 100)
    assert (s >= E[0, 2] - 1e-6).all() and (s <= E[0, 3] + 1e-6).all()


class FakeScorer:
    def render(self, messages):
        return messages[1]["content"]

    def n_tokens(self, prompt):
        return len(prompt.split())

    def score(self, prompts, letters="ABCDEFG"):
        rng = np.random.default_rng(len(prompts[0]))
        p = rng.dirichlet(np.ones(len(letters)) * 5, len(prompts))
        return p, np.full(len(prompts), 0.9)


def test_phase3_end_to_end_with_fake_llm():
    from kronos_agent.evaluate import build_eval_frame
    from kronos_agent.newsrun import build_inputs, run_scoring
    from kronos_agent.phase3 import phase3_report
    from kronos_agent.pipeline import run_sampling
    from kronos_agent.sampler import KronosSampler
    from kronos_agent.store import ResultStore
    from tests.test_pipeline import CFG, FakePredictor, synth_data

    df = synth_data(1500)
    cfg = copy.deepcopy({**DEFAULTS, **CFG})
    cfg.update(timeframe="1h", stride=1, pred_len=6, bars_per_year=8760,
               windows={"post_cutoff": ["2024-01-04", "2024-03-01"]})
    cfg["calib"] = {**cfg["calib"], "warmup": 30}
    cfg["news"] = {**cfg["news"], "symbols": ["SYN"], "asset_name": "Synth", "horizon_text": "6 hours"}
    ts = pd.date_range("2024-01-01", "2024-03-01", freq="3h")
    news = pd.DataFrame({"id": range(len(ts)), "created_at": ts, "updated_at": ts,
                         "headline": [f"headline {i}" for i in range(len(ts))]})
    with tempfile.TemporaryDirectory() as tmp:
        ps = ResultStore(f"{tmp}/paths")
        run_sampling(cfg, df, KronosSampler(FakePredictor(), "fake"), ps, limit=150, log=lambda *_: None)
        sdf = ps.load()
        parts = []
        for v in ("real", "shuffled"):
            inp = build_inputs(cfg, df, sdf, news, v, seed=1)
            assert all(x["cutoff"] <= pd.Timestamp(x["origin_ts"]) for x in inp)
            st = ResultStore(f"{tmp}/llm_{v}")
            assert run_scoring(inp, FakeScorer(), st, v, batch=4, log=lambda *_: None) == 150
            assert run_scoring(inp, FakeScorer(), st, v, batch=4, log=lambda *_: None) == 0
            parts.append(st.load())
        ev = build_eval_frame(sdf, df, cfg["lookback"])
    rep = phase3_report(ev, pd.concat(parts, ignore_index=True), cfg, delay=6)
    f = rep["fusion"]
    assert set(f["variant"]) == {"real", "shuffled"}
    assert set(f.loc[f["method"] == "pool", "prior"]) == {"kronos_raw", "kronos_emos_gap", "gauss_raw"}
    assert f[["crps_prior", "crps_fused", "dm_p"]].notna().all().all()
    assert len(rep["llm"]) == 4 and set(f["q_mode"]) == {"raw", "debiased"}
    assert set(f["method"]) == {"pool", "emos_cov"}
    assert {"kronos_emos", "gauss_emos"} <= set(f["prior"]) and "ic_p" in rep["llm"]


def test_contrarian_llm_gets_negative_lambda_and_emos_learns_sign():
    from kronos_agent.conformal import emos_affine
    from kronos_agent.phase3 import ic_test

    rng = np.random.default_rng(11)
    T, n = 1200, 32
    sig = np.full(T, 0.02)
    E = F.bin_edges(sig)
    s = rng.normal(0, 0.012, T)
    y = -s + rng.normal(0, 0.016, T)          # LLM view s predicts the OPPOSITE move
    K = E.shape[1] + 1
    Q = np.stack([np.exp(-0.5 * ((np.arange(K) - 3 - s[t] / 0.008)) ** 2) for t in range(T)])
    Q /= Q.sum(1, keepdims=True)
    X = rng.normal(0, 0.02, (T, n))
    burn = slice(300, None)
    W, lam, _ = F.online_lambda(X, y, Q, E, delay=1, grid=DEFAULTS["fusion"]["grid"])
    assert lam[burn].mean() < -0.1  # in-sample CRPS optimum on this design is -0.25
    assert M.crps_ensemble(X[burn], y[burn], W[burn]).mean() < M.crps_ensemble(X[burn], y[burn]).mean()
    emu = F.llm_only_samples(Q, E, sig, n).mean(1)
    A, _ = emos_affine(X, y, delay=1)
    B, _ = emos_affine(X, y, delay=1, Z=emu[:, None])
    assert M.crps_ensemble(B[burn], y[burn]).mean() < M.crps_ensemble(A[burn], y[burn]).mean()
    t = ic_test(emu[burn], y[burn], h=1)
    assert t["stat"] < -3 and t["p"] < 0.01


def test_debias_q_removes_standing_bias_causally():
    rng = np.random.default_rng(3)
    T, K = 600, 7
    bias = np.array([0.05, 0.35, 0.3, 0.2, 0.05, 0.03, 0.02])   # pessimistic habit
    sig = rng.dirichlet(np.ones(K) * 20, T)                      # small news-driven changes
    Q = bias * sig
    Q /= Q.sum(1, keepdims=True)
    D = F.debias_q(Q)
    # after debiasing, the average answer is ~uniform and tracks the news component
    assert np.abs(D[100:].mean(0) - 1 / K).max() < 0.02
    assert np.corrcoef(D[100:, 1], sig[100:, 1])[0, 1] > 0.9
    Q2 = Q.copy()
    Q2[300:] = np.roll(Q2[300:], 1, axis=1)
    a, b = F.debias_q(Q), F.debias_q(Q2)
    assert np.array_equal(a[:300], b[:300]) and not np.array_equal(a[301], b[301])
