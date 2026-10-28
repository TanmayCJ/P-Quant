import copy

import numpy as np

from kronos_agent import conformal as C
from kronos_agent import metrics as M
from kronos_agent import policy as P
from kronos_agent.config import DEFAULTS
from kronos_agent.evaluate import build_eval_frame
from kronos_agent.phase2 import phase2_report
from kronos_agent.pipeline import run_sampling
from kronos_agent.sampler import KronosSampler
from kronos_agent.store import ResultStore


def overconfident(T=1500, n=100, seed=0):
    """Ensemble with half the true spread -> raw intervals undercover."""
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.5, (T, n)), rng.normal(0, 1.0, T)


def test_conformal_methods_fix_undercoverage():
    X, y = overconfident()
    burn = slice(300, None)
    lo, hi = C.raw_interval(X, 0.8)
    raw_cov = ((y >= lo) & (y <= hi))[burn].mean()
    assert raw_cov < 0.6
    for lo, hi in (C.cqr_interval(X, y, 0.8, delay=3), C.aci_interval(X, y, 0.8, delay=3)[:2]):
        assert abs(((y >= lo) & (y <= hi))[burn].mean() - 0.8) < 0.04
    for Xr, Wr in (C.pit_recalibrate(X, y, delay=3), C.spread_rescale(X, y, delay=3)):
        lo, hi = C.raw_interval(Xr, 0.8, Wr)
        assert abs(((y >= lo) & (y <= hi))[burn].mean() - 0.8) < 0.05
        assert M.crps_ensemble(Xr[burn], y[burn], None if Wr is None else Wr[burn]).mean() < \
            M.crps_ensemble(X[burn], y[burn]).mean()


def test_emos_removes_bias_and_restores_coverage():
    rng = np.random.default_rng(6)
    T, n = 1500, 32
    s = rng.normal(0, 0.01, T)                      # true predictable component
    y = s + rng.normal(0, 0.01, T)
    mu = -0.05 + 0.5 * s                            # biased, shrunk mean (like Kronos reversion)
    X = mu[:, None] + rng.normal(0, 0.003, (T, n))  # overconfident spread
    burn = slice(300, None)
    Xe, We = C.emos_affine(X, y, delay=2)
    assert abs((X.mean(1) - y)[burn].mean()) > 0.04
    assert abs((Xe.mean(1) - y)[burn].mean()) < 0.003
    lo, hi = C.raw_interval(Xe, 0.8)
    assert abs(((y >= lo) & (y <= hi))[burn].mean() - 0.8) < 0.05
    assert M.crps_ensemble(Xe[burn], y[burn]).mean() < 0.2 * M.crps_ensemble(X[burn], y[burn]).mean()


def test_emos_gap_removes_reversion_keeps_signal():
    rng = np.random.default_rng(7)
    T, n = 1500, 32
    gap = np.cumsum(rng.normal(0, 0.01, T)) * 0.3     # slowly varying, known ex ante
    s = rng.normal(0, 0.01, T)
    y = s + rng.normal(0, 0.01, T)                    # truth ignores gap
    mu = -0.8 * gap + 0.5 * s                         # Kronos-like reversion + real signal
    X = mu[:, None] + rng.normal(0, 0.01, (T, n))
    burn = slice(300, None)
    plain, _ = C.emos_affine(X, y, delay=2)
    withz, _ = C.emos_affine(X, y, delay=2, Z=gap)
    ic = lambda A: np.corrcoef(A.mean(1)[burn], y[burn])[0, 1]
    assert ic(withz) > ic(plain) + 0.1
    assert abs(np.polyfit(gap[burn], withz.mean(1)[burn], 1)[0]) < 0.15
    assert M.crps_ensemble(withz[burn], y[burn]).mean() < M.crps_ensemble(plain[burn], y[burn]).mean()


def test_conformal_start_ignores_early_outcomes():
    X, y = overconfident(T=400)
    y2 = y.copy()
    y2[:100] += 50.0  # garbage before start
    a = C.cqr_interval(X, y, 0.8, delay=2, warmup=20, start=100)
    b = C.cqr_interval(X, y2, 0.8, delay=2, warmup=20, start=100)
    assert all(np.array_equal(u, v) for u, v in zip(a, b))
    a = C.aci_interval(X, y, 0.8, delay=2, start=100)[:2]
    b = C.aci_interval(X, y2, 0.8, delay=2, start=100)[:2]
    assert all(np.array_equal(u, v) for u, v in zip(a, b))


def test_calibrators_are_causal():
    X, y = overconfident(T=300)
    d, t0 = 4, 150
    y2 = y.copy()
    y2[t0:] += 5.0  # change outcomes from t0 on
    safe = slice(0, t0 + d)  # outputs before t0+d must not move
    pairs = [
        (C.cqr_interval(X, y, 0.8, delay=d, warmup=20), C.cqr_interval(X, y2, 0.8, delay=d, warmup=20)),
        (C.aci_interval(X, y, 0.8, delay=d), C.aci_interval(X, y2, 0.8, delay=d)),
        (C.pit_recalibrate(X, y, delay=d, warmup=20)[:1], C.pit_recalibrate(X, y2, delay=d, warmup=20)[:1]),
        (C.spread_rescale(X, y, delay=d, warmup=20)[:1], C.spread_rescale(X, y2, delay=d, warmup=20)[:1]),
        (C.emos_affine(X, y, delay=d, warmup=20)[:1], C.emos_affine(X, y2, delay=d, warmup=20)[:1]),
    ]
    for a, b in pairs:
        for u, v in zip(a, b):
            assert np.array_equal(u[safe], v[safe])
        assert any(not np.array_equal(u, v) for u, v in zip(a, b))


def test_policy_rules_and_backtest():
    mu = np.array([0.01, -0.01, 0.0001, 0.02])
    lo = np.array([0.005, -0.02, -0.01, -0.01])
    hi = np.array([0.02, -0.005, 0.01, 0.05])
    pos, reason = P.positions("edge", mu, lo, hi, cost_side=0.001)
    assert list(pos) == [1, -1, 0, 1] and reason[2] == "no_edge"
    pos, _ = P.positions("interval", mu, lo, hi, cost_side=0.001)
    assert list(pos) == [1, -1, 0, 0]
    bt = P.backtest(np.ones(10), np.full(10, 0.01), cost_side=0.001, step=1, periods_per_year=365)
    assert np.isclose(bt["total_logret"], 0.1 - 0.001) and bt["trade_frac"] == 1.0


def test_risk_coverage_perfect_confidence():
    correct = np.array([1, 1, 1, 0, 0], float)
    cov, acc, aurc = P.risk_coverage(np.array([5, 4, 3, 2, 1.0]), correct, [0.2, 0.6, 1.0])
    assert list(acc) == [1.0, 1.0, 0.6]
    _, _, aurc_bad = P.risk_coverage(np.array([1, 2, 3, 4, 5.0]), correct)
    assert aurc < aurc_bad


def test_phase2_report_end_to_end():
    from tests.test_pipeline import CFG, FakePredictor, synth_data
    import tempfile

    df = synth_data(1500)
    cfg = copy.deepcopy({**DEFAULTS, **CFG})
    cfg["windows"] = {"post_cutoff": ["2024-01-04", "2024-03-01"]}
    cfg.update(bars_per_year=8760, stride=1, pred_len=6)
    cfg["calib"] = {**cfg["calib"], "warmup": 30}
    with tempfile.TemporaryDirectory() as tmp:
        store = ResultStore(tmp)
        run_sampling(cfg, df, KronosSampler(FakePredictor(), "fake"), store, limit=200, log=lambda *_: None)
        ev = build_eval_frame(store.load(), df, cfg["lookback"])
    rep = phase2_report(ev, cfg, delay=6)
    c = rep["calibration"]
    assert {"raw", "recal", "spread", "emos", "emos_gap", "cqr", "aci", "emos+cqr", "emos+aci",
            "emos_gap+cqr", "emos_gap+aci"} <= set(c["method"])
    assert set(rep["policy"]["interval"]) == {"emos_gap+aci"}
    t = rep["crps_tests"]
    assert set(t["vs_gauss"]) == {"raw", "emos_gap"} and {"stat", "p"} <= set(t.columns)
    b = rep["bias"]
    assert {"bias", "slope_mu_gap"} <= set(b.columns) and set(b["dist"]) >= {"raw", "emos"}
    assert set(rep["policy"]["rule"]) == set(P.RULES)
    assert set(rep["risk_coverage"]["confidence"]) == {"abs_mu", "neg_width", "mu_over_width", "random"}
    assert len(rep["decisions"]) == 200 and rep["decisions"]["scored"].sum() == 200 - 35

    cfg["features"] = {"conformal": False, "abstain": False}
    rep2 = phase2_report(ev, cfg, delay=6)
    assert set(rep2["calibration"]["method"]) == {"raw"}
    assert set(rep2["policy"]["rule"]) == {"always", "buy_hold"}
