import copy
import tempfile

import numpy as np
import pandas as pd

from kronos_agent.config import DEFAULTS
from kronos_agent.data import OHLCV
from kronos_agent.phaseb import (asset_metrics, pooled_dm, run_synthetic, slope_dk, slope_influence,
                                 summarize_synthetic, synthetic_series)


class RevertingSampler:
    """Forecasts a move of -k * gap (toward the window mean), like zero-shot Kronos."""

    def __init__(self, k):
        self.k = k
        self.model_id = f"revert{k}"

    def sample(self, ctx, y_ts, n_paths, seed, **_):
        rng = np.random.default_rng(seed)
        c = ctx["close"].to_numpy(float)
        gap = np.log(c[-1]) - np.log(c).mean()
        end = np.log(c[-1]) - self.k * gap + rng.normal(0, 0.01, (n_paths, 1))
        path = np.exp(np.log(c[-1]) + (end - np.log(c[-1])) * np.linspace(1 / len(y_ts), 1, len(y_ts)))
        return {f: path.astype(np.float32) for f in OHLCV}


def test_synthetic_series_is_a_drifting_random_walk():
    df = synthetic_series(3000, drift=0.002, seed=1)
    lr = np.diff(np.log(df["close"].to_numpy()))
    assert abs(lr.mean() - 0.002) < 0.001 and abs(np.corrcoef(lr[:-1], lr[1:])[0, 1]) < 0.06
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()


def test_synthetic_test_separates_reverting_from_unbiased_forecasters():
    s_rev = summarize_synthetic(run_synthetic(RevertingSampler(0.8), 64, n_series=25, log=lambda *_: None))
    s_rw = summarize_synthetic(run_synthetic(RevertingSampler(0.0), 64, n_series=25, log=lambda *_: None))
    allrow = lambda s: s[s["drift"] == "all"].iloc[0]
    assert allrow(s_rev)["slope_fc_gap"] < -0.5 and abs(allrow(s_rw)["slope_fc_gap"]) < 0.1
    up = s_rev[s_rev["drift"] == 0.002].iloc[0]
    assert up["mean_fc"] < 0 < up["true_mean"] and up["share_against_trend"] > 0.6


def test_asset_metrics_and_pooled_dm_on_pipeline_output():
    from kronos_agent.evaluate import build_eval_frame
    from kronos_agent.pipeline import run_sampling
    from kronos_agent.sampler import KronosSampler
    from kronos_agent.store import ResultStore
    from tests.test_pipeline import CFG, FakePredictor, synth_data

    df = synth_data(1500)
    cfg = copy.deepcopy({**DEFAULTS, **CFG})
    cfg.update(stride=6, pred_len=6, windows={"post_cutoff": ["2024-01-04", "2024-03-01"]})
    cfg["calib"] = {**cfg["calib"], "warmup": 20}
    losses = []
    for seed in (1, 2):
        with tempfile.TemporaryDirectory() as tmp:
            st = ResultStore(tmp)
            run_sampling({**cfg, "seed": seed}, df, KronosSampler(FakePredictor(), "fake"), st, log=lambda *_: None)
            ev = build_eval_frame(st.load(), df, cfg["lookback"], seed=seed)
        row, loss = asset_metrics(ev, cfg, delay=1)
        assert row["n"] > 100 and np.isfinite([row["crps_raw"], row["crps_emos"], row["slope_gap"]]).all()
        assert abs(loss["psi_slope"].sum()) < 1e-8   # OLS influence sums to zero
        losses.append(loss)
    p = pooled_dm(losses, "d_emos")
    assert p["n_dates"] == len(losses[0]) and np.isfinite(p["t"])


def _panel(rng, n_assets, T, common):
    """Per-asset OLS slopes of mu on gap with a date shock shared by all assets (weight `common`)."""
    f, c = rng.normal(size=T), rng.normal(size=T)
    rows, slopes = [], []
    dates = pd.bdate_range("2024-01-01", periods=T)
    for i in range(n_assets):
        gap = common * f + rng.normal(size=T)
        mu = -0.5 * gap + common * f * c + rng.normal(size=T)
        b, a = np.polyfit(gap, mu, 1)
        xd = gap - gap.mean()
        rows.append(pd.DataFrame({"asset": i, "date": dates, "psi_slope": xd * (mu - a - b * gap) / (xd @ xd)}))
        slopes.append(b)
    return pd.concat(rows), np.array(slopes)


def test_driscoll_kraay_slope_se():
    rng = np.random.default_rng(0)
    # independent assets: DK ~ naive cross-asset SE; shared date shocks: naive SE is far too small
    for common, lo, hi in ((0.0, 0.7, 1.4), (2.0, 2.0, np.inf)):
        ratios, bbar, se = [], [], []
        for _ in range(200):
            psi, b = _panel(rng, 29, 115, common)
            s = slope_dk(slope_influence(psi), lags=(None,))["se_dk"]
            ratios.append(s / (b.std(ddof=1) / np.sqrt(len(b))))
            bbar.append(b.mean())
            se.append(s)
        assert lo < np.median(ratios) < hi
        assert 0.8 < np.mean(se) / np.std(bbar) < 1.25   # DK SE matches the true sampling sd
