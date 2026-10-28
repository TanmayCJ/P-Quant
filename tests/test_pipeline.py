"""End-to-end pipeline on synthetic data with a fake Kronos predictor (no GPU/download)."""
import tempfile

import numpy as np
import pandas as pd

from kronos_agent.data import COLS, origins_by_window
from kronos_agent.evaluate import build_eval_frame, summarize
from kronos_agent.pipeline import decode_paths, run_sampling
from kronos_agent.sampler import KronosSampler
from kronos_agent.store import ResultStore


class FakePredictor:
    """Mimics KronosPredictor.predict_batch: GBM paths from last close; records leakage info."""

    def __init__(self):
        self.violations = 0

    def predict_batch(self, df_list, x_timestamp_list, y_timestamp_list, pred_len, T, top_p,
                      sample_count, verbose):
        assert sample_count == 1
        out = []
        for df, xt, yt in zip(df_list, x_timestamp_list, y_timestamp_list):
            if xt.iloc[-1] >= yt.iloc[0]:
                self.violations += 1
            c0 = df["close"].iloc[-1]
            close = c0 * np.exp(np.cumsum(np.random.normal(0, 0.01, pred_len)))
            d = pd.DataFrame({"open": close, "high": close * 1.002, "low": close * 0.998,
                              "close": close, "volume": 1.0, "amount": close}, index=pd.DatetimeIndex(yt))
            out.append(d)
        return out


def synth_data(n=900):
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    close = 100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, n)))
    df = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                       "close": close, "volume": 10.0}, index=idx)
    df["amount"] = df["volume"] * df["close"]
    return df[COLS]


CFG = {
    "asset": "SYN", "lookback": 64, "pred_len": 6, "stride": 6, "n_paths": 16, "batch_paths": 8,
    "T": 1.0, "top_p": 0.9, "seed": 7,
    "windows": {"pre_cutoff": ["2024-01-04", "2024-01-20"], "post_cutoff": ["2024-01-20", "2024-02-05"]},
}


def test_resumable_run_and_eval():
    df = synth_data()
    with tempfile.TemporaryDirectory() as tmp:
        store = ResultStore(tmp)
        fake = FakePredictor()
        s = KronosSampler(fake, model_id="fake")
        logs = []
        assert run_sampling(CFG, df, s, store, limit=3, flush_every=2, log=logs.append) == 3
        total = sum(len(v) for v in origins_by_window(df, CFG).values())
        assert run_sampling(CFG, df, s, store, flush_every=5, log=logs.append) == total - 3
        assert run_sampling(CFG, df, s, store, log=logs.append) == 0
        sdf = store.load()
        assert len(sdf) == total and sdf["key"].is_unique
        assert fake.violations == 0
        assert not any(c.startswith(("realized", "r_true", "target")) for c in sdf.columns)
        assert decode_paths(sdf.iloc[0]).shape == (16, 6)

        ev = build_eval_frame(sdf, df, CFG["lookback"])
        table, tests = summarize(ev, dm_h=1)
        assert set(table["period"]) == {"pre_cutoff", "post_cutoff"}
        assert {"kronos", "gauss", "boot", "rw", "vol_kronos"} <= set(table["method"])
        assert table.loc[table.method == "kronos", "crps"].notna().all()
        assert len(tests) > 0


def test_sampling_reproducible():
    df = synth_data(200)
    s = KronosSampler(FakePredictor(), model_id="fake")
    a = s.sample(df.iloc[:64], df.index[64:70], 16, seed=3, batch=8)["close"]
    b = s.sample(df.iloc[:64], df.index[64:70], 16, seed=3, batch=8)["close"]
    c = s.sample(df.iloc[:64], df.index[64:70], 16, seed=4, batch=8)["close"]
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_eval_rejects_changed_data():
    df = synth_data()
    with tempfile.TemporaryDirectory() as tmp:
        store = ResultStore(tmp)
        run_sampling(CFG, df, KronosSampler(FakePredictor(), "fake"), store, limit=2, log=lambda *_: None)
        df2 = df.copy()
        df2["close"] *= 1.01
        try:
            build_eval_frame(store.load(), df2, CFG["lookback"])
        except ValueError:
            return
        raise AssertionError("expected mismatch error")
