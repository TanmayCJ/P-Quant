import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def test_export_asset_end_to_end():
    from kronos_agent.config import load_config, run_tag
    from kronos_agent.export import export_asset
    from kronos_agent.newsrun import build_inputs, llm_dir, llm_tag, origins_frame, run_scoring
    from kronos_agent.pipeline import run_sampling
    from kronos_agent.sampler import KronosSampler
    from kronos_agent.store import ResultStore
    from tests.test_pipeline import FakePredictor, synth_data
    from tests.test_phase3 import FakeScorer

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        df = synth_data(900)
        (root / "data").mkdir()
        df.to_parquet(root / "data" / "SYN_1h.parquet")
        base = {"asset": "SYN", "source": "yfinance", "symbol": "SYN", "timeframe": "1h", "data_start": "2024-01-01",
                "data_end": "2024-03-01", "pred_len": 6, "stride": 6, "seed": 3,
                "windows": {"confirm": ["2024-01-04", "2024-02-05"]}, "n_paths": 16, "batch_paths": 16,
                "calib": {"warmup": 20}}
        kcfg = root / "k.yaml"
        kcfg.write_text(yaml.safe_dump({**base, "lookback": 256}))
        lcfg = root / "l.yaml"
        lcfg.write_text(yaml.safe_dump({**base, "lookback": 64, "llm": {"mode": "labels"},
                                        "news": {"symbols": ["SYN"], "asset_name": "Synth", "horizon_text": "6 hours",
                                                 "lookback_hours": 48, "max_headlines": 5}}))
        cfg = load_config(kcfg, ["lookback=64"])
        run_sampling(cfg, df, KronosSampler(FakePredictor(), cfg["model_id"]), ResultStore(root / "paths" / run_tag(cfg)),
                     log=lambda *_: None)
        ts = pd.date_range("2023-12-01", "2024-03-31 23:00", freq="3h")
        news = pd.DataFrame({"id": range(len(ts)), "created_at": ts, "updated_at": ts,
                             "headline": [f"headline {i}" for i in range(len(ts))], "summary": "", "source": "x",
                             "symbols": "SYN"})
        nd = root / "news" / "SYN"
        nd.mkdir(parents=True)
        for m, g in news.groupby(news["created_at"].dt.to_period("M")):
            g.to_parquet(nd / f"{m.start_time:%Y-%m}.parquet", index=False)
        lc = load_config(lcfg, ["llm.mode=labels"])
        inp = build_inputs(lc, df, origins_frame(lc, df), news, "real")
        run_scoring(inp, FakeScorer(), ResultStore(root / "llm" / llm_dir(llm_tag(lc), lc, "real")), "real",
                    log=lambda *_: None)

        a = export_asset(str(kcfg), root, str(lcfg))
        assert a["asset"] == "SYN" and set(a["metrics"]) == {"kronos64"}
        o = a["origins"]
        assert len(o) > 50 and all(len(x["fc"]["gauss"]) == 5 for x in o)
        assert sum("emos" in x["fc"]["kronos64"] for x in o) == len(o) - 20
        assert all(np.all(np.diff(x["fc"]["kronos64"]["raw"]) >= 0) for x in o)
        assert len(o[0]["fc"]["kronos64"]["path"]) == 6
        withllm = [x for x in o if "llm" in x]
        assert len(withllm) == len(o) and len(withllm[0]["llm"]["heads"]) == 3
        assert all(h["age_h"] > 0 for x in withllm for h in x["llm"]["heads"])
        json.dumps(a, allow_nan=False)

        # offline news read: an uncached month is skipped, never fetched (no keys in this test)
        from kronos_agent.news import load_news
        (nd / "2024-02.parquet").unlink()
        got = load_news(["SYN"], "2024-01-01", "2024-03-15", root / "news", key="", secret="", offline=True)
        assert len(got) and got["created_at"].dt.month.isin([1, 3]).all()
