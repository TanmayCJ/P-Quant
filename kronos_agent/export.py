"""Pack saved results into one compact JSON for the dashboard (CPU).

Per asset: daily closes; per forecast origin: realized horizon return, gap, volatility,
5-quantile horizon-return distributions (raw Kronos/Chronos, EMOS-corrected Kronos, Gaussian),
the raw models' median path, the LLM's direction-label probabilities and the top headlines
it saw; per-run summary metrics. All returns are in percent.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .conformal import emos_affine
from .config import load_config, run_tag
from .data import label_period, load_data, origins_by_window
from .evaluate import build_eval_frame
from .news import info_time, load_news, select_headlines
from .newsrun import llm_dir, llm_tag
from .phaseb import asset_metrics
from .pipeline import decode_paths
from .store import ResultStore

QS = (0.05, 0.25, 0.5, 0.75, 0.95)
RUNS = {
    "kronos64": ["lookback=64"],
    "kronos256": ["lookback=256"],
    "chronos64": ["lookback=64", "model_family=chronos", "model_id=amazon/chronos-t5-small", "amp=false"],
}


def _q(a) -> list[float]:
    return [round(float(v) * 100, 3) for v in np.quantile(np.asarray(a, float), QS)]


def run_frame(cfg_path: str, overrides: list[str], root: Path):
    cfg = load_config(cfg_path, overrides)
    df = load_data(cfg, root / "data")
    s = ResultStore(root / "paths" / run_tag(cfg)).load()
    if s.empty:
        return cfg, df, None, None
    grid = set(np.concatenate(list(origins_by_window(df, cfg).values())).tolist())
    s = s[s["row"].isin(grid)].copy()
    s["period"] = [label_period(pd.Timestamp(t), cfg["windows"]) for t in s["origin_ts"]]
    s = s.sort_values("origin_ts").reset_index(drop=True)
    ev = build_eval_frame(s, df, cfg["lookback"], seed=cfg["seed"]).sort_values("origin_ts").reset_index(drop=True)
    return cfg, df, s, ev


def export_asset(cfg_path: str, root: Path, llm_cfg_path: str | None = None, runs=RUNS,
                 n_heads: int = 3, price_start: str = "2024-04-01") -> dict | None:
    root = Path(root)
    out, base, origins = None, None, {}
    metrics = {}
    for name, ov in runs.items():
        cfg, df, s, ev = run_frame(cfg_path, ov, root)
        if ev is None or ev.empty:
            continue
        if out is None:
            base = (cfg, df)
            close = df["close"]
            if cfg["timeframe"].endswith("h"):
                close = close.resample("1D").last().dropna()
            close = close[close.index >= pd.Timestamp(price_start)]
            out = {"asset": cfg["asset"], "timeframe": cfg["timeframe"], "horizon_bars": int(cfg["pred_len"]),
                   "horizon": "24 hours" if cfg["timeframe"].endswith("h") else f"{cfg['pred_len']} trading days",
                   "prices": [[d.strftime("%Y-%m-%d"), round(float(v), 4)] for d, v in close.items()]}
        row, _ = asset_metrics(ev, cfg, 1)
        metrics[name] = {k: (round(float(v), 6) if isinstance(v, (float, np.floating)) else v) for k, v in row.items()}
        y = ev["r_true"].to_numpy(float)
        X = np.stack(ev["kronos"].to_numpy()).astype(float)
        cal = cfg["calib"]
        E, _ = emos_affine(X, y, delay=1, window=cal["window"], warmup=cal["warmup"], Z=ev["gap"].to_numpy(float))
        ready = np.arange(len(y)) >= cal["warmup"]
        recs = s.set_index("key")
        for j, r in ev.iterrows():
            t = pd.Timestamp(r["origin_ts"]).strftime("%Y-%m-%d %H:%M")
            o = origins.setdefault(t, {"t": t, "r": round(float(r["r_true"]) * 100, 3),
                                       "gap": round(float(r["gap"]) * 100, 3), "fc": {}})
            if name == "kronos64" or "gauss" not in o["fc"]:
                o["fc"]["gauss"] = _q(r["gauss"])
                o["sigma"] = round(float(r["vol_hist"]) * 100, 4)
            rec = recs.loc[r["key"]]
            path = decode_paths(rec).astype(float)
            med = np.median(np.log(path / rec["last_close"]), axis=0) * 100
            o["fc"][name] = {"raw": _q(X[j]), "path": [round(float(v), 3) for v in med]}
            if ready[j]:
                o["fc"][name]["emos"] = _q(E[j])
    if out is None:
        return None
    cfg, df = base
    if llm_cfg_path:
        lc = load_config(llm_cfg_path, ["llm.mode=labels"])
        llm = ResultStore(root / "llm" / llm_dir(llm_tag(lc), lc, "real")).load()
        if not llm.empty:
            nc = lc["news"]
            news = load_news(nc["symbols"], pd.to_datetime(llm["origin_ts"]).min() - pd.Timedelta(days=3),
                             pd.to_datetime(llm["origin_ts"]).max() + pd.Timedelta(days=1), root / "news",
                             offline=True)
            ldf = load_data(lc, root / "data")
            for _, r in llm.iterrows():
                t = pd.Timestamp(r["origin_ts"]).strftime("%Y-%m-%d %H:%M")
                if t not in origins:
                    continue
                q = [round(float(v), 4) for v in r["q"]]
                i = int(ldf.index.get_indexer([pd.Timestamp(r["origin_ts"])])[0])
                cut = info_time(ldf.index, i, lc["timeframe"])
                h = select_headlines(news, cut, nc["lookback_hours"], n_heads)
                origins[t]["llm"] = {"up": q[0], "down": q[1], "none": q[2], "n_heads": int(r["n_heads"]),
                                     "heads": [{"text": str(x["headline"])[:220],
                                                "age_h": round((cut - x["created_at"]).total_seconds() / 3600, 1)}
                                               for _, x in h.iterrows()]}
    out["metrics"] = metrics
    out["origins"] = sorted(origins.values(), key=lambda o: o["t"])
    return out


def llm_config_for(asset: str, code_root: Path) -> str | None:
    """Find the LLM-study config that scored this asset with direction labels."""
    stem = asset.split("-")[0].lower()
    for sub in ("confirm2", "confirm", "artifact"):
        p = Path(code_root) / "configs" / sub / f"{stem}.yaml"
        if p.exists():
            return str(p)
    return None
