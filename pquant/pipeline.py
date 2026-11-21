"""Resumable sampling run over all forecast origins of a config.

Realized future values are deliberately NOT stored with the forecasts; the
evaluator joins them from raw data. Downstream fusion code therefore cannot
read targets by accident.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .data import label_period, origins_by_window
from .sampler import PATH_FIELDS, KronosSampler
from .store import ResultStore


def origin_key(asset: str, ts: pd.Timestamp, seed: int, model_id: str) -> str:
    return f"{asset}|{ts.isoformat()}|{seed}|{model_id}"


def run_sampling(cfg: dict, df: pd.DataFrame, sampler: KronosSampler, store: ResultStore,
                 limit: int | None = None, flush_every: int = 20, log=print) -> int:
    windows = origins_by_window(df, cfg)
    rows = np.unique(np.concatenate(list(windows.values())))
    done = store.done_keys()
    todo = [i for i in rows
            if origin_key(cfg["asset"], df.index[i], cfg["seed"], sampler.model_id) not in done]
    n_done = len(rows) - len(todo)
    if limit is not None:
        todo = todo[:limit]
    per_window = {k: len(v) for k, v in windows.items()}
    log(f"{cfg['asset']}: {len(rows)} origins {per_window}, {n_done} done, running {len(todo)}")

    h, L = cfg["pred_len"], cfg["lookback"]
    buf: list[dict] = []
    t0 = time.time()
    for n, i in enumerate(todo, 1):
        ts = df.index[i]
        ctx = df.iloc[i - L:i]
        y_ts = df.index[i:i + h]  # future calendar only (no values), known ex ante up to holidays
        paths = sampler.sample(ctx, y_ts, cfg["n_paths"], seed=cfg["seed"] + int(i),
                               T=cfg["T"], top_p=cfg["top_p"], batch=cfg["batch_paths"])
        buf.append({
            "key": origin_key(cfg["asset"], ts, cfg["seed"], sampler.model_id),
            "asset": cfg["asset"],
            "origin_ts": ts,
            "row": int(i),
            "period": label_period(ts, cfg["windows"]),
            "seed": int(cfg["seed"]),
            "model_id": sampler.model_id,
            "n_paths": int(cfg["n_paths"]),
            "pred_len": int(h),
            "T": float(cfg["T"]),
            "top_p": float(cfg["top_p"]),
            "last_close": float(ctx["close"].iloc[-1]),
            **{f: paths[f].ravel().tolist() for f in PATH_FIELDS},
        })
        if len(buf) >= flush_every:
            store.append(buf)
            buf = []
            rate = n / (time.time() - t0)
            log(f"  {n}/{len(todo)}  {rate:.2f} origins/s  eta {(len(todo) - n) / rate / 60:.1f} min")
    store.append(buf)
    if todo:
        log(f"done: {len(todo)} origins in {time.time() - t0:.0f}s "
            f"({(time.time() - t0) / len(todo):.2f} s/origin)")
    return len(todo)


def decode_paths(rec, field: str = "close") -> np.ndarray:
    return np.asarray(rec[field], dtype=np.float32).reshape(rec["n_paths"], rec["pred_len"])
