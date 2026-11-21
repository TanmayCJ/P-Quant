"""Market data loading, caching and forecast-origin construction.

All timestamps are tz-naive UTC and denote bar *open* time. For an origin at
row i, the context is rows [i - lookback, i) and the target is rows
[i, i + pred_len). Information time of the origin is index[i]: every context
bar has closed by then.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OHLCV = ["open", "high", "low", "close", "volume"]
COLS = OHLCV + ["amount"]

# Training-data cutoffs (first timestamp NOT seen in training).
# Kronos: paper (arXiv 2508.02739) states pretraining data extends to June 2024.
# Llama-3.1: model card states pretraining data cutoff December 2023.
CUTOFFS = {
    "kronos": pd.Timestamp("2024-07-01"),
    "llama-3.1": pd.Timestamp("2024-01-01"),
}


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[OHLCV].astype("float64")
    df = df[df["close"] > 0].copy()
    # Approximate turnover; Kronos was pretrained with an amount channel.
    df["amount"] = df["volume"] * df[["open", "high", "low", "close"]].mean(axis=1)
    df.index.name = "timestamp"
    return df


def fetch_ccxt(symbol: str, timeframe: str, start: str, end: str,
               exchange: str = "binanceus", limit: int = 1000) -> pd.DataFrame:
    import ccxt

    ex = getattr(ccxt, exchange)({"enableRateLimit": True})
    since = int(pd.Timestamp(start).timestamp() * 1000)
    end_ms = int(pd.Timestamp(end).timestamp() * 1000)
    rows: list = []
    while since < end_ms:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not batch:
            break
        rows.extend(batch)
        nxt = batch[-1][0] + 1
        if nxt <= since:
            break
        since = nxt
    if not rows:
        raise RuntimeError(f"no data from {exchange} for {symbol} {timeframe}")
    df = pd.DataFrame(rows, columns=["ts", *OHLCV])
    df.index = pd.to_datetime(df.pop("ts"), unit="ms")
    return _finalize(df[df.index < pd.Timestamp(end)])


def fetch_yfinance(ticker: str, interval: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, interval=interval,
                     auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"no data from yfinance for {ticker} {interval}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    df.index = idx
    return _finalize(df)


def load_data(cfg: dict, cache_dir: str | Path) -> pd.DataFrame:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{cfg['asset']}_{cfg['timeframe']}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    if cfg["source"] == "ccxt":
        df = fetch_ccxt(cfg["symbol"], cfg["timeframe"], cfg["data_start"],
                        cfg["data_end"], cfg["exchange"])
    elif cfg["source"] == "yfinance":
        df = fetch_yfinance(cfg["symbol"], cfg["timeframe"], cfg["data_start"], cfg["data_end"])
    else:
        raise ValueError(f"unknown source {cfg['source']}")
    df.to_parquet(path)
    return df


def make_origins(index: pd.DatetimeIndex, lookback: int, pred_len: int, stride: int,
                 start, end) -> np.ndarray:
    """Row positions i with a full context and full target, origin in [start, end)."""
    rows = np.arange(lookback, len(index) - pred_len + 1)
    ts = index[rows]
    mask = (ts >= pd.Timestamp(start)) & (ts < pd.Timestamp(end))
    return rows[mask][::stride]


def origins_by_window(df: pd.DataFrame, cfg: dict) -> dict[str, np.ndarray]:
    return {
        name: make_origins(df.index, cfg["lookback"], cfg["pred_len"], cfg["stride"], s, e)
        for name, w in cfg["windows"].items() if w is not None  # null disables a window
        for s, e in [w]
    }


def label_period(ts: pd.Timestamp, windows: dict) -> str | None:
    for name, w in windows.items():
        if w is not None and pd.Timestamp(w[0]) <= ts < pd.Timestamp(w[1]):
            return name
    return None
