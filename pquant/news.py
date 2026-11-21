"""Timestamped news (Alpaca / Benzinga) and leakage-safe headline selection.

Leakage rules:
  * the information time of an origin is when the trade would be placed (info_time);
  * an article is usable only if created_at < info_time AND updated_at <= info_time
    (Alpaca returns the latest version of an article, so a later edit could carry
    post-cutoff information; such articles are dropped, not truncated).
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

ALPACA_NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
US_CLOSE_UTC_HOUR = 20  # 16:00 ET in summer; conservative (1h early) in winter


def _naive_utc(s) -> pd.Series:
    return pd.to_datetime(s, utc=True).dt.tz_convert("UTC").dt.tz_localize(None)


def fetch_alpaca_month(symbols: list[str], month_start: pd.Timestamp, key: str, secret: str,
                       session=None, pause: float = 0.35) -> pd.DataFrame:
    import requests

    sess = session or requests.Session()
    end = month_start + pd.offsets.MonthBegin(1)
    params = {"symbols": ",".join(symbols), "start": month_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
              "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"), "limit": 50, "sort": "asc"}
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    rows: list[dict] = []
    while True:
        for attempt in range(6):
            r = sess.get(ALPACA_NEWS_URL, headers=headers, params=params, timeout=30)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            break
        else:
            raise RuntimeError(f"alpaca news: repeated {r.status_code} for {month_start:%Y-%m}")
        body = r.json()
        for a in body.get("news", []):
            rows.append({"id": a["id"], "created_at": a["created_at"], "updated_at": a["updated_at"],
                         "headline": (a.get("headline") or "").strip(), "summary": (a.get("summary") or "").strip(),
                         "source": a.get("source", ""), "symbols": ",".join(a.get("symbols", []))})
        tok = body.get("next_page_token")
        if not tok:
            break
        params["page_token"] = tok
        time.sleep(pause)
    df = pd.DataFrame(rows, columns=["id", "created_at", "updated_at", "headline", "summary", "source", "symbols"])
    if len(df):
        df["created_at"] = _naive_utc(df["created_at"])
        df["updated_at"] = _naive_utc(df["updated_at"])
    return df


def load_news(symbols: list[str], start, end, cache_dir: str | Path, key: str | None = None,
              secret: str | None = None, log=print, offline: bool = False) -> pd.DataFrame:
    """Monthly-cached fetch. Months fully in the past are cached; re-runs are free.
    offline=True reads cached months only and skips the rest (no API keys needed)."""
    key = key or os.environ.get("ALPACA_API_KEY")
    secret = secret or os.environ.get("ALPACA_SECRET_KEY")
    cache = Path(cache_dir) / "_".join(sorted(symbols))
    cache.mkdir(parents=True, exist_ok=True)
    months = pd.date_range(pd.Timestamp(start).to_period("M").to_timestamp(), pd.Timestamp(end), freq="MS")
    parts = []
    for m in months:
        f = cache / f"{m:%Y-%m}.parquet"
        if not f.exists():
            if offline:
                continue
            if not (key and secret):
                raise RuntimeError("ALPACA_API_KEY / ALPACA_SECRET_KEY not set")
            df = fetch_alpaca_month(symbols, m, key, secret)
            if m + pd.offsets.MonthBegin(1) <= pd.Timestamp.utcnow().tz_localize(None):
                df.to_parquet(f, index=False)
            log(f"  news {'/'.join(symbols)} {m:%Y-%m}: {len(df)} articles")
        else:
            df = pd.read_parquet(f)
        parts.append(df)
    if not parts:
        return pd.DataFrame(columns=["id", "created_at", "updated_at", "headline", "summary", "source", "symbols"])
    out = pd.concat(parts, ignore_index=True).drop_duplicates("id")
    return out.sort_values("created_at").reset_index(drop=True)


def info_time(index: pd.DatetimeIndex, i: int, timeframe: str) -> pd.Timestamp:
    """When a position based on origin row i would be placed (all context bars closed)."""
    if timeframe.endswith("d"):  # daily equity bar stamped at midnight; context ends at prior close
        return pd.Timestamp(index[i - 1].normalize()) + pd.Timedelta(hours=US_CLOSE_UTC_HOUR)
    return pd.Timestamp(index[i])  # intraday: bar open of first forecast bar == last context close


def select_headlines(news: pd.DataFrame, cutoff: pd.Timestamp, lookback_hours: float,
                     max_items: int) -> pd.DataFrame:
    """Newest-first, de-duplicated headlines strictly available at `cutoff`."""
    if news.empty:
        return news
    c = news["created_at"].to_numpy()
    lo = np.searchsorted(c, np.datetime64(cutoff - pd.Timedelta(hours=lookback_hours)), side="left")
    hi = np.searchsorted(c, np.datetime64(cutoff), side="left")  # created_at < cutoff
    w = news.iloc[lo:hi]
    w = w[w["updated_at"] <= cutoff]
    w = w[w["headline"].str.len() > 0]
    w = w.iloc[::-1].drop_duplicates("headline")
    return w.head(max_items)
