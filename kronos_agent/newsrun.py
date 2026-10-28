"""Build per-origin LLM prompts from leakage-safe headlines and score them (resumable).

Variants:
  real      headlines available at the origin's info time
  anon      same headlines, asset name / aliases / date removed (memorization check)
  shuffled  headlines of a different, randomly chosen origin (placebo: should not help)
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .data import label_period
from .fusion import LETTERS, bin_edges
from .llm import LABEL_LETTERS, build_label_messages, build_messages
from .news import info_time, select_headlines
from .store import ResultStore

VARIANTS = ("real", "anon", "shuffled")
# volbins: 7 ranges scaled by recent volatility; fixbins: 7 ranges with a constant per-asset
# scale; labels: up / down / no clear effect
MODES = ("volbins", "fixbins", "labels")


def origins_frame(cfg: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Forecast origins of every configured window without Kronos paths (LLM-only studies)."""
    from .data import origins_by_window
    from .pipeline import origin_key

    rows = np.unique(np.concatenate([v for v in origins_by_window(df, cfg).values()] or [np.array([], int)]))
    return pd.DataFrame({"key": [origin_key(cfg["asset"], df.index[i], cfg["seed"], "none") for i in rows],
                         "origin_ts": df.index[rows], "row": rows.astype(int), "pred_len": int(cfg["pred_len"])})


def llm_tag(cfg: dict) -> str:
    """Folder tag for LLM-only scoring runs (no Kronos settings involved)."""
    return f"{cfg['asset']}_{cfg['timeframe']}_h{cfg['pred_len']}_s{cfg['stride']}_L{cfg['lookback']}"


def horizon_sigma(close: np.ndarray, i: int, lookback: int, h: int) -> float:
    """Same volatility scale as the Gaussian baseline in evaluate.py."""
    return float(np.diff(np.log(close[i - lookback:i])).std(ddof=1) * np.sqrt(h))


def build_inputs(cfg: dict, df: pd.DataFrame, store_df: pd.DataFrame, news: pd.DataFrame,
                 variant: str = "real", seed: int = 0) -> list[dict]:
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    nc = cfg["news"]
    close = df["close"].to_numpy(float)
    recs = [r for r in store_df.to_dict("records")
            if label_period(pd.Timestamp(r["origin_ts"]), cfg["windows"]) is not None]
    cutoffs = [info_time(df.index, r["row"], cfg["timeframe"]) for r in recs]
    heads = [select_headlines(news, c, nc["lookback_hours"], nc["max_headlines"]) for c in cutoffs]
    src = np.arange(len(recs))
    if variant == "shuffled":
        rng = np.random.default_rng(seed)
        src = rng.permutation(len(recs))
        # avoid self-matches; a derangement is not required for a placebo, just no identity rows
        same = src == np.arange(len(recs))
        src[same] = np.roll(src[same], 1) if same.sum() > 1 else (src[same] + 1) % len(recs)
    mode = cfg["llm"].get("mode", "volbins")
    if mode not in MODES:
        raise ValueError(f"llm.mode must be one of {MODES}")
    sigs = np.array([horizon_sigma(close, r["row"], cfg["lookback"], r["pred_len"]) for r in recs])
    # bin scale: per-origin volatility (volbins) or ONE constant per asset (fixbins) so that the
    # ranges carry no time-varying information
    fs = cfg["llm"].get("fixed_sigma", "median")
    const = float(np.median(sigs)) if fs in (None, "median") else float(fs)
    scales = sigs if mode == "volbins" else np.full(len(sigs), const)
    anon, aliases = variant == "anon", nc.get("aliases") or []
    out = []
    for j, r in enumerate(recs):
        k = src[j]
        if mode == "labels":
            e, letters = np.array([]), LABEL_LETTERS
            msgs = build_label_messages(nc["asset_name"], nc["horizon_text"], cutoffs[k], heads[k],
                                        anonymize=anon, aliases=aliases)
        else:
            e, letters = bin_edges(np.array([scales[j]]))[0], LETTERS
            msgs = build_messages(nc["asset_name"], nc["horizon_text"], cutoffs[k], heads[k], e,
                                  anonymize=anon, aliases=aliases)
        out.append({"key": r["key"], "origin_ts": r["origin_ts"], "cutoff": cutoffs[j],
                    "n_heads": int(len(heads[k])), "sigma_h": float(sigs[j]), "scale": float(scales[j]),
                    "edges": e.tolist(), "letters": letters, "mode": mode, "messages": msgs})
    return out


def llm_dir(tag: str, cfg: dict, variant: str) -> str:
    """Folder name for one (asset settings, elicitation mode, variant) scoring run.
    volbins keeps the original naming so earlier runs remain valid."""
    mode = cfg["llm"].get("mode", "volbins")
    return f"{tag}_{variant}" if mode == "volbins" else f"{tag}_{mode}_{variant}"


def run_scoring(inputs: list[dict], scorer, store: ResultStore, variant: str, batch: int = 4,
                flush_every: int = 40, limit: int | None = None, log=print) -> int:
    done = store.done_keys()
    todo = [x for x in inputs if x["key"] not in done]
    n_done = len(inputs) - len(todo)
    if limit is not None:
        todo = todo[:limit]
    log(f"llm[{variant}]: {len(inputs)} origins, {n_done} done, running {len(todo)}")
    buf, t0 = [], time.time()
    for b in range(0, len(todo), batch):
        chunk = todo[b:b + batch]
        prompts = [scorer.render(x["messages"]) for x in chunk]
        letters = chunk[0].get("letters", LETTERS)
        probs, mass = scorer.score(prompts, letters)
        for x, p, ms, pr in zip(chunk, probs, mass, prompts):
            buf.append({"key": x["key"], "origin_ts": x["origin_ts"], "cutoff": x["cutoff"],
                        "variant": variant, "mode": x.get("mode", "volbins"), "n_heads": x["n_heads"],
                        "sigma_h": x["sigma_h"], "scale": x.get("scale", x["sigma_h"]),
                        "edges": x["edges"], "q": [float(v) for v in p], "letter_mass": float(ms),
                        "n_tokens": scorer.n_tokens(pr)})
        if len(buf) >= flush_every:
            store.append(buf)
            buf = []
            n = b + len(chunk)
            rate = n / (time.time() - t0)
            log(f"  {n}/{len(todo)}  {rate:.2f}/s  eta {(len(todo) - n) / rate / 60:.1f} min")
    store.append(buf)
    if todo:
        log(f"done: {len(todo)} in {time.time() - t0:.0f}s")
    return len(todo)
