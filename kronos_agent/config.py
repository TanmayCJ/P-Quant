from __future__ import annotations

from pathlib import Path

import yaml

DEFAULTS = {
    "source": "yfinance",
    "exchange": "binanceus",
    "lookback": 512,
    "pred_len": 24,
    "stride": 24,
    "n_paths": 64,
    "batch_paths": 64,
    "T": 1.0,
    "top_p": 0.9,
    "seed": 1234,
    "model_family": "kronos",   # kronos | chronos (see baselines.make_sampler)
    "model_id": "NeoQuasar/Kronos-base",
    "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
    "max_context": 512,
    "amp": False,               # fp16 autocast on CUDA (validate vs fp32 before using for results)
    # Phase 2
    "cost_bps": 10.0,           # per side
    "bars_per_year": None,      # required for annualization (8760 for 1h crypto, 252 for 1d equity)
    "calib": {"warmup": 50, "window": 250, "gamma": 0.005, "levels": [0.5, 0.8, 0.9]},
    # dist: raw|recal|spread|emos|emos_gap ; interval_method: quantile|cqr|aci (cqr/aci: raw, emos, emos_gap)
    "policy": {"level": 0.8, "dist": "emos_gap", "interval_method": "aci", "width_q": 0.5},
    "features": {"conformal": True, "abstain": True, "news": True},
    # Phase 3
    "news": {"symbols": None, "asset_name": None, "aliases": [], "horizon_text": None,
             "lookback_hours": 48, "max_headlines": 25},
    # mode: volbins | fixbins | labels (see newsrun.MODES); fixed_sigma: "median" or a number
    "llm": {"model_id": "meta-llama/Llama-3.1-8B-Instruct", "batch": 4, "max_tokens": 3072,
            "mode": "volbins", "fixed_sigma": "median"},
    # negative lambda = contrarian pooling (bet against the LLM's reading), chosen online like the rest
    "fusion": {"grid": [-2.0, -1.5, -1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
               "priors": ["kronos_raw", "kronos_emos_gap", "gauss_raw"]},
}

REQUIRED = ["asset", "symbol", "timeframe", "data_start", "data_end", "windows"]


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def _apply_override(cfg: dict, item: str) -> None:
    """'a.b=val' -> cfg['a']['b'] = yaml-parsed val."""
    key, sep, raw = item.partition("=")
    if not sep:
        raise ValueError(f"override must be key=value, got {item!r}")
    *parents, leaf = key.strip().split(".")
    node = cfg
    for p in parents:
        node = node.setdefault(p, {})
    node[leaf] = yaml.safe_load(raw)


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    with open(path) as f:
        cfg = _merge(DEFAULTS, yaml.safe_load(f))
    for item in overrides or []:
        _apply_override(cfg, item)
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise ValueError(f"{path}: missing keys {missing}")
    if cfg["lookback"] > cfg["max_context"]:
        raise ValueError("lookback exceeds Kronos max_context")
    return cfg


def run_tag(cfg: dict) -> str:
    """Output folder name; every setting that changes sampled paths is in it, so runs never mix."""
    tag = (f"{cfg['asset']}_{cfg['timeframe']}_{cfg['model_id'].split('/')[-1]}"
           f"_n{cfg['n_paths']}_L{cfg['lookback']}_T{cfg['T']}_p{cfg['top_p']}")
    return tag + ("_fp16" if cfg["amp"] else "")


def dm_horizon(cfg: dict) -> int:
    """Overlap of consecutive forecast targets, used as the DM-test / HAC horizon."""
    return max(1, -(-cfg["pred_len"] // cfg["stride"]))
