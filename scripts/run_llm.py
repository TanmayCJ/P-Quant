"""Phase 3a: fetch news (cached) and score each forecast origin with Llama (resumable).

Needs env ALPACA_API_KEY, ALPACA_SECRET_KEY (news) and HF_TOKEN (Llama) - set them from
Colab Secrets, never in code. Several configs / variants share one loaded model.

python scripts/run_llm.py --config configs/spy_1d.yaml --root $ROOT --variant real [--dry-run] [--limit 8]
python scripts/run_llm.py --origins-only --config configs/confirm/*.yaml --variant real shuffled anon --root $ROOT
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from kronos_agent.config import load_config, run_tag
from kronos_agent.data import load_data
from kronos_agent.news import load_news
from kronos_agent.newsrun import VARIANTS, build_inputs, llm_dir, llm_tag, origins_frame, run_scoring
from kronos_agent.store import ResultStore


def prepare(config: str, overrides: list[str], root: Path, origins_only: bool):
    cfg = load_config(config, overrides)
    nc = cfg["news"]
    if not (nc["symbols"] and nc["asset_name"] and nc["horizon_text"]):
        raise SystemExit(f"{config}: needs news.symbols, news.asset_name, news.horizon_text")
    df = load_data(cfg, root / "data")
    if origins_only:
        tag, store_df = llm_tag(cfg), origins_frame(cfg, df)
    else:
        tag = run_tag(cfg)
        store_df = ResultStore(root / "paths" / tag).load()
    if store_df.empty:
        raise SystemExit(f"no origins for {tag}")
    ts = pd.to_datetime(store_df["origin_ts"])
    news = load_news(nc["symbols"], ts.min() - pd.Timedelta(days=3), ts.max() + pd.Timedelta(days=1),
                     root / "news")
    print(f"[{cfg['asset']}] news: {len(news)} articles; origins: {len(store_df)}")
    return cfg, df, store_df, news, tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--variant", nargs="+", default=["real"], choices=VARIANTS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="fetch news + print one prompt, no LLM")
    ap.add_argument("--origins-only", action="store_true",
                    help="score config-window origins without Kronos paths (confirmation study)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL")
    args = ap.parse_args()
    root = Path(args.root)

    scorer = None
    for variant in args.variant:
        for config in args.config:
            cfg, df, store_df, news, tag = prepare(config, args.set, root, args.origins_only)
            inputs = build_inputs(cfg, df, store_df, news, variant, seed=cfg["seed"])
            nh = np.array([x["n_heads"] for x in inputs])
            print(f"[{cfg['asset']}:{variant}] headlines/origin mean {nh.mean():.1f}  "
                  f"zero-headline frac {(nh == 0).mean():.2f}")
            out = root / "llm" / llm_dir(tag, cfg, variant)
            out.mkdir(parents=True, exist_ok=True)
            ex = inputs[len(inputs) // 2]
            (out / "example_prompt.json").write_text(json.dumps(ex["messages"], indent=2))
            if args.dry_run:
                for msg in ex["messages"]:
                    print(f"--- {msg['role']} ---\n{msg['content']}")
                continue
            store = ResultStore(out)
            if len(store.done_keys()) >= len(inputs) and args.limit is None:
                print(f"[{cfg['asset']}:{variant}] already complete")
                continue
            if scorer is None:
                os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # before torch
                from kronos_agent.llm import LlamaScorer

                scorer = LlamaScorer(cfg["llm"]["model_id"], token=os.environ.get("HF_TOKEN"),
                                     max_tokens=cfg["llm"]["max_tokens"])
            run_scoring(inputs, scorer, store, variant, batch=cfg["llm"]["batch"], limit=args.limit)


if __name__ == "__main__":
    main()
