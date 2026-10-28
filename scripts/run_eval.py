"""Evaluate stored Kronos paths vs baselines, split by pre/post-cutoff window.

python scripts/run_eval.py --config configs/btc_1h.yaml --root /content/drive/MyDrive/kronos-agent
"""
import argparse
from pathlib import Path

import pandas as pd

from kronos_agent.config import dm_horizon, load_config, run_tag
from kronos_agent.data import load_data
from kronos_agent.evaluate import build_eval_frame, summarize
from kronos_agent.store import ResultStore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    root = Path(args.root)
    df = load_data(cfg, root / "data")
    tag = run_tag(cfg)
    store_df = ResultStore(root / "paths" / tag).load()
    if store_df.empty:
        raise SystemExit(f"no sampled paths for {tag}")

    ev = build_eval_frame(store_df, df, cfg["lookback"], seed=cfg["seed"])
    table, tests = summarize(ev, dm_h=dm_horizon(cfg))
    out = root / "reports" / tag
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "metrics.csv", index=False)
    tests.to_csv(out / "tests.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.precision", 4):
        print(table.to_string(index=False))
        print()
        print(tests.to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
