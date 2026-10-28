"""Kronos bias diagnostic: does the tokenizer round-trip lose price level as a function of
how far the last price sits from the context-window mean?

python scripts/diagnose_kronos.py --config configs/spy_1d.yaml --root $ROOT --kronos-path $KRONOS \
    --window post_cutoff --n 60 [--set lookback=128]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from kronos_agent.config import load_config
from kronos_agent.data import COLS, load_data, origins_by_window
from kronos_agent.sampler import KronosSampler

CLOSE = COLS.index("close")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--kronos-path", required=True)
    ap.add_argument("--window", default="post_cutoff")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL")
    args = ap.parse_args()

    import torch

    cfg = load_config(args.config, args.set)
    df = load_data(cfg, Path(args.root) / "data")
    sampler = KronosSampler.from_pretrained(cfg["model_id"], cfg["tokenizer_id"], args.kronos_path,
                                            None, cfg["max_context"])
    pred = sampler.predictor
    tok, dev, clip = pred.tokenizer, pred.device, getattr(pred, "clip", 5)
    rows_all = origins_by_window(df, cfg)[args.window]
    sel = rows_all[np.linspace(0, len(rows_all) - 1, min(args.n, len(rows_all))).astype(int)]
    L = cfg["lookback"]

    out = []
    for i in sel:
        ctx = df.iloc[i - L:i][COLS].to_numpy(float)
        mean, std = ctx.mean(0), ctx.std(0)
        xn = np.clip((ctx - mean) / (std + 1e-5), -clip, clip)
        x = torch.from_numpy(xn.astype(np.float32))[None].to(dev)
        with torch.no_grad():
            rec = tok.decode(tok.encode(x, half=True), half=True)[0].float().cpu().numpy()
        rec = rec * (std + 1e-5) + mean
        c = ctx[:, CLOSE]
        out.append({
            "origin": df.index[i],
            "gap%": np.log(c[-1] / c.mean()) * 100,
            "z_last": xn[-1, CLOSE],
            "rec_err_last%": np.log(rec[-1, CLOSE] / c[-1]) * 100,
            "rec_err_avg%": np.mean(np.log(rec[:, CLOSE] / c)) * 100,
            "rec_mae%": np.mean(np.abs(np.log(rec[:, CLOSE] / c))) * 100,
        })
    d = pd.DataFrame(out)
    print(f"lookback={L}  n={len(d)}  window={args.window}")
    print(d.drop(columns="origin").describe().loc[["mean", "std", "min", "max"]].round(3).to_string())
    print("corr(rec_err_last, gap) = %.2f   slope = %.3f" % (
        np.corrcoef(d["rec_err_last%"], d["gap%"])[0, 1], np.polyfit(d["gap%"], d["rec_err_last%"], 1)[0]))


if __name__ == "__main__":
    main()
